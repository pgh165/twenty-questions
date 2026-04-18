"""스무고개 턴 진행 로직 + 프롬프트 구성.

Ollama 호출은 ollama_client.py에 격리되어 있고, 여기서는 프롬프트 빌드 +
응답 파싱 방어 로직만 담당한다.
"""
import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor

from .models import GameSession
from .ollama_client import OllamaError, call_ollama, get_embedding

log = logging.getLogger(__name__)

# 최대 턴. 초과 시 강제 추측 모드 진입.
MAX_TURNS = 20

# JSON 파싱 실패 또는 응답 형식 오류 시 복구용 기본 질문.
FALLBACK_QUESTION = "혹시 그 대상은 실존 인물인가요?"

# 질문에 섞이면 정보 이득이 떨어지는 금지 표현. 생성된 질문에서 발견되면 재생성.
BANNED_ASK_PHRASES = ("특정", "관련 있", "관련있", "관련 된", "관련된", "관련이 있")

# 추측(guess) answer에서 발견되면 복수/모호로 간주하고 재생성.
BANNED_GUESS_PHRASES = (",", " 또는 ", " / ", "중 한", " 중 ", "같은", "비슷한", "등의")

# 재생성 최대 시도 횟수. 이만큼 해도 위반이면 마지막 결과를 그대로 사용.
MAX_REGENERATE = 2

# 매 턴 유지할 후보군 상한. 너무 크면 프롬프트가 부풀고, 작으면 LLM 탐색 공간이 막힘.
MAX_CANDIDATES = 10

# Best-of-N: 매 턴 N개의 질문 후보를 병렬 생성 → 2차 호출로 변별력 가장 높은 것 선택.
BEST_OF_N = 3
# N개 생성 시 다양성 확보를 위해 평소보다 temperature를 높임.
BEST_OF_GEN_TEMPERATURE = 0.6
# 선택기는 결정론적으로 동작해야 함.
SELECTOR_TEMPERATURE = 0.1

# 단일 경로(Best-of-N 미적용 또는 폴백)의 기본 temperature.
# 창의성보다 규칙 준수가 중요한 태스크라 0.15로 낮게 설정.
DEFAULT_TEMPERATURE = 0.15

# 의미적 중복 판정 임계값. 코사인 유사도 >= 이 값이면 과거 질문과 같은 속성으로 간주.
# bge-m3 + 짧은 한국어 질문 기준: 반대 개념(육식/초식 ~0.87)과 하위 분류(개과/고양이과 ~0.91)도
# 구조적으로 유사하게 측정되므로 0.92로 올려 false-positive를 줄인다.
SIMILARITY_THRESHOLD = 0.92

SELECTOR_SYSTEM_PROMPT = """당신은 스무고개 게임의 '질문 선택기'입니다.
현재 남은 후보 리스트와 여러 질문 후보가 주어집니다. 각 질문에 대해 '예'라고 답할 후보의 \
비율을 추정하고, 50%에 가장 가까운(=정보 이득이 최대인) 질문의 번호를 선택합니다.

규칙:
1. 반드시 JSON으로만 출력: {"selected_index": <정수>}
2. selected_index는 1부터 시작하는 질문 번호입니다.
3. 설명, 서론, 마크다운 없이 JSON 한 줄만.
"""

CANDIDATE_SYSTEM_PROMPT = """당신은 스무고개 게임의 '후보 추정기'입니다.
주어진 카테고리와 지금까지의 질문-답변 이력을 바탕으로, 사용자가 생각했을 가능성이 \
가장 높은 대상 후보를 최대 {max_n}개 추정합니다.

규칙:
1. 반드시 JSON으로만 출력: {{"candidates": ["후보1", "후보2", ...]}}
2. 각 후보는 단일 고유명사 하나 (예: "아인슈타인"). "아인슈타인 같은 과학자", "뉴턴 또는 \
아인슈타인" 등 복합 표현 금지.
3. 이미 '아니오'로 답변된 속성을 가진 대상은 반드시 제외하세요.
4. '예'로 답변된 속성을 모두 만족하는 대상만 포함하세요.
5. 확실한 후보가 적으면 리스트를 짧게 유지해도 됩니다. 빈 리스트도 허용.
6. 설명, 서론, 마크다운 없이 JSON 한 줄만.
""".format(max_n=MAX_CANDIDATES)

SYSTEM_PROMPT = """당신은 '스무고개 출제자'입니다. 사용자가 머릿속으로 한 가지 대상을 생각하고, \
당신은 예/아니오 질문만으로 그 대상을 맞춰야 합니다.

규칙:
1. 매 턴 정확히 하나의 질문을 하거나, 충분히 좁혀졌을 때만 추측합니다.
2. 사용자의 답은 "예", "아니오", "잘 모름" 세 가지뿐입니다. 이 셋 중 하나로 답할 수 있는 단일 질문만 하세요.
3. "A 또는 B?", "A, B, C 중 무엇?", "~은 무엇인가요?" 같은 선택형/주관식 질문은 절대 금지.
4. 넓은 범주 → 구체적 특징 순으로 좁혀갑니다.
5. 이전 턴에서 이미 나온 질문, 또는 **같은 속성을 다른 말로 재확인하는 질문은 절대 금지**. 직전 대화 기록을 반드시 검토하세요.
   다음은 모두 **같은 축**으로 간주되며, 한 번이라도 답이 나왔으면 다시 물을 수 없습니다:
   - "학원물 배경인가요?" ≡ "학교 생활이 배경?" ≡ "교육기관 배경?" ≡ "학창시절 이야기?" ≡ "학생 신분으로 등장?"
   - "인간인가요?" ≡ "인간의 모습을 하고 있나요?" ≡ "외형이 인간형?" ≡ "인간이 아닌 종족?"
   - "과학자인가요?" ≡ "과학 분야에서 활동?" ≡ "연구자?"
   - "애니메이션에 등장?" ≡ "만화·애니메이션 매체?" ≡ "시각 예술 매체로 알려진 캐릭터?"
   한 축이 이미 '예' 또는 '아니오'로 답해졌다면, **그 축은 완전히 종료된 것으로 간주하고 다른 축으로 이동**하세요. 표현만 바꿔서 다시 확인하는 행위는 턴 낭비입니다.
6. **정보 이득이 큰 질문만** 하세요. 남은 후보를 대략 절반으로 가를 수 있는 질문이 이상적입니다. 아래 모호한 표현은 금지:
   - "주로 ~" ("주로 예술 분야인가요?" 등 → 그냥 "예술가인가요?"로)
   - "특정 ~" ("특정 시대의 인물인가요?", "특정 국가 출신인가요?" 등 → 동어반복)
   - "~과 관련이 있나요?", "~와 관련된 일을 하나요?" (약한 연관)
   - 이미 '예'로 답한 사실을 돌려 묻기 (예: "역사적 인물=예" 이후 "과거에 살았던 인물인가요?" 금지)
   - 거의 모든 대상에 '예'가 나오는 질문 (예: "이름이 있나요?")
7. 추측(guess)의 answer는 반드시 **단일 고유명사 하나**여야 합니다. 아래는 모두 금지:
   - "아인슈타인 또는 뉴턴" (여러 후보)
   - "아인슈타인, 뉴턴, 갈릴레이 중 한 명" (열거)
   - "유명한 과학자 중 한 명" (일반화)
   - "아인슈타인 같은 인물" (모호)
   확신이 서지 않으면 guess 하지 말고 ask로 더 좁히세요.
8. 설명이나 서론 없이, 반드시 JSON 한 줄로만 출력합니다.

출력 형식:
- 질문: {"action": "ask", "question": "..."}
- 추측: {"action": "guess", "answer": "..."}

좋은 예:
{"action": "ask", "question": "이 인물은 과학자인가요?"}
{"action": "guess", "answer": "아인슈타인"}

나쁜 예 (절대 따라하지 말 것):
{"action": "ask", "question": "직업이 과학자인가요, 아니면 예술가인가요?"}  ← 선택형
{"action": "ask", "question": "이 인물은 주로 예술 분야와 관련이 있나요?"}  ← 모호어 "주로/관련"
{"action": "ask", "question": "이 인물은 특정 시대의 인물인가요?"}  ← "특정" 동어반복
{"action": "guess", "answer": "아인슈타인 또는 뉴턴"}  ← 복수 후보

다른 키, 다른 필드, 추가 텍스트, 마크다운 코드블록은 절대 포함하지 마세요."""


# 카테고리별 질문 우선순위. 넓은 축 → 좁은 축 순. 앞 축이 결정되기 전에 뒤 축을 먼저
# 묻는 난잡한 순서를 방지. 키는 session.category와 정확히 일치해야 함.
# 인물은 "실존 여부"에 따라 이후 유용한 축이 크게 달라지므로 분기 구조 사용.
CATEGORY_ROADMAPS: dict[str, dict[str, list[str]] | list[str]] = {
    "인물": {
        # 아직 실존 여부가 결정되지 않은 상태. 1번 축을 먼저 묻게 유도.
        "default": [
            "실존 인물인가 vs 가상/허구 캐릭터인가  ← 이것을 가장 먼저 물어 분기를 결정하세요",
            "성별 (남성/여성)",
        ],
        # 실존 = 예
        "real": [
            "성별 (남성/여성)",
            "생존 여부 (현재 생존 vs 이미 사망)",
            "주 활동 분야 (과학/정치/예술·연예/스포츠/기업·경영/군사·종교 등)",
            "활동 시대 (현대 ≈ 1950년 이후 / 근대 / 고대·중세)",
            "출신 대륙·국가 (아시아/유럽/아메리카/기타)",
            "대중적 인지도, 외형, 대표작 등 구체 특징",
        ],
        # 실존 = 아니오 (가상 캐릭터)
        "fictional": [
            "성별 (남성/여성 / 해당 없음)",
            "등장 매체 (만화·애니메이션 / 게임 / 영화·드라마 / 소설·웹툰)",
            "장르 (액션·배틀 / 학원·일상 / 판타지·SF / 로맨스 / 호러·스릴러)",
            "외형 (인간형 vs 비인간형 — 동물·로봇·몬스터 등)",
            "작품 내 역할 (주인공 / 조연 / 악역)",
            "출신 문화권 (일본 / 서양 / 한국 / 기타)",
            "구체 특징 (대표 능력·외모·대사 등)",
        ],
    },
    "동물": [
        "분류군 (포유류/조류/어류/파충류/양서류/곤충/무척추)",
        "서식 환경 (육상/수중/비행)",
        "식성 (육식/초식/잡식)",
        "대략적 크기 (사람보다 큰가)",
        "가축·반려 vs 야생",
        "주 서식 지역",
        "구체 종 특징",
    ],
    "사물": [
        "자연물 vs 인공물",
        "대략적 크기 (손에 쥘 수 있나 / 사람보다 큰가)",
        "용도 분류 (도구/가구/전자기기/식품/교통수단/의류 등)",
        "주 사용 장소 (실내/실외)",
        "전원·동력 필요 여부",
        "구체 특징",
    ],
}


def _pick_roadmap(session: GameSession) -> list[str] | None:
    """세션 카테고리 + Q&A 이력을 보고 적절한 로드맵을 반환.

    인물 카테고리는 '실존' 키워드가 포함된 질문의 답변을 근거로 real/fictional 분기.
    '잘 모름' 또는 아직 묻지 않았으면 default(실존 질문 먼저 유도).
    """
    raw = CATEGORY_ROADMAPS.get(session.category)
    if raw is None:
        return None
    if isinstance(raw, list):
        return raw

    # dict(분기형) — 현재는 '인물'만 해당.
    branch = "default"
    for qa in session.qas.all():
        if "실존" in qa.question:
            if qa.answer == "예":
                branch = "real"
            elif qa.answer == "아니오":
                branch = "fictional"
            break
    return raw.get(branch) or raw.get("default")


def _history_block(session: GameSession) -> list[str]:
    """카테고리 + Q&A 이력을 공통 프롬프트 조각으로 직렬화."""
    qas = list(session.qas.all())
    lines = [f"카테고리: {session.category}"]

    if qas:
        lines.append("")
        lines.append("지금까지의 질문-답변:")
        for qa in qas:
            lines.append(f"  Q{qa.turn}: {qa.question}")
            lines.append(f"  A{qa.turn}: {qa.answer}")

    return lines


def _build_candidate_prompt(session: GameSession) -> str:
    """후보군 추정용 user 프롬프트. 시스템 프롬프트는 CANDIDATE_SYSTEM_PROMPT."""
    lines = _history_block(session)
    lines.append("")
    lines.append(
        f"위 조건을 모두 만족하는 후보를 최대 {MAX_CANDIDATES}개 JSON으로 출력하세요."
    )
    return "\n".join(lines)


def _build_user_prompt(session: GameSession, force_guess: bool = False) -> str:
    """대화 이력 + 현재 후보군 + 지시사항을 문자열로 직렬화."""
    lines = _history_block(session)
    qas = list(session.qas.all())

    if qas:
        lines.append("")
        lines.append(
            "⚠ 위 Q1~Q{} 중 어느 하나와 같거나 의미가 겹치는 질문을 다시 하면 실패입니다. "
            "반드시 새로운 속성을 묻는 질문을 하세요.".format(len(qas))
        )
    else:
        lines.append("")
        lines.append("아직 아무 질문도 하지 않았습니다. 첫 질문을 해주세요.")

    roadmap = _pick_roadmap(session)
    if roadmap:
        lines.append("")
        lines.append(
            f"[{session.category} 카테고리 질문 우선순위 - 위에서 아래 순으로 좁혀가세요]"
        )
        for i, axis in enumerate(roadmap, 1):
            lines.append(f"  {i}. {axis}")
        lines.append(
            "규칙: 위 축 중 아직 결정되지 않은(이력에 '예/아니오'로 답이 나오지 않은) "
            "가장 앞 순위의 축을 먼저 묻습니다. 앞 축이 결정되기 전에 뒤 축으로 건너뛰지 마세요. "
            "모든 축이 결정되었거나 후보가 1~2개로 좁혀지면 구체 특징으로 진행하거나 guess."
        )

    candidates = session.candidates or []
    if candidates:
        lines.append("")
        lines.append(
            f"[현재 남은 후보 추정 {len(candidates)}개]: "
            + ", ".join(candidates)
        )
        lines.append(
            "이 후보들을 가능한 한 절반씩 '예/아니오'로 가를 수 있는 질문을 선택하세요. "
            "한쪽에만 치우친(거의 모두 예, 또는 거의 모두 아니오) 질문은 정보 이득이 낮습니다."
        )

    lines.append("")
    if force_guess:
        lines.append(
            f"[강제 추측 모드] 이미 {MAX_TURNS}턴이 소진되었습니다. "
            "반드시 action=\"guess\"로, 지금까지의 답변을 바탕으로 한 최선의 추측을 하나만 내세요."
        )
    else:
        lines.append(
            "다음 행동을 결정하세요: 더 좁히고 싶으면 action=\"ask\"로 질문, "
            "충분히 확신이 들면 action=\"guess\"로 추측."
        )

    return "\n".join(lines)


def _parse_candidates(raw: str) -> list[str]:
    """후보 추정 응답 파싱. 실패/형식 오류 시 빈 리스트."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("후보 응답 JSON 파싱 실패: %r", raw[:200])
        return []

    if not isinstance(data, dict):
        return []

    raw_list = data.get("candidates")
    if not isinstance(raw_list, list):
        return []

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw_list:
        if not isinstance(item, str):
            continue
        name = item.strip()
        # 복합 표현이 섞여 들어오면 버림 (BANNED_GUESS_PHRASES와 같은 정책)
        if not name or any(p in name for p in BANNED_GUESS_PHRASES):
            continue
        if name in seen:
            continue
        seen.add(name)
        cleaned.append(name)
        if len(cleaned) >= MAX_CANDIDATES:
            break
    return cleaned


def _update_candidates(session: GameSession) -> list[str]:
    """LLM을 호출해 현재 세션의 후보군을 갱신하고 DB에 저장."""
    prompt = _build_candidate_prompt(session)
    try:
        raw = call_ollama(
            prompt=prompt,
            system=CANDIDATE_SYSTEM_PROMPT,
            temperature=0.2,
        )
    except OllamaError as e:
        log.warning("후보 추정 호출 실패, 기존 후보 유지: %s", e)
        return list(session.candidates or [])

    candidates = _parse_candidates(raw)
    session.candidates = candidates
    session.save(update_fields=["candidates"])
    return candidates


def _parse_response(raw: str) -> dict:
    """LLM의 JSON 응답을 파싱. 실패 시 FALLBACK_QUESTION으로 복구."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("Ollama 응답 JSON 파싱 실패: %r", raw[:200])
        return {"action": "ask", "question": FALLBACK_QUESTION}

    if not isinstance(data, dict):
        log.warning("Ollama 응답이 객체가 아님: %r", data)
        return {"action": "ask", "question": FALLBACK_QUESTION}

    action = data.get("action")
    if action == "ask":
        question = data.get("question")
        if isinstance(question, str) and question.strip():
            return {"action": "ask", "question": question.strip()}
    elif action == "guess":
        answer = data.get("answer")
        if isinstance(answer, str) and answer.strip():
            return {"action": "guess", "answer": answer.strip()}

    log.warning("Ollama 응답 형식 오류: %r", data)
    return {"action": "ask", "question": FALLBACK_QUESTION}


def _cosine(a: list[float], b: list[float]) -> float:
    """두 벡터의 코사인 유사도. 길이 불일치/영벡터는 0.0."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _past_question_embeddings(session: GameSession) -> list[list[float]]:
    """세션의 과거 질문 임베딩을 모아 반환. 누락된 것은 지금 채워서 저장."""
    embeddings: list[list[float]] = []
    for qa in session.qas.all():
        if qa.question_embedding:
            embeddings.append(qa.question_embedding)
            continue
        try:
            emb = get_embedding(qa.question)
        except OllamaError as e:
            log.warning("과거 질문 임베딩 실패(turn=%d): %s", qa.turn, e)
            continue
        qa.question_embedding = emb
        qa.save(update_fields=["question_embedding"])
        embeddings.append(emb)
    return embeddings


def _is_semantic_duplicate(
    question: str,
    past_embeddings: list[list[float]],
    past_questions: list[str] | None = None,
) -> bool:
    """질문이 과거 질문 중 어느 하나와 유사도 임계값 이상이면 True.

    진단 로그: 매 호출마다 과거 질문별 유사도 전체 + 최대값 + 임계 초과 여부를
    INFO 레벨로 기록해서, 임계값 조정 / 임베딩 모델 교체 판단 근거로 사용.
    """
    if not past_embeddings:
        return False
    try:
        emb = get_embedding(question)
    except OllamaError as e:
        log.warning("질문 임베딩 호출 실패, 중복 탐지 스킵: %s", e)
        return False

    sims = [_cosine(emb, past) for past in past_embeddings]
    max_sim = max(sims) if sims else 0.0
    is_dup = max_sim >= SIMILARITY_THRESHOLD

    # 진단 로그: 과거 질문별 유사도를 나열. past_questions가 있으면 짝지어 출력.
    if past_questions and len(past_questions) == len(sims):
        pairs = ", ".join(
            f"{q!r}={s:.3f}" for q, s in zip(past_questions, sims)
        )
    else:
        pairs = ", ".join(f"{s:.3f}" for s in sims)
    log.info(
        "[dedup] q=%r max_sim=%.3f threshold=%.3f dup=%s | sims: %s",
        question,
        max_sim,
        SIMILARITY_THRESHOLD,
        is_dup,
        pairs,
    )
    return is_dup


def _generate_n_asks(
    session: GameSession,
    n: int,
    past_embeddings: list[list[float]],
    past_questions: list[str] | None = None,
) -> list[str]:
    """N개의 질문을 병렬 생성. 파싱 실패/금지표현/guess/의미적 중복은 버림."""
    base = _build_user_prompt(session, force_guess=False)
    prompt = (
        base
        + "\n\n[이번 호출 제약] 반드시 action=\"ask\"만 출력하세요. "
        "후보가 남아있으므로 guess는 이 호출에서는 금지입니다."
    )

    def _single() -> str | None:
        try:
            raw = call_ollama(
                prompt=prompt,
                system=SYSTEM_PROMPT,
                temperature=BEST_OF_GEN_TEMPERATURE,
            )
        except OllamaError as e:
            log.warning("[best-of-N] 호출실패: %s", e)
            return None
        log.info("[best-of-N] raw 응답: %r", raw[:300])
        parsed = _parse_response(raw)
        if parsed["action"] != "ask":
            log.info("[best-of-N] 탈락(guess/형식오류): %r", parsed)
            return None
        violation = _find_violation(parsed)
        if violation is not None:
            log.info(
                "[best-of-N] 탈락(금지표현 %r): %r", violation, parsed["question"]
            )
            return None
        if _is_semantic_duplicate(
            parsed["question"], past_embeddings, past_questions
        ):
            log.info("[best-of-N] 탈락(의미중복): %r", parsed["question"])
            return None
        log.info("[best-of-N] 통과: %r", parsed["question"])
        return parsed["question"]

    with ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(lambda _i: _single(), range(n)))

    questions = [q for q in results if q]
    # 완전 중복 제거, 순서 유지
    return list(dict.fromkeys(questions))


def _select_best_question(candidates: list[str], questions: list[str]) -> str:
    """후보군 기준 변별력 가장 높은 질문을 LLM에게 선택시킴. 실패 시 첫 번째."""
    if len(questions) == 1:
        return questions[0]

    numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    prompt = (
        f"카테고리 후보({len(candidates)}개): {', '.join(candidates)}\n\n"
        f"질문 후보:\n{numbered}\n\n"
        "각 질문에 대해 위 후보 중 '예'라고 답할 비율을 머릿속으로 계산하고, "
        f"50%에 가장 가까운 질문의 번호(1~{len(questions)})를 selected_index로 출력하세요."
    )

    try:
        raw = call_ollama(
            prompt=prompt,
            system=SELECTOR_SYSTEM_PROMPT,
            temperature=SELECTOR_TEMPERATURE,
        )
    except OllamaError as e:
        log.warning("질문 선택 호출 실패, 첫 번째 질문 사용: %s", e)
        return questions[0]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("질문 선택 JSON 파싱 실패: %r", raw[:200])
        return questions[0]

    idx = data.get("selected_index") if isinstance(data, dict) else None
    if isinstance(idx, int) and 1 <= idx <= len(questions):
        return questions[idx - 1]

    log.warning("질문 선택 인덱스 이상: %r", data)
    return questions[0]


def _find_violation(parsed: dict) -> str | None:
    """위반 표현이 있으면 해당 표현을 반환, 없으면 None."""
    if parsed["action"] == "ask":
        for p in BANNED_ASK_PHRASES:
            if p in parsed["question"]:
                return p
    elif parsed["action"] == "guess":
        for p in BANNED_GUESS_PHRASES:
            if p in parsed["answer"]:
                return p
    return None


def next_turn(session: GameSession) -> dict:
    """세션 상태 → LLM 호출 → 파싱된 액션 반환.

    위반 표현 감지 시 최대 MAX_REGENERATE번 재생성. 재시도에는 이전 위반 내용을
    프롬프트에 명시해 같은 실수를 피하도록 유도.

    반환값:
        {"action": "ask",   "question": "..."}  또는
        {"action": "guess", "answer":   "..."}
    """
    turn_count = session.qas.count()
    force_guess = turn_count >= MAX_TURNS

    # 후보군은 강제 추측 모드에서도 유용 (guess 정확도 향상). 첫 턴은 이력이 없어 스킵.
    if turn_count > 0:
        _update_candidates(session)

    # 과거 질문 임베딩 + 질문 텍스트는 Best-of-N 경로 + 단일 경로 모두에서 쓰임.
    if turn_count > 0:
        past_embeddings = _past_question_embeddings(session)
        past_questions = [qa.question for qa in session.qas.all()]
        # 임베딩 계산이 일부 실패했을 수 있으므로 길이를 맞춰 잘라냄.
        past_questions = past_questions[: len(past_embeddings)]
    else:
        past_embeddings = []
        past_questions = []

    # Best-of-N 분기: 후보군이 2개 이상 있고 강제 추측이 아닐 때만 적용.
    # 후보가 0~1개면 분할할 대상이 없어 의미 없음.
    candidates = list(session.candidates or [])
    if not force_guess and len(candidates) >= 2:
        questions = _generate_n_asks(
            session, BEST_OF_N, past_embeddings, past_questions
        )
        if questions:
            best = _select_best_question(candidates, questions)
            return {"action": "ask", "question": best}
        # N개 모두 실패하면 단일 경로로 폴백.
        log.info("Best-of-N 전원 실패, 단일 생성 경로로 폴백")

    base_prompt = _build_user_prompt(session, force_guess=force_guess)

    rejected: list[str] = []
    parsed: dict = {"action": "ask", "question": FALLBACK_QUESTION}

    for attempt in range(MAX_REGENERATE + 1):
        prompt = base_prompt
        if rejected:
            prompt += (
                "\n\n[직전 시도가 거부되었습니다. 같은 실수를 반복하지 마세요]\n"
                + "\n".join(rejected)
            )

        try:
            raw = call_ollama(
                prompt=prompt, system=SYSTEM_PROMPT, temperature=DEFAULT_TEMPERATURE
            )
        except OllamaError as e:
            log.error("Ollama 호출 실패, fallback 사용: %s", e)
            return {"action": "ask", "question": FALLBACK_QUESTION}

        parsed = _parse_response(raw)
        violation = _find_violation(parsed)

        reject_reason: str | None = None
        if violation is not None:
            reject_reason = f'금지 표현 "{violation}" 포함'
        elif (
            parsed["action"] == "ask"
            and past_embeddings
            and _is_semantic_duplicate(
                parsed["question"], past_embeddings, past_questions
            )
        ):
            reject_reason = "과거 질문과 의미가 중복됨"

        if reject_reason is None:
            return parsed

        key = "question" if parsed["action"] == "ask" else "answer"
        rejected.append(
            f'- 거부됨 ({parsed["action"]}): "{parsed[key]}" — {reject_reason}'
        )
        log.info("재생성 %d/%d: %s", attempt + 1, MAX_REGENERATE, rejected[-1])

    log.warning("재생성 %d회 모두 위반, 마지막 결과 통과: %s", MAX_REGENERATE, parsed)
    return parsed
