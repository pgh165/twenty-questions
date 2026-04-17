"""아키네이터 턴 진행 로직 + 프롬프트 구성.

Ollama 호출은 ollama_client.py에 격리되어 있고, 여기서는 프롬프트 빌드 +
응답 파싱 방어 로직만 담당한다.
"""
import json
import logging

from .models import GameSession
from .ollama_client import OllamaError, call_ollama

log = logging.getLogger(__name__)

# 최대 턴. 초과 시 강제 추측 모드 진입.
MAX_TURNS = 20

# JSON 파싱 실패 또는 응답 형식 오류 시 복구용 기본 질문.
FALLBACK_QUESTION = "혹시 그 대상은 실존 인물인가요?"

SYSTEM_PROMPT = """당신은 '아키네이터'입니다. 사용자가 머릿속으로 한 가지 대상을 생각하고, \
당신은 예/아니오 질문만으로 그 대상을 맞춰야 합니다.

규칙:
1. 매 턴 정확히 하나의 질문을 하거나, 자신 있으면 추측합니다.
2. 사용자의 답은 "예", "아니오", "잘 모름" 세 가지뿐입니다. 이에 맞는 질문만 하세요.
3. 넓은 범주(살아있는가? 인물인가? 등) → 구체적 특징(특정 국가/시대/직업 등) 순서로 좁혀갑니다.
4. 이전 질문과 의미가 겹치는 질문을 하지 마세요.
5. 설명이나 서론 없이, 반드시 아래 JSON 형식 중 하나로만 출력합니다.

출력 형식:
- 질문할 때: {"action": "ask", "question": "..."}
- 추측할 때: {"action": "guess", "answer": "..."}

다른 키, 다른 필드, 추가 텍스트는 절대 포함하지 마세요."""


def _build_user_prompt(session: GameSession, force_guess: bool = False) -> str:
    """대화 이력 + 현재 지시사항을 문자열로 직렬화."""
    qas = list(session.qas.all())
    lines = [f"카테고리: {session.category}"]

    if qas:
        lines.append("")
        lines.append("지금까지의 질문-답변:")
        for qa in qas:
            lines.append(f"  Q{qa.turn}: {qa.question}")
            lines.append(f"  A{qa.turn}: {qa.answer}")
    else:
        lines.append("")
        lines.append("아직 아무 질문도 하지 않았습니다. 첫 질문을 해주세요.")

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


def next_turn(session: GameSession) -> dict:
    """세션 상태 → LLM 호출 → 파싱된 액션 반환.

    반환값:
        {"action": "ask",   "question": "..."}  또는
        {"action": "guess", "answer":   "..."}
    """
    turn_count = session.qas.count()
    force_guess = turn_count >= MAX_TURNS

    prompt = _build_user_prompt(session, force_guess=force_guess)

    try:
        raw = call_ollama(prompt=prompt, system=SYSTEM_PROMPT, temperature=0.7)
    except OllamaError as e:
        log.error("Ollama 호출 실패, fallback 사용: %s", e)
        return {"action": "ask", "question": FALLBACK_QUESTION}

    return _parse_response(raw)
