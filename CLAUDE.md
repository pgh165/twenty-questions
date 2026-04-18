# Twenty Questions (Django + Ollama + Docker)

## 프로젝트 목표
사용자가 마음속으로 생각한 대상(인물/동물/사물 등)을 LLM이 예/아니오 질문으로 맞추는 Twenty Questions 챗봇. Django 웹 앱에서 로컬 Ollama LLM과 연동하여 대화형 추론을 수행한다.

## 기술 스택
- **Backend**: Django 5.x, SQLite
- **LLM**: Ollama (컨테이너), 기본 모델 `llama3.1:8b` (한국어 실험 시 `qwen2.5:7b` 또는 `EEVE-Korean-10.8B`로 교체 예정)
- **Frontend**: 최소 HTML + fetch API (추후 React로 확장 가능)
- **배포/개발 환경**: Docker Compose (WSL2 Ubuntu, NVIDIA GPU passthrough)
- **Python**: 3.11+

## 개발 환경 규칙 (중요)
이 프로젝트는 **전적으로 Docker Compose로 관리**한다. 호스트에서 직접 `python manage.py`나 `pip install`을 실행하지 않는다.

**올바른 명령 패턴**:
```bash
docker compose exec web python manage.py <command>
docker compose exec web pip install <package>    # 이후 requirements.txt에 반영
docker compose exec ollama ollama pull <model>
```

**잘못된 패턴 (하지 말 것)**:
```bash
python manage.py migrate        # ❌ 호스트에서 직접 실행
pip install django             # ❌ venv 만들어 호스트에 설치
```

의존성 추가 시 반드시 `requirements.txt` 업데이트 → `docker compose up -d --build`로 재빌드.

## 폴더 구조
```
twenty-questions/
├── .dockerignore
├── .env.example
├── .env                         # .gitignore됨, .env.example 복사해서 사용
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── CLAUDE.md                    # 이 파일
├── manage.py
├── tq_project/                  # Django 프로젝트 설정
│   ├── settings.py              # 환경변수 기반 설정
│   ├── urls.py
│   └── wsgi.py
└── game/                        # 메인 앱
    ├── models.py                # GameSession, QA
    ├── views.py                 # start_game, answer 엔드포인트
    ├── urls.py
    ├── ollama_client.py         # Ollama API 호출 격리
    ├── akinator_logic.py        # 프롬프트 구성 + 턴 진행 로직
    └── templates/game/index.html
```

## 핵심 설계 결정

### 1. Stateless LLM 대응
Ollama 호출은 매번 독립적이므로, **매 턴마다 전체 대화 이력을 프롬프트에 포함**해서 전송한다. 세션 상태는 Django DB(SQLite)에 저장.

### 2. JSON 강제 출력
Ollama의 `format: "json"` 옵션을 사용해 LLM 응답을 JSON으로 강제한다:
```json
{"action": "ask", "question": "..."}
{"action": "guess", "answer": "..."}
```
파싱 실패 시 fallback 질문으로 복구하는 방어 로직 포함.

### 3. 턴 제한
최대 20턴. 초과 시 프롬프트에 "반드시 guess로 답하라" 지시를 넣어 강제 추측 모드 진입. 무한 질문 루프 방지.

### 4. 카테고리 고정
세션 시작 시 카테고리(인물/동물/사물/브랜드 등)를 고정하여 탐색 공간을 좁힌다.

### 5. 지표 가중치 (프롬프트 튜닝 시 참고)
넓은 범주 → 구체적 특징 순서로 질문하도록 시스템 프롬프트에서 유도. 중복 질문 방지는 v1에서는 프롬프트에만 의존, v2에서 임베딩 유사도로 고도화 예정.

## 데이터 모델

### GameSession
- `session_id` (UUID, unique)
- `category` (CharField, default="인물")
- `is_finished` (BooleanField, default=False)
- `final_guess` (CharField, blank=True)
- `created_at` (auto)

### QA
- `session` (FK to GameSession, related_name="qas")
- `turn` (IntegerField)
- `question` (TextField)
- `answer` (CharField, "예"/"아니오"/"잘 모름")
- `created_at` (auto)
- `Meta.ordering = ['turn']`

## API 엔드포인트

### POST `/api/start/`
요청: `{"category": "인물"}`
응답: `{"session_id": "...", "question": "...", "action": "ask"}`

### POST `/api/answer/`
요청: `{"session_id": "...", "answer": "예"}`
응답:
- 계속: `{"action": "ask", "question": "..."}`
- 추측: `{"action": "guess", "answer": "아인슈타인"}`

## Docker 구성

### 서비스
- **web**: Django 앱, 포트 `8000:8000`, 코드 볼륨 마운트로 핫리로드
- **ollama**: Ollama 컨테이너, 호스트 포트 `11435`로 매핑 (호스트의 기존 Ollama 11434와 충돌 방지), GPU 할당, 모델은 `ollama_models` 볼륨에 영속화

### 네트워크
`tq-net` 브리지 네트워크. 컨테이너끼리는 `http://ollama:11434`로 통신.

### 볼륨
- `ollama_models`: 모델 파일 영속화 (컨테이너 삭제해도 유지)
- `sqlite_data`: DB 영속화

## 환경변수 (`.env`)
```
DJANGO_SECRET_KEY=<랜덤 문자열>
DJANGO_DEBUG=True
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1
OLLAMA_HOST=http://ollama:11434
OLLAMA_MODEL=llama3.1:8b
```
`settings.py`와 `ollama_client.py`에서 `python-dotenv`로 로드한다.

## 코딩 스타일
- 한국어 주석 OK (단, 함수/변수명은 영어)
- 타입 힌트 사용 (Python 3.11+ 문법)
- 함수는 작게, 단일 책임 원칙
- Ollama 관련 호출은 반드시 `ollama_client.py`에만 둘 것 (교체/테스트 용이)
- 프롬프트 문자열은 `akinator_logic.py` 상단 상수로 분리

## Git / GitHub
- 계정: `pgh165`
- 푸시 전 반드시 `.env`가 `.gitignore`에 포함되어 있는지 확인
- 커밋 메시지: 영어, conventional commits 스타일 선호 (`feat:`, `fix:`, `refactor:` 등)

## 작업 진행 순서 (추천)

Claude Code가 처음 이 프로젝트를 세팅할 때 따를 단계:

1. **프로젝트 뼈대 생성**
   - `.gitignore`, `.dockerignore`, `.env.example`, `requirements.txt` 먼저 작성
   - `git init` 실행

2. **Docker 파일 작성**
   - `Dockerfile`, `docker-compose.yml` 작성
   - 이 시점에서는 아직 Django 프로젝트가 없어도 됨

3. **Django 프로젝트 생성 (컨테이너 안에서)**
   ```bash
   docker compose run --rm web django-admin startproject tq_project .
   docker compose run --rm web python manage.py startapp game
   ```
   `INSTALLED_APPS`에 `'game'` 추가.

4. **settings.py 환경변수화**
   - `SECRET_KEY`, `DEBUG`, `ALLOWED_HOSTS`를 `os.getenv`로 읽도록 수정

5. **모델 작성 + 마이그레이션**
   - `game/models.py`에 `GameSession`, `QA` 정의
   - `docker compose exec web python manage.py makemigrations game`
   - `docker compose exec web python manage.py migrate`

6. **Ollama 클라이언트 + 로직 작성**
   - `game/ollama_client.py`: `call_ollama(prompt, system, temperature)` 함수
   - `game/akinator_logic.py`: 시스템 프롬프트 상수, `next_turn(session)` 함수
   - JSON 파싱 방어 로직 포함

7. **뷰 + URL 작성**
   - `game/views.py`: `index`, `start_game`, `answer`
   - `game/urls.py` + 프로젝트 `urls.py` include

8. **템플릿 작성**
   - `game/templates/game/index.html`: 최소 HTML + fetch

9. **동작 테스트**
   - `docker compose up -d`
   - `docker compose exec ollama ollama pull llama3.1:8b` (최초 1회)
   - `http://localhost:8000` 접속 확인

10. **Git 초기 커밋 + GitHub 푸시**

## 중요: 절대 금지 사항
- `.env` 파일을 git에 커밋하지 말 것
- `db.sqlite3`를 커밋하지 말 것
- `SECRET_KEY`를 하드코딩하지 말 것
- 호스트에 `venv`를 만들지 말 것 (Docker로만 관리)
- 호스트의 기존 Ollama(11434)와 포트 충돌시키지 말 것 (컨테이너는 11435 사용)

## 향후 확장 계획 (지금 아키텍처가 막지 않도록)
- **스트리밍 UI**: Ollama의 `stream: True` + SSE로 토큰 단위 출력
- **중복 질문 방지**: 임베딩 유사도 기반 필터링
- **카테고리 자동 추론**: 사용자가 카테고리 안 고르면 LLM이 먼저 묻기
- **피드백 루프**: 맞췄을 때 사용자 피드백 → DB 저장 → fine-tuning 데이터
- **모델 A/B 테스트**: `OLLAMA_MODEL` 환경변수만 바꿔서 비교
- **React 프론트엔드**: 지금은 최소 HTML, 나중에 분리 가능하도록 API 먼저 설계
- **배포**: 라즈베리파이(ARM) 이식 가능하도록 이미지 선택 시 multi-arch 고려

## 자주 쓰는 명령어 (Claude Code가 바로 쓸 수 있게)

```bash
# 빌드 + 실행
docker compose up -d --build

# 로그
docker compose logs -f web
docker compose logs -f ollama

# Django 명령
docker compose exec web python manage.py makemigrations
docker compose exec web python manage.py migrate
docker compose exec web python manage.py shell
docker compose exec web python manage.py createsuperuser

# Ollama 명령
docker compose exec ollama ollama list
docker compose exec ollama ollama pull <model>
docker compose exec ollama nvidia-smi   # GPU 확인

# 정리
docker compose down           # 컨테이너만 중지
docker compose down -v        # 볼륨까지 삭제 (모델 사라짐, 주의)
```

## 참고: 트러블슈팅

- **GPU 인식 안 될 때**: `docker compose exec ollama nvidia-smi`로 확인. 안 되면 호스트 `nvidia-container-toolkit` 설치 점검.
- **JSON 파싱 실패 빈발**: 시스템 프롬프트를 더 강하게, 또는 few-shot 예시 추가. Ollama `format: "json"` 옵션 적용 확인.
- **한국어 품질 낮음**: `qwen2.5:7b` 또는 `EEVE-Korean-10.8B`로 모델 교체.
- **포트 충돌 (11434)**: 호스트의 기존 Ollama와 겹침. compose에서 `11435:11434`로 매핑되어 있는지 확인.