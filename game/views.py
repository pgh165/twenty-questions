"""게임 세션 API 뷰."""
import json
import logging

from django.db import transaction
from django.http import HttpRequest, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from .akinator_logic import next_turn
from .models import GameSession, QA

log = logging.getLogger(__name__)


@require_GET
def index(request: HttpRequest):
    """최소 UI 페이지."""
    return render(request, "game/index.html")


def _parse_json(request: HttpRequest) -> dict:
    try:
        return json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return {}


def _apply_action(session: GameSession, action: dict) -> dict:
    """next_turn 결과를 DB에 반영하고 클라이언트 응답용 dict를 반환."""
    if action["action"] == "ask":
        session.pending_question = action["question"]
        session.save(update_fields=["pending_question"])
        return {"action": "ask", "question": action["question"]}

    # guess
    session.is_finished = True
    session.final_guess = action["answer"]
    session.pending_question = ""
    session.save(update_fields=["is_finished", "final_guess", "pending_question"])
    return {"action": "guess", "answer": action["answer"]}


@csrf_exempt
@require_POST
def start_game(request: HttpRequest) -> JsonResponse:
    """새 세션 시작 + 첫 질문 생성."""
    body = _parse_json(request)
    category = (body.get("category") or "인물").strip() or "인물"

    session = GameSession.objects.create(category=category)
    action = next_turn(session)
    payload = _apply_action(session, action)
    payload["session_id"] = str(session.session_id)
    return JsonResponse(payload)


@csrf_exempt
@require_POST
def answer(request: HttpRequest) -> JsonResponse:
    """사용자 답변 저장 + 다음 질문/추측 생성."""
    body = _parse_json(request)
    session_id = body.get("session_id")
    user_answer = body.get("answer")

    if not session_id or not user_answer:
        return JsonResponse(
            {"error": "session_id와 answer가 필요합니다."}, status=400
        )

    valid_answers = {c[0] for c in QA.ANSWER_CHOICES}
    if user_answer not in valid_answers:
        return JsonResponse(
            {"error": f"answer는 {sorted(valid_answers)} 중 하나여야 합니다."},
            status=400,
        )

    try:
        session = GameSession.objects.get(session_id=session_id)
    except GameSession.DoesNotExist:
        return JsonResponse({"error": "세션을 찾을 수 없습니다."}, status=404)

    if session.is_finished:
        return JsonResponse({"error": "이미 종료된 세션입니다."}, status=400)

    if not session.pending_question:
        return JsonResponse(
            {"error": "답변할 질문이 없습니다."}, status=400
        )

    # 이전 질문 + 방금 받은 답변을 QA로 커밋
    with transaction.atomic():
        QA.objects.create(
            session=session,
            turn=session.qas.count() + 1,
            question=session.pending_question,
            answer=user_answer,
        )
        session.pending_question = ""
        session.save(update_fields=["pending_question"])

    action = next_turn(session)
    payload = _apply_action(session, action)
    return JsonResponse(payload)
