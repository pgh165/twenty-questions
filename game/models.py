import uuid

from django.db import models


class GameSession(models.Model):
    """한 판의 Twenty Questions 게임 세션."""

    session_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    category = models.CharField(max_length=32, default="인물")
    is_finished = models.BooleanField(default=False)
    final_guess = models.CharField(max_length=200, blank=True)
    # LLM이 제시했고 아직 사용자가 답하지 않은 질문. 답이 도착하면 QA로 옮기고 비운다.
    pending_question = models.TextField(blank=True, default="")
    # 매 턴 LLM이 추정한 남은 후보 Top-N. 질문 생성 시 변별력 판단 근거로 사용.
    candidates = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"{self.category} / {self.session_id}"


class QA(models.Model):
    """세션 내 한 턴의 질문-답변 쌍."""

    ANSWER_YES = "예"
    ANSWER_NO = "아니오"
    ANSWER_UNKNOWN = "잘 모름"
    ANSWER_CHOICES = [
        (ANSWER_YES, ANSWER_YES),
        (ANSWER_NO, ANSWER_NO),
        (ANSWER_UNKNOWN, ANSWER_UNKNOWN),
    ]

    session = models.ForeignKey(
        GameSession, on_delete=models.CASCADE, related_name="qas"
    )
    turn = models.IntegerField()
    question = models.TextField()
    answer = models.CharField(max_length=8, choices=ANSWER_CHOICES)
    # 질문 임베딩 (의미적 중복 탐지용). 임베딩 호출 실패 시 null.
    question_embedding = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["turn"]
        constraints = [
            models.UniqueConstraint(
                fields=["session", "turn"], name="unique_session_turn"
            )
        ]

    def __str__(self) -> str:
        return f"T{self.turn}: {self.question[:30]} → {self.answer}"
