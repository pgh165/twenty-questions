"""Ollama HTTP API 호출 격리 모듈.

다른 모듈은 이 파일의 함수만 사용해야 한다 (교체/테스트 용이).
"""
import os
from typing import Any

import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
DEFAULT_TIMEOUT = 120


class OllamaError(RuntimeError):
    """Ollama 호출 실패 전용 예외."""


def call_ollama(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.7,
    model: str | None = None,
    json_mode: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Ollama /api/generate 비스트리밍 호출. 응답 문자열을 그대로 반환."""
    payload: dict[str, Any] = {
        "model": model or OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if system:
        payload["system"] = system
    if json_mode:
        payload["format"] = "json"

    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise OllamaError(f"Ollama 호출 실패: {e}") from e

    data = resp.json()
    return data.get("response", "")
