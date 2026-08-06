"""Minimal OpenAI-compatible client with retries and JSON parsing."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .config import LLMConfig


class LLMRequestError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResponse:
    content: dict[str, Any]
    response_id: str | None
    usage: dict[str, Any]


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig) -> None:
        if not config.is_configured:
            raise ValueError("LLM configuration is incomplete")
        self.config = config

    def complete_json(self, system: str, user: str) -> LLMResponse:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }

        raw = self._post_json("/chat/completions", payload)
        try:
            message = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError("API response has no message content") from exc

        if not isinstance(message, str):
            raise LLMRequestError("API message content is not text")

        try:
            content = json.loads(_strip_code_fence(message))
        except json.JSONDecodeError as exc:
            raise LLMRequestError("API message is not valid JSON") from exc
        if not isinstance(content, dict):
            raise LLMRequestError("API JSON response must be an object")

        usage = raw.get("usage")
        return LLMResponse(
            content=content,
            response_id=raw.get("id"),
            usage=usage if isinstance(usage, dict) else {},
        )

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "rsna-knee/0.1",
        }

        last_error: Exception | None = None
        with httpx.Client(
            timeout=self.config.timeout_seconds,
            headers={
                **headers,
            },
        ) as client:
            for attempt in range(self.config.max_retries + 1):
                try:
                    response = client.post(url, json=payload)
                    if response.status_code >= 400:
                        detail = response.text[:500]
                        last_error = LLMRequestError(
                            f"API returned HTTP {response.status_code}: {detail}"
                        )
                        if response.status_code not in {
                            408,
                            409,
                            429,
                            500,
                            502,
                            503,
                            504,
                        }:
                            raise last_error
                    else:
                        decoded = response.json()
                        if not isinstance(decoded, dict):
                            raise LLMRequestError(
                                "API response must be a JSON object"
                            )
                        return decoded
                except (
                    httpx.NetworkError,
                    httpx.TimeoutException,
                    json.JSONDecodeError,
                ) as exc:
                    last_error = exc

                if attempt < self.config.max_retries:
                    delay = min(2**attempt + random.random(), 10.0)
                    time.sleep(delay)

        raise LLMRequestError("API request failed after retries") from last_error


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped
