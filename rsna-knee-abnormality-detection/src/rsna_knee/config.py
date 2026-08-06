"""Environment configuration without secret disclosure."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    base_url: str
    api_key: str
    timeout_seconds: int = 120
    max_retries: int = 3

    @property
    def is_configured(self) -> bool:
        return all((self.provider, self.model, self.base_url, self.api_key))

    def safe_summary(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "has_api_key": bool(self.api_key),
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
        }


def load_llm_config(env_path: str | Path | None = None) -> LLMConfig:
    """Load LLM settings from an explicit path or the current environment."""
    if env_path is not None:
        load_dotenv(Path(env_path), override=False)

    return LLMConfig(
        provider=os.getenv("LLM_PROVIDER", "").strip(),
        model=os.getenv("LLM_MODEL", "").strip(),
        base_url=os.getenv("LLM_BASE_URL", "").strip().rstrip("/"),
        api_key=os.getenv("LLM_API_KEY", "").strip(),
        timeout_seconds=int(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "120")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
    )
