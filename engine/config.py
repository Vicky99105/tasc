"""Loads .env once. os.environ always wins over the file, so CI/tests can override
without touching the file on disk."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path


def _read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        values[k.strip()] = v.strip()
    return values


@dataclass(frozen=True)
class Config:
    openrouter_api_key: str
    openrouter_base_url: str
    model_link: str
    model_summary: str
    reference_date: date
    slack_webhook_url: str | None
    langfuse_host: str | None
    langfuse_public_key: str | None
    langfuse_secret_key: str | None


def load_config(env_path: str = ".env") -> Config:
    values = _read_dotenv(Path(env_path))
    values.update({k: v for k, v in os.environ.items() if k in values})

    def get(key: str, default: str | None = None, required: bool = False) -> str | None:
        v = values.get(key, default)
        if required and not v:
            raise RuntimeError(f"missing required env var: {key}")
        return v

    model_link = get("OPENROUTER_MODEL", "google/gemini-3.7-flash")
    return Config(
        openrouter_api_key=get("OPENROUTER_API_KEY", required=True),
        openrouter_base_url=(get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1") or "").rstrip("/"),
        model_link=model_link,
        model_summary=get("MODEL_SUMMARY", model_link),
        reference_date=date.fromisoformat(get("REFERENCE_DATE", required=True)),
        slack_webhook_url=get("SLACK_WEBHOOK_URL") or None,
        langfuse_host=get("LANGFUSE_HOST") or None,
        langfuse_public_key=get("LANGFUSE_PUBLIC_KEY") or None,
        langfuse_secret_key=get("LANGFUSE_SECRET_KEY") or None,
    )
