"""Small role-based model factory for the main agent and its subagents.

Every role uses an OpenAI-compatible chat endpoint. Roles fall back to the
base LLM_* settings, so one provider works by default while each role can be
given a different provider, key, endpoint, or model when needed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _role_value(role: str, name: str, fallback: str | None = None) -> str | None:
    prefix = f"AGENT_{role.upper()}_"
    return os.getenv(prefix + name) or os.getenv("LLM_" + name, fallback)


def _base_url(endpoint: str | None) -> str | None:
    if not endpoint:
        return None
    return endpoint.rstrip("/").removesuffix("/chat/completions")


def agent_ready(role: str = "primary") -> bool:
    return bool(_role_value(role, "API_KEY") and _role_value(role, "MODEL") and _base_url(_role_value(role, "ENDPOINT") or _role_value(role, "BASE_URL")))


def agent_model(role: str = "primary") -> ChatOpenAI:
    if not agent_ready(role):
        raise RuntimeError(f"No model configuration is available for the {role} agent.")
    extra_headers = _role_value(role, "EXTRA_HEADERS", "") or ""
    headers = json.loads(extra_headers) if extra_headers else {}
    auth_header = _role_value(role, "AUTH_HEADER")
    if auth_header and auth_header.lower() != "authorization":
        prefix = _role_value(role, "AUTH_PREFIX", "Bearer") or ""
        headers[auth_header] = f"{prefix} {_role_value(role, 'API_KEY')}".strip()
    settings = {}
    max_tokens = _role_value(role, "MAX_TOKENS")
    if max_tokens:
        settings["max_tokens"] = int(max_tokens)
    provider = (_role_value(role, "PROVIDER", "openai_compatible") or "openai_compatible").lower()
    if provider not in {"openai", "openai_compatible", "openrouter", "groq", "together", "deepseek", "ollama"}:
        raise ValueError(f"Unsupported provider '{provider}' for the {role} agent. Use an OpenAI-compatible endpoint.")
    return ChatOpenAI(
        model=_role_value(role, "MODEL"),
        api_key=_role_value(role, "API_KEY"),
        base_url=_base_url(_role_value(role, "ENDPOINT") or _role_value(role, "BASE_URL")),
        temperature=0,
        timeout=float(_role_value(role, "TIMEOUT_SECONDS", "90") or "90"),
        max_retries=2,
        default_headers=headers or None,
        **settings,
    )


def provider_summary() -> dict[str, dict[str, str | bool]]:
    return {
        role: {
            "configured": agent_ready(role),
            "provider": _role_value(role, "PROVIDER", "openai_compatible") or "openai_compatible",
            "model": _role_value(role, "MODEL", "") or "",
        }
        for role in ("primary", "research", "insights", "code")
    }
