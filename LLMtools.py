from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


def _load_dotenv() -> None:
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(name, value)


_load_dotenv()

_MAX_RETRIES = 4
_BASE_BACKOFF_SECONDS = 2.0
_MAX_BACKOFF_SECONDS = 60.0


def _setting(name: str, *, fallback: str | None = None) -> str:
    value = os.getenv(name, fallback)
    if not value:
        raise RuntimeError(f"{name} is required for LLM operations.")
    return value


def _endpoint(value: str) -> str:
    endpoint = value.rstrip("/")
    if endpoint.endswith("/chat/completions"):
        return endpoint
    return f"{endpoint}/chat/completions"


def _retry_after_seconds(error: HTTPError) -> float | None:
    header = (error.headers or {}).get("Retry-After") if error.headers else None
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    try:
        body = error.read().decode(errors="replace")[:800]
    except Exception:
        body = ""
    match = re.search(r"reset[_ -]?(?:seconds|after)[^0-9]*(\d+(?:\.\d+)?)", body, re.IGNORECASE)
    return float(match.group(1)) if match else None


def _request(messages: list[dict[str, str]], *, max_tokens: int, json_mode: bool = False) -> str:
    payload: dict[str, Any] = {
        "model": _setting("LLM_MODEL"),
        "messages": messages,
        "temperature": 0 if json_mode else 0.2,
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", str(max_tokens))),
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    api_key = _setting("LLM_API_KEY")
    auth_header = os.getenv("LLM_AUTH_HEADER", "Authorization")
    auth_prefix = os.getenv("LLM_AUTH_PREFIX", "Bearer")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": os.getenv(
            "LLM_USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ),
        auth_header: f"{auth_prefix} {api_key}".strip(),
    }
    extra_headers = os.getenv("LLM_EXTRA_HEADERS", "")
    if extra_headers:
        try:
            configured_headers = json.loads(extra_headers)
        except json.JSONDecodeError as error:
            raise ValueError("LLM_EXTRA_HEADERS must be a JSON object.") from error
        if not isinstance(configured_headers, dict):
            raise ValueError("LLM_EXTRA_HEADERS must be a JSON object.")
        headers.update({str(name): str(value) for name, value in configured_headers.items()})
    endpoint = os.getenv("LLM_ENDPOINT") or os.getenv("LLM_BASE_URL")
    if not endpoint:
        raise RuntimeError("LLM_ENDPOINT or LLM_BASE_URL is required for LLM operations.")
    endpoint = _endpoint(endpoint)
    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "90"))
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        request = Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as error:
            transient = error.code == 429 or 500 <= error.code <= 599
            if not transient or attempt == _MAX_RETRIES:
                body = error.read().decode("utf-8", errors="replace").strip()
                if len(body) > 500:
                    body = body[:500] + "..."
                raise RuntimeError(f"LLM request failed with HTTP {error.code}: {body or error.reason}") from error
            wait = _retry_after_seconds(error)
            time.sleep(min(wait if wait is not None else _BASE_BACKOFF_SECONDS * (2 ** attempt), _MAX_BACKOFF_SECONDS))
            last_error = error
        except (URLError, TimeoutError, OSError) as error:
            if attempt == _MAX_RETRIES:
                raise RuntimeError(f"Cannot reach LLM endpoint {endpoint}") from error
            time.sleep(min(_BASE_BACKOFF_SECONDS * (2 ** attempt), _MAX_BACKOFF_SECONDS))
            last_error = error
        except json.JSONDecodeError as error:
            raise RuntimeError("LLM endpoint returned invalid JSON") from error
    else:
        raise RuntimeError(f"LLM request exhausted retries: {last_error}")

    if "choices" in result and result["choices"]:
        choice = result["choices"][0]
        if choice.get("finish_reason") == "length":
            raise RuntimeError(
                "The LLM stopped at its token limit. Increase LLM_MAX_TOKENS and try again."
            )
        content = choice.get("message", {}).get("content")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if content is not None:
            return str(content)
    if result.get("output_text") is not None:
        return str(result["output_text"])
    if result.get("content") is not None:
        return str(result["content"])
    raise ValueError("LLM response did not contain recognized text content.")


def generate_cleaning_plan(df: pd.DataFrame, report: dict[str, Any], model: str | None = None) -> list[dict[str, Any]]:
    """Ask the model for bounded cleaning actions based on a compact data profile."""
    profile = {
        "columns": [{"name": column, "dtype": str(df[column].dtype), "missing": int(df[column].isna().sum()), "unique": int(df[column].nunique(dropna=False))} for column in df.columns],
        "sample": df.head(20).where(df.notna(), None).to_dict(orient="records"),
        "audit": report,
    }
    if model:
        os.environ["LLM_MODEL"] = model
    content = _request([
            {"role": "system", "content": "You are a cautious data-cleaning planner. Return JSON only with an actions array. Use observed values, column names, data types, missing counts, and the sample to infer safe corrections. Never invent a column. Never drop a column unless it is clearly useless or explicitly flagged. Prefer fill_missing with a value from the data, replace_value for an obvious invalid value, and drop_duplicate_rows for exact duplicates. Every action needs a short reason and confidence from 0 to 1. Do not execute anything."},
            {"role": "user", "content": "Create a proposed cleaning plan for this profile:\n" + json.dumps(profile, default=str)},
        ], max_tokens=8000, json_mode=True)
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(
            "The LLM cleaning plan was incomplete or invalid JSON. "
            "Increase LLM_MAX_TOKENS and try again."
        ) from error
    actions = payload.get("actions", [])
    if not isinstance(actions, list) or not all(isinstance(action, dict) for action in actions):
        raise ValueError("The LLM cleaning plan must contain an actions array of objects.")

    from dataquality import validate_recommendations

    validate_recommendations(df, actions)
    return actions


def generatesummary(text: str, model: str | None = None) -> str:
    """Generate an explanation of an audit report; never mutate the dataset."""
    if model:
        os.environ["LLM_MODEL"] = model
    return _request([
            {"role": "system", "content": "You explain data-quality audit results clearly. Do not invent values or claim that data was changed."},
            {"role": "user", "content": f"Summarize these findings and explain the proposed actions:\n\n{text}"},
        ], max_tokens=4000).strip()