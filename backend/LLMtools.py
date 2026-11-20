"""Small LangChain helpers kept for the optional CLI flags."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from .csv_io import preview
from .dataquality import profile_dataframe, validate_recommendations
from .graph import CleaningPlan, _agent_context, _llm, llm_ready

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def generate_cleaning_plan(df: pd.DataFrame, report: dict[str, Any], model: str | None = None) -> list[dict[str, Any]]:
    if model:
        os.environ["LLM_MODEL"] = model
    if not llm_ready():
        raise RuntimeError("Set LLM_API_KEY, LLM_ENDPOINT, and LLM_MODEL before using --llm.")
    planner = _llm().with_structured_output(CleaningPlan)
    context = _agent_context({
        "profile": profile_dataframe(df),
        "report": report,
        "preview": preview(df),
    })
    response = planner.invoke([
        SystemMessage(content="You are a cautious CSV cleaning planner. Return only safe, explainable actions for the supported cleaning vocabulary. Never invent columns or values."),
        HumanMessage(content="Create a proposed cleaning plan from this bounded profile and audit context:\n" + json.dumps(context, default=str)),
    ])
    actions = [action.model_dump(exclude_none=True) for action in response.actions]
    validate_recommendations(df, actions)
    return actions


def generatesummary(text: str, model: str | None = None) -> str:
    if model:
        os.environ["LLM_MODEL"] = model
    if not llm_ready():
        raise RuntimeError("Set LLM_API_KEY, LLM_ENDPOINT, and LLM_MODEL before using --summary.")
    response = _llm().invoke([
        SystemMessage(content="Explain the audit clearly. Do not invent values or claim that data was changed."),
        HumanMessage(content=f"Summarize these findings:\n\n{text}"),
    ])
    return str(response.content).strip()
