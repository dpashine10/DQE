"""Small LangGraph pipeline for resumable CSV audits."""

from __future__ import annotations

import json
import operator
import sqlite3
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .csv_io import preview, read_csv
from .dataquality import audit_dataframe, profile_dataframe, summary_text, validate_recommendations
from .providers import agent_model, agent_ready
from .model_context import bounded_json

STORAGE = Path(__file__).resolve().parent / "storage"
STORAGE.mkdir(exist_ok=True)


class CleaningAction(BaseModel):
    action: str = Field(description="One supported action: fill_missing, replace_value, drop_rows_with_missing, drop_duplicate_rows, drop_column, trim_whitespace")
    column: str | None = None
    value: str | int | float | bool | None = None
    from_value: str | int | float | bool | None = None
    to_value: str | int | float | bool | None = None
    reason: str = ""
    confidence: float = 0.5
    decision: Literal["recommended", "nothing", "not_recommended"] = "recommended"
    decision_reason: str = ""


class CleaningPlan(BaseModel):
    actions: list[CleaningAction] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    job_id: str
    input_path: str
    catalog_path: str
    file_name: str
    status: str
    profile: dict[str, Any]
    report: dict[str, Any]
    preview: dict[str, Any]
    plan: list[dict[str, Any]]
    summary: str
    llm_used: bool
    llm_error: str
    skip_llm: bool
    agent_pending: bool
    events: Annotated[list[dict[str, Any]], operator.add]


def _event(step: str, message: str, status: str = "done") -> dict[str, Any]:
    return {"step": step, "message": message, "status": status}


def llm_ready() -> bool:
    return agent_ready("primary")


def _llm() -> ChatOpenAI:
    return agent_model("primary")


def _load(state: AgentState):
    return {"status": "loading", "events": [_event("load", f"Loaded {state['file_name']}")]}


def _profile(state: AgentState):
    frame = read_csv(state["input_path"])
    catalog_path = Path(state["input_path"]).with_name("value-catalog.json")
    catalog = {}
    missing_values = {"", "?", "null", "none", "nan", "n/a", "missing"}
    for column in frame.columns:
        values = frame[column][~frame[column].astype(str).str.strip().str.casefold().isin(missing_values)].astype(str)
        counts = values.value_counts().head(10_000)
        catalog[column] = {str(value): int(count) for value, count in counts.items()}
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    return {
        "profile": profile_dataframe(frame), "preview": preview(frame), "catalog_path": str(catalog_path), "status": "profiled",
        "events": [_event("profile", f"Profiled {len(frame)} rows across {len(frame.columns)} columns")],
    }


def _audit(state: AgentState):
    frame = read_csv(state["input_path"])
    report = audit_dataframe(frame)
    return {
        "report": report, "status": "audited",
        "events": [_event("audit", f"Found {len(report['issues'])} quality signals")],
    }


def _agent_context(state: AgentState) -> dict[str, Any]:
    """Build a bounded model context without sending the local catalog wholesale."""
    profile = json.loads(json.dumps(state["profile"], default=str))
    for column in profile.get("columns", []):
        catalog = column.get("value_catalog", {})
        catalog["values"] = []
        column["value_catalog"] = catalog
        column["top_values"] = [
            {"value": str(item.get("value", ""))[:80], "count": item.get("count", 0)}
            for item in column.get("top_values", [])[:5]
        ]

    report = json.loads(json.dumps(state["report"], default=str))
    report["issues"] = report.get("issues", [])[:100]
    report["recommendations"] = report.get("recommendations", [])[:100]
    report["context_note"] = "The complete report and capped value catalog remain local; this prompt contains a bounded context excerpt."
    return {"profile": profile, "audit": report, "sample": state["preview"]}


def _plan(state: AgentState):
    frame = read_csv(state["input_path"])
    deterministic = state["report"]["recommendations"]
    if state.get("skip_llm") or not llm_ready():
        message = "Used deterministic recommendations while the agent plan is queued" if state.get("skip_llm") else "Used deterministic recommendations; LLM is not configured"
        return {
            "plan": deterministic, "llm_used": False, "status": "planned",
            "events": [_event("plan", message, "fallback")],
        }
    try:
        context = _agent_context(state)
        context["profile"]["catalog_note"] = "A capped unique-value catalog is stored locally for this job. Only compact top-value metadata is included in this prompt."
        planner = _llm().with_structured_output(CleaningPlan)
        response = planner.invoke([
            SystemMessage(content="You are a cautious CSV cleaning planner. Suggest only actions supported by the schema. Never invent columns or values. Do not fix an outlier only because it is unusual. For each action classify the decision as recommended, nothing, or not_recommended. Use not_recommended when a replacement cannot be proven, such as guessing a person's age. Return only safe, explainable actions."),
            HumanMessage(content="Create a proposed plan from this audit, profile, and bounded value metadata. Pay special attention to values that are null-like, impossible, out of range, misspelled, or inconsistent.\n" + bounded_json(context)),
        ])
        actions = [action.model_dump(exclude_none=True) for action in response.actions]
        known = {(action.get("action"), action.get("column"), action.get("from_value"), action.get("to_value")) for action in actions}
        for action in deterministic:
            signature = (action.get("action"), action.get("column"), action.get("from_value"), action.get("to_value"))
            if signature not in known:
                actions.append(action)
        validate_recommendations(frame, actions)
        return {
            "plan": actions, "llm_used": True, "status": "planned",
            "events": [_event("plan", f"LangChain proposed {len(actions)} validated actions")],
        }
    except Exception as error:
        return {
            "plan": deterministic, "llm_used": False, "llm_error": str(error), "status": "planned",
            "events": [_event("plan", "LLM plan failed; kept deterministic recommendations", "fallback")],
        }


def _summarize(state: AgentState):
    summary = summary_text(state["report"])
    return {"summary": summary, "status": "complete", "agent_pending": False, "events": [_event("complete", "Audit and plan are ready") ]}


def build_graph(checkpointer):
    builder = StateGraph(AgentState)
    builder.add_node("load", _load)
    builder.add_node("profile", _profile)
    builder.add_node("audit", _audit)
    builder.add_node("plan", _plan)
    builder.add_node("summarize", _summarize)
    builder.add_edge(START, "load")
    builder.add_edge("load", "profile")
    builder.add_edge("profile", "audit")
    builder.add_edge("audit", "plan")
    builder.add_edge("plan", "summarize")
    builder.add_edge("summarize", END)
    return builder.compile(checkpointer=checkpointer)


_connection = sqlite3.connect(str(STORAGE / "checkpoints.sqlite"), check_same_thread=False)
_checkpointer = SqliteSaver(_connection)
_checkpointer.setup()
graph = build_graph(_checkpointer)


def run_job(job_id: str, input_path: Path, file_name: str, *, skip_llm: bool = False):
    config = {"configurable": {"thread_id": job_id}}
    return graph.invoke({"job_id": job_id, "input_path": str(input_path), "file_name": file_name, "skip_llm": skip_llm, "events": []}, config)


def mark_agent_pending(job_id: str):
    config = {"configurable": {"thread_id": job_id}}
    graph.update_state(config, {"status": "planning", "agent_pending": True}, as_node="summarize")


def run_deferred_plan(job_id: str):
    config = {"configurable": {"thread_id": job_id}}
    snapshot = graph.get_state(config)
    if not snapshot:
        return None
    graph.update_state(config, {"skip_llm": False, "status": "planning", "agent_pending": True, "llm_error": ""}, as_node="audit")
    return graph.invoke(None, config)


def retry_job(job_id: str):
    config = {"configurable": {"thread_id": job_id}}
    snapshot = graph.get_state(config)
    if snapshot and snapshot.values.get("report"):
        # A completed audit has all the context needed to retry only the plan onward.
        graph.update_state(config, {"status": "retrying", "llm_error": ""}, as_node="audit")
    return graph.invoke(None, config)


def get_job(job_id: str):
    snapshot = graph.get_state({"configurable": {"thread_id": job_id}})
    return snapshot.values if snapshot else None


def summarize_job(job_id: str):
    state = get_job(job_id)
    if not state:
        raise ValueError("Job not found.")
    fallback = state.get("summary", "The audit is ready for review.")
    if not llm_ready():
        return {"summary": fallback, "source": "deterministic"}
    try:
        context = _agent_context(state)
        context["plan"] = state.get("plan", [])[:100]
        response = _llm().invoke([
            SystemMessage(content="You are a careful data quality reviewer. Summarize the audit in plain language. Mention the score, the most important risks, and what the user should review. Do not claim that data was changed. Keep it under 120 words."),
            HumanMessage(content=bounded_json({"file": state.get("file_name"), **context})),
        ])
        content = response.content
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return {"summary": str(content).strip(), "source": "langchain"}
    except Exception as error:
        return {"summary": fallback, "source": "deterministic", "llm_error": str(error)}
