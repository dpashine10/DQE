"""Notebook planning, subagent research, and safe ipynb generation.

The application writes notebook JSON. It never imports, evaluates, or runs the
generated Python code.
"""

from __future__ import annotations

import inspect
import json
import operator
import re
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

import nbformat
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .csv_io import read_csv
from .dataquality import audit_dataframe, profile_dataframe
from .graph import STORAGE, _connection, _event
from .notebook_cells import plot_data, preprocess_dataframe, train_baseline
from .providers import agent_model, agent_ready, provider_summary
from .model_context import bounded_json


MAX_CATALOG_AGENT_BATCHES = 20


class ResearchBrief(BaseModel):
    task: str = "exploration"
    recommended_libraries: list[str] = Field(default_factory=list)
    recommended_charts: list[str] = Field(default_factory=list)
    x_column: str | None = None
    y_column: str | None = None
    target_column: str | None = None
    model_parameters: list[str] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)
    notes: str = ""


class DataInsights(BaseModel):
    important_columns: list[str] = Field(default_factory=list)
    target_column: str | None = None
    feature_columns: list[str] = Field(default_factory=list)
    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class NotebookPlan(BaseModel):
    title: str = "Dataset exploration"
    task: Literal["exploration", "regression", "classification"] = "exploration"
    target_column: str | None = None
    feature_columns: list[str] = Field(default_factory=list)
    chart_columns: list[str] = Field(default_factory=list)
    use_xgboost: bool = False
    libraries: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    rationale: str = ""


class NotebookCell(BaseModel):
    cell_type: Literal["markdown", "code"]
    source: str


class NotebookSpec(BaseModel):
    title: str = "Generated data notebook"
    cells: list[NotebookCell] = Field(default_factory=list)


class NotebookState(TypedDict, total=False):
    notebook_thread_id: str
    job_id: str
    input_path: str
    file_name: str
    request: str
    profile: dict[str, Any]
    report: dict[str, Any]
    research: dict[str, Any]
    insights: dict[str, Any]
    plan: dict[str, Any]
    notebook: dict[str, Any]
    notebook_path: str
    provider_status: dict[str, dict[str, str | bool]]
    catalog_batches: list[dict[str, Any]]
    catalog_path: str
    llm_errors: list[str]
    events: Annotated[list[dict[str, Any]], operator.add]


def _context(state: NotebookState, value_batch: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = json.loads(json.dumps(state["profile"], default=str))
    for column in profile.get("columns", []):
        catalog = column.get("value_catalog", {})
        catalog["values"] = []
        column["value_catalog"] = catalog
        column["top_values"] = [
            {"value": str(item.get("value", ""))[:80], "count": item.get("count", 0)}
            for item in column.get("top_values", [])[:5]
        ]
    audit = json.loads(json.dumps(state.get("report", {}), default=str))
    audit["issues"] = audit.get("issues", [])[:100]
    audit["recommendations"] = audit.get("recommendations", [])[:100]
    audit["context_note"] = "The complete audit and capped value catalog remain local; this prompt contains bounded excerpts and one bounded catalog batch."
    context = {
        "request": state["request"],
        "file_name": state["file_name"],
        "profile": profile,
        "audit": audit,
        "catalog_batch_count": len(state.get("catalog_batches", [])),
    }
    if value_batch:
        context["value_batch"] = value_batch
    return context


def _agent_batches(batches: list[dict[str, Any]], limit: int = MAX_CATALOG_AGENT_BATCHES) -> list[dict[str, Any]]:
    """Select an evenly distributed, bounded set of batches for model calls."""
    if limit < 1:
        return []
    if len(batches) <= limit:
        return batches
    indexes = [round(position * (len(batches) - 1) / (limit - 1)) for position in range(limit)] if limit > 1 else [0]
    return [batches[index] for index in indexes]


def _catalog_batches(frame, values_per_column: int = 100, columns_per_batch: int = 6) -> list[dict[str, Any]]:
    """Split useful values into bounded batches while retaining full local coverage."""
    missing = {"", "?", "null", "none", "nan", "n/a", "missing"}
    columns = []
    for column in frame.columns:
        normalized = re.sub(r"[^a-z0-9]+", "_", column.lower()).strip("_")
        if normalized == "id" or normalized.endswith("_id"):
            continue
        values = frame[column].astype(str).str.strip()
        values = values[~values.str.casefold().isin(missing)].value_counts().head(10_000).index.tolist()
        chunks = [values[start:start + values_per_column] for start in range(0, len(values), values_per_column)] or [[]]
        columns.append((column, chunks))

    batches = []
    for start in range(0, len(columns), columns_per_batch):
        group = columns[start:start + columns_per_batch]
        max_chunks = max((len(chunks) for _, chunks in group), default=1)
        for chunk_index in range(max_chunks):
            batches.append({
                "batch": len(batches) + 1,
                "columns": [
                    {"name": name, "values": chunks[chunk_index] if chunk_index < len(chunks) else []}
                    for name, chunks in group
                ],
            })
    return batches or [{"batch": 1, "columns": []}]


def _requested_task(request: str) -> str:
    text = request.casefold()
    if any(word in text for word in ("regression", "regressor", "predict a number", "xgboost regression")):
        return "regression"
    if any(word in text for word in ("classification", "classifier", "classify", "predict a class")):
        return "classification"
    return "exploration"


def _columns(profile: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    numeric, categorical, all_columns = [], [], []
    for item in profile.get("columns", []):
        name = item["name"]
        all_columns.append(name)
        if item.get("kind") == "numeric":
            numeric.append(name)
        elif item.get("kind") == "text":
            categorical.append(name)
    return numeric, categorical, all_columns


def _fallback_research(state: NotebookState) -> dict[str, Any]:
    numeric, categorical, columns = _columns(state["profile"])
    task = _requested_task(state["request"])
    target = next((column for column in numeric if column.casefold() != "id" and not column.casefold().endswith("_id")), None) if task == "regression" else (categorical[-1] if task == "classification" and categorical else None)
    x_column = numeric[0] if numeric else (columns[0] if columns else None)
    y_column = numeric[1] if len(numeric) > 1 else (numeric[0] if numeric else None)
    return ResearchBrief(
        task=task,
        recommended_libraries=["pandas", "numpy", "matplotlib", "seaborn"] + (["scikit-learn"] if task != "exploration" else []),
        recommended_charts=["histograms", "bar charts", "correlation heatmap", "scatter plot"],
        x_column=x_column,
        y_column=y_column,
        target_column=target,
        model_parameters=["train/test split", "random_state=42", "evaluation metrics"],
        cautions=["Confirm the target column before training.", "The generated notebook is not executed by DataQualityEng."],
    ).model_dump()


def _fallback_insights(state: NotebookState) -> dict[str, Any]:
    numeric, categorical, columns = _columns(state["profile"])
    task = _requested_task(state["request"])
    target = next((column for column in numeric if column.casefold() in state["request"].casefold()), None) if task == "regression" else None
    if not target and task == "regression":
        target = numeric[-1] if numeric else None
    if not target and task == "classification":
        target = categorical[-1] if categorical else (columns[-1] if columns else None)
    features = [column for column in columns if column != target and not column.casefold().endswith("_id") and column.casefold() != "id"]
    return DataInsights(
        important_columns=columns[:8],
        target_column=target,
        feature_columns=features[:20],
        numeric_columns=numeric,
        categorical_columns=categorical,
        concerns=["Review missing values before training.", "Confirm the target column and whether identifiers should be excluded."],
        suggestions=["Compare distributions before choosing a model.", "Use a held-out test set."],
    ).model_dump()


def _fallback_plan(state: NotebookState) -> dict[str, Any]:
    research, insights = state["research"], state["insights"]
    task = _requested_task(state["request"])
    target = insights.get("target_column") or research.get("target_column")
    chart_columns = insights.get("numeric_columns", [])[:8]
    return NotebookPlan(
        title=f"{task.title()} notebook for {state['file_name']}",
        task=task,
        target_column=target,
        feature_columns=insights.get("feature_columns", []),
        chart_columns=chart_columns,
        use_xgboost="xgboost" in state["request"].casefold(),
        libraries=research.get("recommended_libraries", []),
        steps=["Load the CSV", "Inspect shape, dtypes, missing values, and duplicates", "Create distribution and category plots", "Create a correlation heatmap", "Train and evaluate a model" if task != "exploration" else "Write observations and next steps"],
        rationale="The plan combines deterministic column inference with the request. Confirm the target and generated assumptions before running it.",
    ).model_dump()


def _source_lines(source: str) -> list[str]:
    return [line + "\n" for line in source.splitlines()] or ["\n"]


def _markdown(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "id": uuid.uuid4().hex[:12], "source": _source_lines(text)}


def _code(text: str) -> dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "id": uuid.uuid4().hex[:12], "outputs": [], "source": _source_lines(text)}


def deterministic_notebook(state: NotebookState) -> dict[str, Any]:
    """Create the single deterministic, preprocessing-first notebook fallback."""
    plan = NotebookPlan.model_validate(state["plan"])
    data_file = Path(state["file_name"]).name
    profile_columns = state["profile"].get("columns", [])
    column_kinds = {item["name"]: item.get("kind", "text") for item in profile_columns}
    numeric_rules = {}
    for item in profile_columns:
        name = item["name"].casefold()
        words = set(re.sub(r"[^a-z0-9]+", "_", name).split("_"))
        if item.get("kind") != "numeric":
            continue
        rule = {}
        if any(token in name for token in ("price", "amount", "mileage", "quantity", "count", "weight", "height", "seats")):
            rule["min"] = 0
        if "age" in words:
            rule.update({"min": 0, "max": 120, "integer": True})
        if "year" in words:
            rule.update({"min": 1000, "max": 2100, "integer": True})
        numeric_rules[item["name"]] = rule

    numeric = [name for name, kind in column_kinds.items() if kind == "numeric"]
    dates = [name for name, kind in column_kinds.items() if kind == "date"]
    features = plan.feature_columns or [name for name, kind in column_kinds.items() if name != plan.target_column and kind != "identifier"]
    chart_columns = plan.chart_columns or numeric[:8]
    dependencies = ["pandas", "numpy", "matplotlib", "seaborn"]
    if plan.task != "exploration":
        dependencies.append("scikit-learn")
    if plan.use_xgboost:
        dependencies.append("xgboost")

    cells = [
        _markdown(f"# {plan.title}\n\nGenerated from: `{state['request']}`\n\nThis notebook normalizes invalid values before plotting or training. DataQualityEng generates the file but never executes it."),
        _code("# Install only if dependencies are missing.\n%pip install " + " ".join(dict.fromkeys(dependencies))),
        _code(f'''# REQUIRED PREPROCESSING: this cell runs before every plot and model.
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = Path({data_file!r})
MISSING_MARKERS = ["", "?", "N/A", "NA", "n/a", "na", "NULL", "null", "None", "none", "NaN", "nan", "missing", "-", "--"]
raw_df = pd.read_csv(DATA_PATH, keep_default_na=False, na_values=MISSING_MARKERS)

{inspect.getsource(preprocess_dataframe)}

column_kinds = {column_kinds!r}
numeric_rules = {numeric_rules!r}
df, preprocessing_report = preprocess_dataframe(raw_df, column_kinds, MISSING_MARKERS, numeric_rules)
display(preprocessing_report)
display(df.head())'''),
        _code('''# Inspection happens after preprocessing.
print(df.info())
display(df.describe(include="all").T.head(40))'''),
    ]

    plot_specs = [{"kind": "histogram", "x": column} for column in chart_columns]
    if len(chart_columns) >= 2:
        plot_specs.append({"kind": "scatter", "x": chart_columns[0], "y": chart_columns[1]})
    plot_specs.append({"kind": "heatmap"})
    cells.append(_code(f'''# Plots happen after preprocessing and use cleaned/coerced values only.
{inspect.getsource(plot_data)}
plot_data(df, {plot_specs!r})'''))

    if plan.task != "exploration":
        target = plan.target_column or "REPLACE_WITH_TARGET_COLUMN"
        cells.extend([
            _markdown(f"## {plan.task.title()} model\n\nTarget selected by the agents: `{target}`. Confirm it before running."),
            _code(f'''# Training happens after preprocessing and plotting.
{inspect.getsource(train_baseline)}
TARGET = {target!r}
FEATURES = {features!r}
column_kinds = {column_kinds!r}
training_result = train_baseline(df, TARGET, FEATURES, column_kinds, task={plan.task!r}, use_xgboost={plan.use_xgboost!r})'''),
        ])
    cells.append(_markdown("## Review before running\n\n- Run the preprocessing cell first.\n- It treats `?`, `N/A`, `null`, blanks, and similar markers as missing.\n- Numeric and date columns are coerced before plotting.\n- Invalid target rows are excluded; feature missingness is imputed inside the training pipeline.\n- Small or unusable datasets skip training with a clear message."))
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}}


def _cell_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _valid_notebook(notebook: dict[str, Any], task: str = "exploration") -> tuple[bool, str]:
    try:
        nbformat.validate(notebook)
    except Exception as error:
        return False, f"Notebook JSON is invalid: {error}"
    cells = notebook.get("cells", [])
    if notebook.get("nbformat") != 4 or not cells:
        return False, "Notebook is missing nbformat 4 cells."
    code_cells = [(index, _cell_text(cell)) for index, cell in enumerate(cells) if cell.get("cell_type") == "code"]
    markdown_cells = [cell for cell in cells if cell.get("cell_type") == "markdown"]
    if len(code_cells) < 3:
        return False, "Notebook must use separate setup, preprocessing, and analysis code cells."
    if len(markdown_cells) < 2:
        return False, "Notebook needs separate title and review/explanation cells."

    preprocessing_cells = [(index, source) for index, source in code_cells if "REQUIRED PREPROCESSING" in source]
    if not preprocessing_cells:
        return False, "Notebook must include a dedicated REQUIRED PREPROCESSING cell."
    preprocessing_index, preprocessing_source = preprocessing_cells[0]
    if "na_values" not in preprocessing_source or "pd.to_numeric" not in preprocessing_source or not re.search(r"errors\s*=\s*['\"]coerce['\"]", preprocessing_source):
        return False, "Notebook must normalize missing markers and coerce dirty numeric values before analysis."
    if not any("read_csv" in source for _, source in code_cells):
        return False, "Notebook must load the CSV with pandas read_csv."

    plot_cells = [(index, source) for index, source in code_cells if any(token in source for token in ("plot_data(", "plt.", "sns.", ".hist("))]
    model_cells = [(index, source) for index, source in code_cells if any(token in source for token in ("train_baseline(", "train_test_split", ".fit("))]
    if any(index < preprocessing_index for index, _ in plot_cells + model_cells):
        return False, "Notebook preprocessing must happen before plotting or training."
    if task == "exploration" and not any(index > preprocessing_index for index, _ in plot_cells):
        return False, "Exploration notebooks need a dedicated plotting cell after preprocessing."
    if task in {"regression", "classification"}:
        if not any(index > preprocessing_index for index, _ in model_cells):
            return False, "Training notebooks need a dedicated model cell after preprocessing."
        if not any("TARGET" in source or "target" in source for _, source in model_cells):
            return False, "Training notebooks must identify the target column."

    joined = "\n".join(_cell_text(cell) for cell in cells)
    forbidden = ("os.system", "subprocess.", "eval(", "exec(", "shutil.rmtree", "socket.", "requests.get", "urllib.request")
    if any(token in joined for token in forbidden):
        return False, "Notebook contains an unsafe execution or network pattern."
    return True, "ok"


def _load_context(state: NotebookState):
    frame = read_csv(state["input_path"])
    catalog_path = Path(state["input_path"]).with_name("notebook-value-catalog.json")
    catalog = {column: frame[column].astype(str).value_counts().head(10_000).to_dict() for column in frame.columns}
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    return {"profile": profile_dataframe(frame), "report": audit_dataframe(frame), "provider_status": provider_summary(), "catalog_batches": _catalog_batches(frame), "catalog_path": str(catalog_path), "events": [_event("notebook_load", "Loaded the dataset context and split value catalogs into bounded batches")]}


def _research(state: NotebookState):
    fallback = _fallback_research(state)
    if not agent_ready("research"):
        return {"research": fallback, "llm_errors": ["research agent is not configured"], "events": [_event("research_subagent", "Used deterministic research fallback", "fallback")]}
    try:
        responses = []
        batches = _agent_batches(state.get("catalog_batches", []))
        for batch in batches:
            responses.append(agent_model("research").with_structured_output(ResearchBrief).invoke([
                SystemMessage(content="You are the notebook research subagent. Research the requested plotting or machine-learning task from the supplied dataset profile and this value batch. Identify what x, y, target, features, charts, libraries, and model parameters make sense. Do not write or execute code. Treat ?, N/A, null, and similar markers as missing values."),
                HumanMessage(content=bounded_json(_context(state, batch))),
            ]))
        merged = ResearchBrief(
            task=responses[0].task if responses else fallback["task"],
            recommended_libraries=list(dict.fromkeys(value for response in responses for value in response.recommended_libraries)),
            recommended_charts=list(dict.fromkeys(value for response in responses for value in response.recommended_charts)),
            x_column=next((response.x_column for response in responses if response.x_column), None),
            y_column=next((response.y_column for response in responses if response.y_column), None),
            target_column=next((response.target_column for response in responses if response.target_column), None),
            model_parameters=list(dict.fromkeys(value for response in responses for value in response.model_parameters)),
            cautions=list(dict.fromkeys(value for response in responses for value in response.cautions)),
            notes=" ".join(response.notes for response in responses if response.notes)[:2_000],
        ) if responses else ResearchBrief.model_validate(fallback)
        return {"research": merged.model_dump(), "events": [_event("research_subagent", f"Research subagent reviewed {len(responses) or 1} bounded value batches across {len(state.get('catalog_batches', []))} local batches")]}
    except Exception as error:
        return {"research": fallback, "llm_errors": [f"research agent: {error}"], "events": [_event("research_subagent", "Research provider failed; used deterministic fallback", "fallback")]}


def _insights(state: NotebookState):
    fallback = _fallback_insights(state)
    if not agent_ready("insights"):
        return {"insights": fallback, "llm_errors": ["insights agent is not configured"], "events": [_event("insights_subagent", "Used deterministic data-insights fallback", "fallback")]}
    try:
        responses = []
        batches = _agent_batches(state.get("catalog_batches", []))
        for batch in batches:
            responses.append(agent_model("insights").with_structured_output(DataInsights).invoke([
                SystemMessage(content="You are the data-insights subagent. Inspect this bounded value batch plus the profile. Decide which columns are useful, which are identifiers, and what target/features make sense for the user's task. Do not write or execute code. Never invent a column name. Treat ?, N/A, null, and similar markers as missing values."),
                HumanMessage(content=bounded_json({**_context(state, batch), "research": state["research"]})),
            ]))
        merged = DataInsights(
            important_columns=list(dict.fromkeys(value for response in responses for value in response.important_columns)),
            target_column=next((response.target_column for response in responses if response.target_column), None),
            feature_columns=list(dict.fromkeys(value for response in responses for value in response.feature_columns)),
            numeric_columns=list(dict.fromkeys(value for response in responses for value in response.numeric_columns)),
            categorical_columns=list(dict.fromkeys(value for response in responses for value in response.categorical_columns)),
            concerns=list(dict.fromkeys(value for response in responses for value in response.concerns)),
            suggestions=list(dict.fromkeys(value for response in responses for value in response.suggestions)),
        ) if responses else DataInsights.model_validate(fallback)
        return {"insights": merged.model_dump(), "events": [_event("insights_subagent", f"Data-insights subagent reviewed {len(responses) or 1} bounded value batches across {len(state.get('catalog_batches', []))} local batches")]}
    except Exception as error:
        return {"insights": fallback, "llm_errors": [f"insights agent: {error}"], "events": [_event("insights_subagent", "Insights provider failed; used deterministic fallback", "fallback")]}


def _primary_plan(state: NotebookState):
    fallback = _fallback_plan(state)
    if not agent_ready("primary"):
        return {"plan": fallback, "llm_errors": ["primary agent is not configured"], "events": [_event("primary_planner", "Used deterministic notebook plan", "fallback")]}
    try:
        response = agent_model("primary").with_structured_output(NotebookPlan).invoke([
            SystemMessage(content="You are the primary notebook agent. Combine the user's request, research subagent report, and data-insights report into a concrete plan. Choose real columns only. Select regression or classification only when requested. Mark target, feature columns, chart columns, dependencies, and steps. Do not write code yet."),
            HumanMessage(content=bounded_json({**_context(state), "research": state["research"], "insights": state["insights"]})),
        ])
        valid_columns = {item["name"] for item in state["profile"].get("columns", [])}
        requested_task = _requested_task(state["request"])
        if response.target_column and response.target_column not in valid_columns:
            raise ValueError(f"Primary planner chose unknown target column: {response.target_column}")
        if requested_task != "exploration" and response.task != requested_task:
            response = response.model_copy(update={"task": requested_task})
        response = response.model_copy(update={
            "feature_columns": [column for column in response.feature_columns if column in valid_columns],
            "chart_columns": [column for column in response.chart_columns if column in valid_columns],
        })
        return {"plan": response.model_dump(), "events": [_event("primary_planner", "Primary agent selected the notebook structure")]}
    except Exception as error:
        return {"plan": fallback, "llm_errors": [f"primary agent: {error}"], "events": [_event("primary_planner", "Primary provider failed; used deterministic plan", "fallback")]}


def _write_notebook(state: NotebookState):
    fallback = deterministic_notebook(state)
    if not agent_ready("code"):
        return {"notebook": fallback, "events": [_event("notebook_writer", "Generated a deterministic notebook", "fallback")]}
    try:
        response = agent_model("code").with_structured_output(NotebookSpec).invoke([
            SystemMessage(content="You are the notebook-writing subagent. Write a complete Jupyter notebook as separate markdown and code cells from the supplied plan. Include a title, dependency/setup cell, a dedicated code cell whose comment contains exactly REQUIRED PREPROCESSING, a post-cleaning inspection cell, a separate plotting cell, and a separate model/evaluation cell when training is requested. The preprocessing cell must load the CSV with pandas using na_values, call pd.to_numeric(..., errors=\"coerce\"), and run before every plot or model. Use pandas, numpy, matplotlib, seaborn, scikit-learn, and optionally xgboost. Do not execute code. Do not include shell commands, network calls, filesystem deletion, eval, exec, or subprocess."),
            HumanMessage(content=bounded_json({"plan": state["plan"], "research": state["research"], "insights": state["insights"], "file_name": state["file_name"]})),
        ])
        notebook = {
            "cells": [
                {"cell_type": cell.cell_type, "metadata": {}, "id": uuid.uuid4().hex[:12], "source": _source_lines(cell.source), **({"execution_count": None, "outputs": []} if cell.cell_type == "code" else {})}
                for cell in response.cells
            ],
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        }
        task = state.get("plan", {}).get("task", "exploration")
        valid, reason = _valid_notebook(notebook, task)
        if not valid:
            raise ValueError(reason)
        return {"notebook": notebook, "events": [_event("notebook_writer", "Code subagent wrote and passed notebook structure and safety checks")]}
    except Exception as error:
        return {"notebook": fallback, "llm_errors": [f"code agent: {error}"], "events": [_event("notebook_writer", "Code provider failed validation; used deterministic notebook", "fallback")]}


def _save_notebook(state: NotebookState):
    notebook = state["notebook"]
    task = state.get("plan", {}).get("task", "exploration")
    valid, reason = _valid_notebook(notebook, task)
    if not valid:
        raise ValueError(reason)
    output_path = Path(state["input_path"]).with_name("generated_notebook.ipynb")
    output_path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"notebook_path": str(output_path), "events": [_event("notebook_validate", "Notebook JSON was validated and saved without execution")]}


def build_notebook_graph(checkpointer):
    builder = StateGraph(NotebookState)
    builder.add_node("load_context", _load_context)
    builder.add_node("research_subagent", _research)
    builder.add_node("insights_subagent", _insights)
    builder.add_node("primary_planner", _primary_plan)
    builder.add_node("notebook_writer", _write_notebook)
    builder.add_node("notebook_validate", _save_notebook)
    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", "research_subagent")
    builder.add_edge("research_subagent", "insights_subagent")
    builder.add_edge("insights_subagent", "primary_planner")
    builder.add_edge("primary_planner", "notebook_writer")
    builder.add_edge("notebook_writer", "notebook_validate")
    builder.add_edge("notebook_validate", END)
    return builder.compile(checkpointer=checkpointer)


notebook_graph = build_notebook_graph(SqliteSaver(_connection))


def run_notebook_job(job_id: str, input_path: Path, file_name: str, request: str):
    # Each prompt gets a fresh thread so a second notebook request cannot reuse
    # the previous plan or append stale cells to the new notebook.
    thread_id = f"notebook:{job_id}:{uuid.uuid4().hex}"
    state = {"notebook_thread_id": thread_id, "job_id": job_id, "input_path": str(input_path), "file_name": file_name, "request": request, "events": [], "llm_errors": []}
    return notebook_graph.invoke(state, {"configurable": {"thread_id": thread_id}})
