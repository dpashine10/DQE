# LangChain and LangGraph in DataQualityEng

This project uses the two libraries for different jobs:

- **LangChain** talks to the configured chat-completions model and asks for a typed `CleaningPlan`.
- **LangGraph** controls the order of work and saves state in SQLite after each node.

The code intentionally keeps the graph small. It is easier to debug than a large collection of agents and tools.

## The graph

```text
upload CSV
    ↓
load
    ↓
profile
    ↓
deterministic audit
    ↓
LangChain cleaning plan
    ↓
summary
    ↓
review and approve
    ↓
clean a copy and download
```

The implementation is in `backend/graph.py`.

### 1. Load

The API stores the upload under `backend/storage/<job-id>/input.csv`. The job ID becomes the LangGraph `thread_id`.

### 2. Profile

`profile_dataframe` records the row count, inferred kind, missing count, unique count, top values, basic numeric statistics, and a bounded unique-value catalog. The graph also stores up to 10,000 frequent values per column locally for the job. The cleaning model receives compact top-value metadata and a bounded audit excerpt; it does not receive the complete catalog or uploaded file bytes.

### 3. Audit

`audit_dataframe` is deterministic. It finds measured problems before the model runs. This makes the score reproducible and gives the model evidence instead of asking it to “look at the CSV” vaguely.

### 4. Plan

When `LLM_API_KEY`, `LLM_ENDPOINT`, and `LLM_MODEL` are present, LangChain uses `ChatOpenAI.with_structured_output(CleaningPlan)`. The model can suggest only the small action vocabulary understood by the backend:

```text
fill_missing
replace_value
trim_whitespace
drop_rows_with_missing
drop_duplicate_rows
drop_column
```

The plan is passed to `validate_recommendations` before it reaches the UI. A made-up column, unsupported action, missing replacement value, unsafe scalar, or action against a removed column is rejected.

If the model is not configured or its request fails, the graph keeps the deterministic recommendations. The run is still useful and the API includes `llm_error` when a model request failed. Deterministic candidates include median replacements for clearly numeric invalid values, but those lower-confidence actions remain selectable in Evidence.

### 5. Summary

The graph creates a deterministic summary automatically. The UI also has an **Ask agent for summary** control. When model settings are present, that control asks LangChain for a concise grounded explanation; otherwise it returns the deterministic summary without failing.

Notebook generation is a separate graph in `backend/notebook.py`:

```text
request → load_context → research_subagent → insights_subagent
        → primary_planner → notebook_writer → notebook_validate
```

The subagents have explicit roles rather than an uncontrolled swarm. The research role thinks about what the task requires, the insights role reads the real columns and value catalogs, the primary role chooses the target/features/plot plan, and the code role writes the cells. The final node validates the notebook and saves it. No graph node executes generated code.

The notebook graph keeps capped value catalogs on disk (up to 10,000 frequent values per column) but sends only a bounded, evenly distributed sample of batches to each research/insights role. Identifier columns are not useful model context, so they stay local. Other columns are split into chunks of at most 100 values per column and six columns per batch; at most 20 batches are sent per role. The structured responses are merged before planning, preventing oversized requests and unbounded API-call counts.

## What recovery means

LangGraph uses `backend/storage/checkpoints.sqlite`. With a stable job ID, each completed node is saved under the same thread. If a model call fails, the profile and audit state already exist in the checkpoint. The retry endpoint can resume the same thread instead of losing the earlier context.

```text
POST /api/jobs/<job-id>/retry
```

There is an important limit: a library cannot reconstruct tokens that a provider never returned. LangGraph preserves completed context and lets the node run again; the provider still needs to complete the retried request.

## Why not make every step an agent?

Data cleaning benefits from fixed boundaries:

1. pandas measures the data.
2. the model proposes.
3. Python validates.
4. a human approves.
5. pandas writes a new file.

That shape is safer and simpler than letting a model call arbitrary Python or decide when validation should be skipped.

The final apply request carries selected action indexes from the Evidence panel. The API validates only that selected subset again, so unchecking an action genuinely prevents it from reaching the dataframe copy.

## Adding stronger rules later

The next useful extension is a schema input, for example:

```json
{
  "price": {"type": "number", "min": 0},
  "year": {"type": "integer", "min": 1900, "max": 2026},
  "region": {"type": "category", "allowed": ["North", "South", "East", "West"]}
}
```

Those rules should be deterministic checks. LangChain can explain or prioritize the findings, but should not invent the allowed range.
