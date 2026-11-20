# DataQualityEng: complete explanation

## 1. The problem

CSV files often look usable while containing subtle problems:

- A missing value is written as `?` instead of being empty.
- A numeric column contains `five`, `abc`, or a negative value.
- A category appears as both `automatic` and `Automatic`.
- A city is misspelled once.
- A year is outside a realistic range.
- An exact record appears twice.
- A value is a statistical outlier that may or may not be wrong.

A human can catch some of these by looking at a spreadsheet, but inspection becomes unreliable as files get wider and longer. DataQualityEng separates measurement, reasoning, approval, and mutation so the agent can find more problems without being allowed to silently rewrite the source.

## 2. The complete workflow

```text
1. User selects a CSV
2. Browser sends it to the FastAPI process on port 8000
3. API saves an immutable input copy under a job ID
4. LangGraph starts a job using that ID as thread_id
5. load confirms the job entered the graph
6. profile infers column kinds and builds a local value catalog
7. audit runs deterministic checks and calculates the score
8. LangChain proposes structured cleaning actions
9. Python validates every proposed action
10. summary explains the result
11. Evidence lets the user select actions
12. API applies only selected actions to a dataframe copy
13. Browser downloads a new cleaned CSV
14. User can request a plotting or model-training notebook
15. Notebook research, insights, planning, and code roles produce an ipynb
```

The graph is intentionally linear. That makes it possible to inspect each step and makes recovery predictable.

## 3. Upload and strict parsing

`backend/csv_io.py` is shared by the API, graph, and CLI. It handles:

- UTF-8 and UTF-16 with a byte-order mark.
- Comma, semicolon, tab, and pipe separators.
- Quoted values.
- Leading zeros in identifiers.
- Duplicate header names.
- Ragged rows.
- Empty files.
- A 512 MiB (`512 * 1024 * 1024` byte) file limit.
- A 1,000,000-row limit.
- A 500-column limit and a 10,000,000-cell limit.

The parser reads CSV cells as strings first. That is important: an identifier like `00127` must not silently become the number `127`, and a literal `?` must remain visible to the audit as a missing placeholder.

The project is intentionally CSV-only. A PDF may contain a table visually, but it does not reliably provide structured rows and columns. A PDF table should be exported to CSV before upload.

## 4. Profiling and the value catalog

The profile records:

- Row and column counts.
- Inferred column kind: identifier, email, date, numeric, or text.
- Missing count.
- Unique count.
- Top values and frequencies.
- Numeric min, max, mean, and median.
- Up to 200 unique values per column in the local profile's bounded catalog excerpt.

The graph also writes a local value catalog, capped at 10,000 frequent values per column to:

```text
backend/storage/<job-id>/value-catalog.json
```

This is the important difference between looking at five sample rows and understanding a column. The catalog is capped at 10,000 frequent values per column to bound retained context. The cleaning model receives compact top-value metadata rather than the complete catalog, while high-cardinality identifier-like columns remain local without flooding model context with IDs.

The catalog is not used as permission to change values. It is evidence. A proposed change still has to use a real column and a value that exists in the uploaded data or is a validated fill value.

Notebook agents do not receive this catalog as one giant prompt. The notebook graph stores the capped catalog locally, skips identifier columns for model context, splits non-identifier values into batches of at most 100 values per column and six columns per batch, and sends at most 20 evenly distributed batches to each research/insights role. Results are merged before planning.

## 5. Deterministic audit

The audit is in `backend/dataquality.py`. It runs before any model call.

### Missing values

These are treated as missing:

```text
blank, ?, null, none, nan, n/a, missing
```

For a numeric column, the candidate fill value is its median. For a categorical/text column, the candidate is its observed mode. Identifier, email, and date columns are not automatically filled because guessing those values is risky.

### Numeric validity

Numeric columns are identified by their names and values. The audit reports values that cannot become finite numbers. Strong numeric names such as `price`, `amount`, `mileage`, `age`, `year`, `quantity`, `count`, and `seats` also get domain checks:

- Non-negative for quantities, prices, mileage, counts, and similar fields.
- Age between 0 and 120.
- Year between 1000 and next year.
- Whole numbers for age, seats, quantities, and counts.

For clearly numeric columns, invalid values receive a median-replacement candidate with lower confidence. The user can leave that candidate unchecked.

### Dates and emails

Date columns are parsed using mixed date formats. Invalid dates and dates before 1900 or after tomorrow are reported. Email columns use a simple syntax check for a local part, `@`, and a domain.

### Categories and consistency

Whitespace is reported. Repeated case variants such as `petrol` and `Petrol` get a high-confidence normalization candidate. Similar values such as `Mumabi` and `Mumbai` get a warning and a review-only suggestion, not an automatically selectable replacement, because fuzzy similarity is evidence rather than proof.

### Duplicates

The generic column named `id` is checked for repeated values. Columns ending in `_id`, such as `customer_id`, are treated as possible foreign keys and are allowed to repeat. Exact duplicate rows are safe candidates for removal.

### Outliers

Numeric outliers use the 1.5×IQR rule. They appear as warnings. An unusual value is not necessarily wrong, so an outlier alone does not receive the same penalty as an invalid value.

## 6. The score

The score uses affected rows instead of adding one penalty for every issue object:

```text
100 × (1 − (error rows + 0.25 × outlier-only rows) / total rows)
```

If one row has a missing value, an invalid number, and an impossible range, it counts as one affected error row. This prevents duplicate penalties from hiding behind a number that still looks like the 90s.

Constant columns are reported as informational. A constant column could be useless, but it could also be a legitimate batch or source marker, so the engine does not treat it as corruption automatically.

## 7. LangChain

LangChain is used in the graph's `plan` node. It creates a `ChatOpenAI` client for the configured OpenAI-compatible endpoint and calls:

```python
planner = model.with_structured_output(CleaningPlan)
```

The result must have an `actions` list. Each action has:

```text
action
column
value / from_value / to_value
reason
confidence
```

The model receives a compact deterministic audit, profile metadata, and a five-row preview. Its prompt tells it to search for null-like, impossible, out-of-range, misspelled, and inconsistent values while avoiding unsupported guesses. The uploaded file bytes and full local catalog are not sent wholesale to the provider.

LangChain proposes. It does not mutate a dataframe and it cannot run arbitrary code.

When the provider is unavailable, the graph keeps deterministic candidates and records `llm_error`. The product remains useful without a model key.

## 8. LangGraph

The graph in `backend/graph.py` contains these nodes:

```text
START → load → profile → audit → plan → summarize → END
```

Each completed node is checkpointed in:

```text
backend/storage/checkpoints.sqlite
```

The upload's job ID is the graph `thread_id`. That means a retry can recover the profile and audit context instead of starting from an empty conversation.

The retry endpoint is:

```text
POST /api/jobs/<job-id>/retry
```

The retry starts at the plan boundary when prior audit context is available. It cannot reconstruct tokens that the model provider never returned, but it does preserve every completed local step and retries the model decision with that context.

## 9. Evidence and human selection

The plan is not silently applied. The frontend renders each validated action as a checkbox with:

- Action name.
- Column.
- Reason.
- Confidence.

The user may uncheck a recommended action. Actions marked `nothing` or `not_recommended` remain visible as evidence but are disabled for export. The frontend sends the selected action indexes to:

```text
POST /api/jobs/<job-id>/apply
```

The API maps those indexes back to the checkpointed plan, validates the selected subset again, applies it to `df.copy(deep=True)`, and returns a new CSV. A user can therefore accept an easy whitespace fix while skipping a lower-confidence median replacement.

## 10. Applying changes safely

Supported actions are:

```text
fill_missing
replace_value
trim_whitespace
drop_rows_with_missing
drop_duplicate_rows
drop_column
```

Before execution, the backend checks:

- Action type is supported.
- Referenced columns still exist.
- Fill and replacement values are scalar and not missing.
- Replacement source values exist.
- The last column cannot be dropped.
- The action list is bounded.

The source dataframe is never mutated. The export is always created from a copy.

## 11. What makes this different

The unique product direction is not “a chatbot that cleans a spreadsheet.” It is a local-first, evidence-first repair system:

1. **A score that counts affected rows once.** It is harder for overlapping issue penalties to create false confidence.
2. **A value atlas.** The agent sees column vocabulary instead of only a few sample rows.
3. **Confidence-aware fixes.** A clear case normalization can be selected separately from a speculative numeric repair.
4. **A human-readable repair ledger.** Every action has a reason and can be accepted or rejected.
5. **A deterministic safety boundary.** The model proposes; Python validates; the user approves; pandas writes a copy.
6. **Recoverable work.** A failed model request does not erase the profile and audit.
7. **Privacy by default.** Data stays local except for the compact context sent to the configured provider.

## 12. Strong next ideas

The best next features are:

- A schema editor for explicit min/max, allowed categories, required fields, and primary keys.
- Before/after diff previews for every selected action.
- A repair ledger exported beside the cleaned CSV.
- Dataset fingerprints to detect column drift across repeated uploads.
- A “safe automatic fixes only” mode that accepts only high-confidence actions.
- A catalog search tool for high-cardinality columns instead of putting every identifier into a prompt.
- Row-level issue navigation from an audit signal to the exact source preview.

These ideas preserve the same boundary: the model can reason about evidence, but the application remains responsible for deterministic validation and file mutation.

## 13. Notebook generation

Notebook generation is a second graph tied to the original audit job. It is not mixed into cleaning because writing code and mutating data have different safety boundaries.

```text
notebook request
  → load dataset profile and audit
  → research subagent
  → data-insights subagent
  → primary planner
  → notebook code writer
  → notebook safety validator
  → .ipynb download
```

The research role interprets the request and considers targets, features, plots, libraries, metrics, and model parameters. The data-insights role reads the real profile and manageable value catalogs to identify useful columns and exclude identifiers. The primary role combines both reports with the request. The code role writes markdown and code cells for pandas, NumPy, Matplotlib, Seaborn, scikit-learn, and optional XGBoost.

The generated notebook includes separate dependency, preprocessing, inspection, plotting, and optional model cells. It contains dependency guidance, `DATA_PATH`, `pd.read_csv`, data inspection, plots, correlation heatmaps when possible, preprocessing, train/test split, and model metrics when requested. The application does not execute notebook code. It validates notebook JSON, requires staged code and markdown cells, requires preprocessing before plotting or training, and rejects obvious unsafe patterns such as subprocess, eval, exec, sockets, network requests, and deletion calls. If a code provider fails structural validation, a deterministic writer creates the notebook instead.

Preprocessing is a hard ordering rule. The generated cell runs before any plot or model cell and uses `keep_default_na=False` plus an explicit marker list so values such as `?` and `N/A` cannot silently remain strings. It normalizes whitespace, converts known numeric columns with `pd.to_numeric(errors="coerce")`, parses dates, reports missingness, and leaves the model pipeline to impute remaining feature gaps. Targets are coerced separately and invalid target rows are removed before splitting. Very small or single-class datasets skip training with an explanation instead of raising a confusing model error.

Notebook generation is queued in a background worker, just like the upload-time model plan. The browser polls `/api/jobs/{job_id}/notebook/status`, so a slow research or code provider cannot block the API or make the upload screen appear frozen.

A catalog of up to 10,000 frequent unique values per column is written locally, but it is not sent as one huge prompt. `_catalog_batches` skips identifier-like columns and splits other values into bounded batches. The research and insights roles receive an evenly distributed sample of at most 20 batches each and merge their structured results. This preserves representative category awareness while bounding calls and context size; a provider can still reject a request if its own limit is lower.

## 14. Multiple notebook providers

`backend/providers.py` provides role-specific model configuration. Notebook roles are `RESEARCH`, `INSIGHTS`, `PRIMARY`, and `CODE`. Each role can have its own API key, endpoint, model, timeout, and headers. Missing role values inherit from the base `LLM_*` configuration. This supports a cheaper model for data research, a stronger planner, and a code-focused model without changing the graph. In this version each provider must expose an OpenAI-compatible chat API.

## Model context budget and impact preview

All dataset-derived model contexts are encoded as valid JSON with a 48,000-byte ceiling. Long strings and lists are progressively shortened; a truncation notice tells the model the local audit has fuller findings. System messages, structured-output schemas and responses add overhead, so provider limits can still be lower. Provider failures fall back to local results. The deterministic audit always covers the full accepted dataset.

The impact preview applies selected recommended actions to an in-memory copy and re-audits it. It reports before/after scores, row and column counts, and error-row counts without saving the copy. A better score does not establish that a guessed replacement is semantically correct.
