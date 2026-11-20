# DataQualityEng

DataQualityEng is an evidence-first CSV auditing and cleaning workspace. It finds quality problems that are easy to miss, explains why they matter, proposes safe fixes, and lets a person choose which fixes to approve. The original upload is never overwritten.

The project combines:

- **Deterministic Python checks** for repeatable scores and validation.
- **LangChain** for structured, context-aware cleaning proposals.
- **LangGraph** for the ordered workflow and SQLite checkpoints.
- **A plain HTML/CSS/JavaScript frontend** with no frontend build step.

## Start locally

Requirements: **Python 3.14+** and [`uv`](https://docs.astral.sh/uv/).

Install dependencies once:

```bash
uv sync
```

Start the app:

```bash
./.venv/bin/python start.py
# or
uv run start.py
```

Open <http://127.0.0.1:8000> and upload a CSV. The same FastAPI process serves the frontend and API, so there is no second static server to keep track of.

The file picker is CSV-only. PDF files are not data tables that this engine can audit; export a PDF table as CSV first.

### Deployment command

The server defaults to `127.0.0.1`, which is correct for local use. For a container or hosted deployment, bind it to the platform interface:

```bash
HOST=0.0.0.0 PORT=8000 uv run start.py
```

If a reverse proxy or platform gateway sits in front of the app, configure its request-body limit and timeout to allow at least **512 MiB** uploads and enough time for deterministic profiling. The application itself enforces the limits below.

Deployment security notes:

- This app has no built-in user authentication or authorization. Put it behind your identity-aware proxy or keep it on a private network before exposing it beyond a trusted workstation.
- `backend/storage/` contains uploaded CSVs, local value catalogs, cleaned exports, and SQLite checkpoints. It must be writable and protected. If the deployment filesystem is ephemeral, jobs and checkpoints disappear on restart.
- Keep `.env` private. Configured model providers may receive compact dataset values, column names, audit excerpts, and notebook context as described below.

## Input limits

| Limit | Maximum | Meaning |
| --- | ---: | --- |
| File size | 512 MiB (`512 * 1024 * 1024` bytes) | Checked before CSV parsing; the limit includes uploaded file bytes. |
| Data rows | 1,000,000 | Header is not counted. Blank lines are ignored. |
| Columns | 500 | Header fields are counted. |
| Data cells | 10,000,000 | Rows multiplied by columns; applies alongside the row and column limits. |
| Notebook request | 1,000 characters | Natural-language request sent to the notebook graph. |
| Cleaning actions | 1,000 | Maximum actions accepted by the validated executor. |
| Browser preview | 5 rows | The source remains local; only five rows are shown in the UI. |

The parser accepts UTF-8/UTF-8-BOM and UTF-16 with a byte-order mark, detects comma/semicolon/tab/pipe separators, preserves leading zeros, rejects duplicate headers, and rejects ragged rows. A file must contain a header and at least one data row.

Uploads are copied to disk in chunks and CSV validation reads the file as a stream. The resulting dataframe and the audit still run in memory. The cell limit bounds wide files, but the limits are ceilings rather than a guarantee that every accepted file will fit on every machine; long values, concurrent jobs and cleaning copies increase memory use. For larger workloads, a disk-backed execution engine and background job queue are needed.

## What happens after upload

```text
CSV upload
  → strict parsing and size checks
  → dataset profile
  → local value catalog (up to 10,000 values per column)
  → deterministic audit and score
  → LangChain structured cleaning plan
  → user selects fixes in Evidence
  → validated actions run on a copy
  → cleaned CSV download
  → optional notebook request
  → research subagent + insights subagent + primary planner + code writer
  → validated .ipynb download
```

The original upload is never overwritten. The **Evidence** panel shows every proposed action with its reason and confidence. Uncheck anything you do not want to apply before choosing **Approve & download cleaned CSV**.

**Preview impact** applies your current selection to an in-memory copy and re-runs the audit, showing the score, row count and error-row count before and after. It does not save or modify the dataset. A higher score does not prove that an inferred replacement is correct; inspect the proposed actions and their values.

Upload returns the deterministic score and audit immediately. If a model is configured, the agent plan continues in the background and the interface polls until it is ready. A slow or unavailable provider therefore does not block the file upload.

The deterministic audit always runs locally before any model call. Without model settings, the full audit and deterministic cleaning plan still work.

## What it checks

The audit looks for:

- Blank cells and placeholders such as `?`, `null`, `none`, `NaN`, and `N/A`.
- Invalid numeric values, negative values in obvious non-negative columns, impossible ages, and impossible years.
- Numeric values that are far outside the interquartile range.
- Invalid dates and dates outside the plausible 1900-to-tomorrow range.
- Invalid email syntax.
- Leading or trailing whitespace.
- Inconsistent category casing such as `automatic` and `Automatic`.
- Possible category typos such as `Mumabi` and `Mumbai`.
- Repeated values in a generic `id` column.
- Exact duplicate rows.
- Constant columns as informational findings.

The engine automatically creates conservative candidates for filling missing values, dropping rows with missing required identifiers, trimming whitespace, standardizing clear category variants, dropping exact duplicates, and replacing clearly invalid numeric values with the median of valid values. The UI lets you approve or skip recommended candidates. Possible category typos are review-only and are never automatically selected.

The model can suggest additional actions, but it cannot execute arbitrary Python. Every model action must pass backend validation first.

## Generate a notebook

After uploading a CSV, use the **Notebook lab** prompt. Examples:

```text
Create an exploratory notebook with bar charts and a correlation heatmap.
Create an ipynb file for training a Regression model on this dataset.
Create an XGBoost classification notebook and explain the target and features.
```

The notebook workflow does not run code. It creates a Jupyter `.ipynb` file with separate dependency, preprocessing, inspection, plotting, and optional model-training cells. It first asks a research role what the task requires, asks a data-insights role to inspect the columns and candidate target, asks a primary role to choose the plan, and asks a code role to write the notebook. If a model role is unavailable or writes a notebook that fails the structural checks, deterministic fallback code is generated.

Every generated notebook puts a required preprocessing cell before inspection plots or model training. It reads `?`, `N/A`, `NA`, `null`, `none`, `nan`, blank, and similar markers as missing, trims text, coerces known numeric columns with `errors="coerce"`, parses dates, and prints the missing-value report. The model pipeline also imputes remaining feature missingness, drops unusable target rows, and skips training when too few valid rows remain.

Download the resulting notebook, place it beside the CSV, review the target/path assumptions, and run it yourself in a controlled notebook environment.

## Scoring

Scores are row-based so many issues in one row do not create a misleading pile of separate deductions:

```text
100 × (1 − (error rows + 0.25 × outlier-only rows) / total rows)
```

Each row counts once. Outliers alone are warnings because unusual does not automatically mean incorrect. Type errors, missing values, impossible ranges, duplicate identifiers, and duplicate records count as errors.

More precisely:

- `error_rows` is the union of rows with completeness, validity, consistency, or uniqueness errors.
- `warning_only_rows` is the set of outlier rows that are not already error rows.
- The overall score subtracts one point per error row and one-quarter point per outlier-only row, then clamps the result to `0.0`–`100.0`.
- Dimension scores use the same row-union idea for completeness, validity, consistency, uniqueness, and outliers. Constant columns are informational and do not affect the score.
- An empty dataset scores `0.0`; a non-empty dataset with no measured issues scores `100.0`.

The score is a strong measured signal, not proof of business correctness. Add explicit schema rules for company-specific requirements such as allowed regions, approved price ranges, and required identifiers.

## Model configuration

Copy `.env.example` to `.env` and configure an OpenAI-compatible chat-completions provider:

```dotenv
LLM_API_KEY=your-api-key
LLM_ENDPOINT=https://provider.example/v1/chat/completions
LLM_MODEL=your-model-name
# Optional:
# LLM_TIMEOUT_SECONDS=90
# LLM_MAX_TOKENS=12000
# LLM_AUTH_HEADER=X-API-Key
# LLM_AUTH_PREFIX=
# LLM_EXTRA_HEADERS={"HTTP-Referer":"https://your-app.example"}
```

Notebook roles can use different providers. Each role falls back to the base `LLM_*` values when its role-specific values are omitted:

```dotenv
AGENT_RESEARCH_PROVIDER=openai_compatible
AGENT_RESEARCH_API_KEY=research-key
AGENT_RESEARCH_ENDPOINT=https://provider.example/v1/chat/completions
AGENT_RESEARCH_MODEL=research-model
AGENT_INSIGHTS_MODEL=insights-model
AGENT_PRIMARY_MODEL=primary-model
AGENT_CODE_MODEL=code-model
```

The roles are `RESEARCH`, `INSIGHTS`, `PRIMARY`, and `CODE`. `AGENT_*_PROVIDER` is currently a label/guard for OpenAI-compatible providers; the endpoint controls the actual gateway.

## How the agents use the data

The Python application reads and audits the uploaded CSV locally. The configured provider does **not** receive the uploaded file bytes or arbitrary Python execution access. When a model is configured, the main cleaning planner receives a compact profile, a bounded audit excerpt, and the five-row preview. That context can still contain dataset values such as column names and top categories, so do not send sensitive data to an external provider unless that is acceptable for your deployment.

Notebook generation keeps a catalog of up to 10,000 frequent values per column in the local job folder. It excludes identifier-like columns from notebook-agent context, splits other values into batches of at most 100 values per column and six columns per batch, and sends at most 20 evenly distributed batches to each of the research and insights roles. If a catalog has more than 20 local batches, the capped catalog is retained locally but the model receives a representative spread rather than an oversized request. The primary planner and code writer receive the compact reports, not the full catalog. Structured model responses are validated before they affect the job.

Every dataset-derived model context is serialized as valid JSON with a **48,000-byte maximum**. Long strings and lists are progressively shortened, with a truncation notice; this is a byte budget, not a guarantee that every provider model will accept the request. Provider schemas, system instructions and output tokens add overhead. Local auditing still covers the entire accepted dataset, and provider errors fall back to local results. Model suggestions based on incomplete context must be reviewed.

Agent execution is bounded and asynchronous:

- Upload: deterministic audit first; the optional primary cleaning plan runs in the background.
- Notebook: research and insights use at most 20 calls each, followed by one primary-planner call and one code-writer call.
- Summary: the optional summary button makes one additional primary-agent call.
- Provider failure, timeout, malformed structured output, or unsafe notebook structure falls back to deterministic behavior.

DataQualityEng never executes generated notebook code.

Without model settings, the deterministic audit and cleaning candidates still work. With model settings, use an OpenAI-compatible endpoint configured through `.env`; the endpoint and gateway must also accept the deployment's request size and timeout.

## API endpoints

```text
GET  /api/health
POST /api/jobs
POST /api/jobs/sample
GET  /api/jobs/{job_id}
POST /api/jobs/{job_id}/retry
POST /api/jobs/{job_id}/preview
POST /api/jobs/{job_id}/apply
GET  /api/jobs/{job_id}/report
POST /api/jobs/{job_id}/summary
POST /api/jobs/{job_id}/notebook
GET  /api/jobs/{job_id}/notebook/status
GET  /api/jobs/{job_id}/notebook
```

`POST /api/jobs/{job_id}/preview` and `POST /api/jobs/{job_id}/apply` accept an optional body:

```json
{"selected_indexes": [0, 2, 4]}
```

If the body is omitted, all actions classified as `recommended` are applied. Actions classified as `nothing` or `not_recommended` are never applied automatically. The frontend always sends the actions selected in Evidence.

## Project structure

```text
frontend/
  index.html              Workspace markup
  styles.css              Responsive visual system
  app.js                  Upload, evidence selection, and downloads
backend/
  api.py                  FastAPI upload and export endpoints
  graph.py                LangGraph workflow and SQLite checkpoints
  csv_io.py               Strict CSV parsing and limits
  dataquality.py          Profiling, checks, score, and safe operations
  providers.py            Role-based model configuration
  notebook.py             Notebook subagents, graph, and ipynb writer
  LLMtools.py             LangChain helpers for the CLI
  main.py                 Optional CLI entrypoint
  tests/                  Regression tests
start.py                  Starts the single FastAPI server in one command
explanation.md            Full end-to-end explanation
LANGCHAIN_LANGGRAPH.md    Focused LangChain/LangGraph notes
```

## Optional CLI

```bash
./.venv/bin/python -m backend.main data.csv
./.venv/bin/python -m backend.main data.csv --apply --output cleaned.csv
./.venv/bin/python -m backend.main data.csv --summary
```

`--summary` prints the same deterministic explanation used by the graph, so a frontend is not required to understand a run.

## Verification

Run the checks with:

```bash
./.venv/bin/python -m unittest discover -s backend/tests -v
./.venv/bin/python -m compileall -q backend start.py
node --check frontend/app.js
uv lock --check
```

The suite covers score behavior, parser limits, placeholder handling, cleaning safety, catalog batching, notebook structure/order, and the FastAPI health/upload/report/apply/notebook flow. A live provider call is intentionally not part of the test suite; verify provider credentials and endpoint compatibility separately in the deployment environment.

For the full design and data-flow explanation, read [explanation.md](explanation.md).
