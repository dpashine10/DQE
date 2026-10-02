# DataQualityEng

DataQualityEng audits CSV files locally before making any changes. The deterministic audit reports missing values, duplicate rows, constant columns, numeric outliers, and a score from 0 to 100. Cleaning is opt-in and always operates on a copy of the input frame.

## Run it

```bash
.venv/bin/python main.py data.csv
.venv/bin/python main.py data.csv --apply --output cleaned.csv
.venv/bin/python main.py data.csv --llm --apply --output cleaned.csv
```

To ask the model to find and fix every issue it can justify, use `--llm --apply`. `--llm` creates the complete cleaning plan; `--apply` approves and applies every validated action in that plan:

```bash
.venv/bin/python main.py evaluation_candidates.csv --llm --apply --output evaluation_candidates.cleaned.csv
```

The original CSV is not overwritten unless you explicitly choose it as the output path.

Use `--llm` when the model should decide context-aware cleaning actions. It receives a compact profile and returns structured actions such as `fill_missing`, `replace_value`, `drop_duplicate_rows`, or `drop_column`. The actions are validated against the real dataframe and are only executed when `--apply` is also present.

Configure any provider or gateway that accepts a chat-completions-style JSON request in `.env`:

```dotenv
LLM_API_KEY=your-api-key
LLM_ENDPOINT=https://provider.example/v1/chat/completions
LLM_MODEL=your-model-name
# Optional: increase this if the provider stops long responses early.
# LLM_MAX_TOKENS=12000
# Optional for providers with non-Bearer authentication:
# LLM_AUTH_HEADER=X-API-Key
# LLM_AUTH_PREFIX=
# Optional provider-specific headers, encoded as one JSON object:
# LLM_EXTRA_HEADERS={"HTTP-Referer":"https://your-app.example","X-Title":"DataQualityEng"}
```

Use `--summary` separately to send the compact audit report to the configured model:

```bash
.venv/bin/python main.py data.csv --summary
```

The LLM does not receive the full CSV or mutate the dataframe directly. It sees column metadata, up to 20 sample rows, and deterministic audit findings. Numeric missing values can use a mean or median; categorical values should come from observed context. Approval is represented by `approved=True` in `apply_recommendations` and by `--apply` in the CLI.

“Wrong” values cannot be inferred safely from a CSV alone. Add a schema or validation rules for constraints such as allowed categories, valid date formats, ranges, and required identifiers before enabling those checks.
