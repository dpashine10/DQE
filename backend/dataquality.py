"""Explainable CSV checks and a small, validated set of cleaning operations."""

import math
import re
from datetime import date
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

MISSING = {"", "?", "null", "none", "nan", "n/a", "missing"}
NUMERIC_NAMES = {"age", "year", "seats", "quantity", "price", "amount", "mileage", "count", "weight", "height"}
MAX_ACTIONS = 1_000


def missing_mask(series):
    return series.isna() | series.astype(str).str.strip().str.casefold().isin(MISSING)


def column_kind(column, series):
    name = re.sub(r"[^a-z0-9]+", "_", column.lower()).strip("_")
    words = set(name.split("_"))
    if name == "id" or name.endswith("_id") or words & {"zip", "postal", "phone", "code"}:
        return "identifier"
    if "email" in words:
        return "email"
    if words & {"date", "dob", "timestamp"} or any(token in name for token in ("date", "time")):
        return "date"
    values = series[~missing_mask(series)]
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_ratio = numeric.notna().mean() if len(values) else 0
    mostly_numeric = len(values) <= 5 and numeric.notna().sum() >= 2
    if words & NUMERIC_NAMES or (len(values) >= 2 and (numeric_ratio >= 0.8 or mostly_numeric)):
        return "numeric"
    return "text"


def profile_dataframe(df):
    columns = []
    for column in df.columns:
        values = df[column][~missing_mask(df[column])].astype(str)
        info = {
            "name": column, "kind": column_kind(column, df[column]),
            "missing": int(missing_mask(df[column]).sum()), "unique": int(values.nunique()),
            "top_values": [{"value": str(value)[:200], "count": int(count)}
                           for value, count in values.value_counts().head(8).items()],
        }
        catalog_values = values.value_counts().index.tolist()
        info["value_catalog"] = {
            "total_unique": len(catalog_values),
            "complete": len(catalog_values) <= 200,
            "sampled": len(catalog_values) > 200,
            "values": [str(value)[:200] for value in catalog_values[:200]],
        }
        numbers = pd.to_numeric(values, errors="coerce")
        numbers = numbers[np.isfinite(numbers)]
        if info["kind"] == "numeric" and len(numbers):
            info["statistics"] = {"min": float(numbers.min()), "max": float(numbers.max()),
                                  "mean": float(numbers.mean()), "median": float(numbers.median())}
        columns.append(info)
    return {"rows": len(df), "columns": columns}


def _missing_decision(column):
    words = set(re.sub(r"[^a-z0-9]+", "_", column.lower()).split("_"))
    if words & {"age", "year"}:
        return "not_recommended", "Do not invent a person's age or a record's year from a median."
    return "recommended", "The observed mode or median is a reasonable first repair for this column."


def _drop_missing_decision(column):
    normalized = re.sub(r"[^a-z0-9]+", "_", column.lower()).strip("_")
    if normalized == "id" or normalized.endswith("_id"):
        return "recommended", "A row without its identifier cannot be reliably joined or traced."
    return "not_recommended", "Dropping a row loses data; confirm that this field is required before doing that."


def audit_dataframe(df):
    df = df.reset_index(drop=True)
    rows = len(df)
    issues, recommendations = [], []
    affected = {key: np.zeros(rows, dtype=bool) for key in ("completeness", "validity", "consistency", "uniqueness", "outliers")}

    def add(kind, column, mask, dimension, message, severity="error", counts_as_error=True):
        mask_array = np.asarray(mask, dtype=bool)
        count = int(mask_array.sum())
        if not count:
            return
        if dimension and counts_as_error:
            affected[dimension] |= mask_array
        examples = [{"row": int(index) + 2, "value": str(df.at[index, column])[:160] if column else "duplicate record"}
                    for index in np.flatnonzero(mask_array)[:5]]
        issues.append({"type": kind, "column": column, "count": count,
                       "rate": round(count / max(rows, 1), 4), "severity": severity,
                       "dimension": dimension, "message": message, "examples": examples})

    for column in df.columns:
        series = df[column]
        text = series.astype(str)
        missing = missing_mask(series)
        present = ~missing
        kind = column_kind(column, series)
        words = set(re.sub(r"[^a-z0-9]+", "_", column.lower()).split("_"))
        add("missing_values", column, missing, "completeness", "Empty cells or missing-value placeholders.")
        if missing.any() and kind in {"identifier", "email", "date"}:
            decision, decision_reason = _drop_missing_decision(column)
            recommendations.append({"action": "drop_rows_with_missing", "column": column,
                                    "reason": f"Drop rows where {column} is missing instead of inventing a value.",
                                    "confidence": 0.85 if decision == "recommended" else 0.55,
                                    "decision": decision, "decision_reason": decision_reason})
        if missing.any() and "age" in words:
            recommendations.append({"action": "drop_rows_with_missing", "column": column,
                                    "reason": f"Drop rows where {column} is missing instead of inventing an age.",
                                    "confidence": 0.55, "decision": "not_recommended",
                                    "decision_reason": "Dropping a row loses data and the column may not be required."})
        if missing.any() and kind not in {"identifier", "email", "date"}:
            values = series[present]
            if kind == "numeric":
                numbers = pd.to_numeric(values, errors="coerce")
                finite_values = numbers[np.isfinite(numbers)]
                if words & NUMERIC_NAMES:
                    finite_values = finite_values[finite_values >= 0]
                if "age" in words:
                    finite_values = finite_values[(finite_values <= 120) & finite_values.mod(1).eq(0)]
                if "year" in words:
                    finite_values = finite_values[(finite_values >= 1000) & (finite_values <= date.today().year + 1) & finite_values.mod(1).eq(0)]
                if words & {"seats", "quantity", "count"}:
                    finite_values = finite_values[finite_values.mod(1).eq(0)]
                numbers = finite_values
                value = float(numbers.median()) if len(numbers) else None
                strategy = "median"
            else:
                modes = values.mode()
                value = str(modes.iloc[0]) if len(modes) else None
                strategy = "mode"
            if value is not None:
                decision, decision_reason = _missing_decision(column)
                recommendations.append({"action": "fill_missing", "column": column, "value": value,
                                        "reason": f"Fill {int(missing.sum())} missing cells with the observed {strategy}.",
                                        "confidence": 0.7, "decision": decision,
                                        "decision_reason": decision_reason})

        spaces = present & text.ne(text.str.strip())
        add("surrounding_whitespace", column, spaces, "consistency", "Leading or trailing whitespace.")
        if spaces.any():
            recommendations.append({"action": "trim_whitespace", "column": column,
                                    "reason": "Remove surrounding whitespace.", "confidence": 1.0,
                                    "decision": "recommended", "decision_reason": "Whitespace changes the representation, not the meaning."})

        if kind == "numeric":
            numbers = pd.to_numeric(series, errors="coerce")
            finite = pd.Series(np.isfinite(numbers), index=df.index)
            add("invalid_number", column, present & ~finite, "validity", "Expected a finite number; found text or infinity.")
            domain_bad = pd.Series(False, index=df.index)
            if words & NUMERIC_NAMES:
                domain_bad |= numbers.lt(0)
            if "age" in words:
                domain_bad |= numbers.gt(120) | numbers.mod(1).ne(0)
            if "year" in words:
                domain_bad |= numbers.lt(1000) | numbers.gt(date.today().year + 1) | numbers.mod(1).ne(0)
            if words & {"seats", "quantity", "count"}:
                domain_bad |= numbers.mod(1).ne(0)
            add("suspect_range", column, present & finite & domain_bad, "validity",
                "Outside the built-in range inferred from this column name; confirm the domain rule.")
            values = numbers[present & finite & ~domain_bad]
            if len(values):
                median = float(values.median())
                repairable = present & (~finite | domain_bad)
                if words & NUMERIC_NAMES or "age" in words or "year" in words:
                    for raw_value in text[repairable].drop_duplicates().head(50):
                        recommendations.append({"action": "replace_value", "column": column,
                                                "from_value": str(raw_value), "to_value": median,
                                                "reason": f"Replace an invalid {column} value with the median of valid values; review before approval.",
                                                "confidence": 0.55, "decision": "not_recommended",
                                                "decision_reason": "A median is a fallback, not proof of the intended value."})
            if kind != "identifier" and len(values) >= 4:
                q1, q3 = values.quantile([0.25, 0.75])
                spread = q3 - q1
                # Zero IQR is useful for detecting isolated deviations in otherwise repeated measurements.
                outliers = (numbers.lt(q1 - 1.5 * spread) | numbers.gt(q3 + 1.5 * spread))
                add("numeric_outliers", column, present & finite & ~domain_bad & outliers,
                    "outliers", "Outside the 1.5 × IQR range; unusual does not necessarily mean incorrect.", "warning")
        elif kind == "date":
            parsed = pd.to_datetime(series.where(present), errors="coerce", format="mixed", utc=True)
            add("invalid_date", column, present & parsed.isna(), "validity", "Date cannot be parsed.")
            date_range_bad = present & parsed.notna() & ((parsed.dt.year < 1900) | (parsed > pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)))
            add("suspect_date_range", column, date_range_bad, "validity", "Date is outside the plausible 1900-to-tomorrow range.")
        elif kind == "email":
            valid = text.str.strip().str.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
            add("invalid_email", column, present & ~valid, "validity", "Email address has invalid syntax.")

        if kind == "text":
            values = text[present].str.strip()
            # Case standardization applies only to repeated categories, not free-form prose.
            if values.nunique() <= 30:
                for _, group in values.groupby(values.str.casefold()):
                    counts = group.value_counts()
                    if len(counts) > 1:
                        canonical = str(counts.index[0])
                        variants = list(counts.index[1:])
                        add("inconsistent_category", column, present & text.str.strip().isin(variants),
                            "consistency", f"Case variants of '{canonical}' occur in the same category.")
                        for variant in variants:
                            recommendations.append({"action": "replace_value", "column": column,
                                                    "from_value": str(variant), "to_value": canonical,
                                                    "reason": "Use the most common case for this category.", "confidence": 0.95,
                                                    "decision": "recommended", "decision_reason": "Only capitalization differs and an observed canonical value exists."})
                categories = list(values.drop_duplicates())
                for left_index, left in enumerate(categories):
                    for right in categories[left_index + 1:]:
                        if len(left) < 4 or len(right) < 4 or left.casefold() == right.casefold():
                            continue
                        similarity = SequenceMatcher(None, left.casefold(), right.casefold()).ratio()
                        if similarity >= 0.82:
                            add("possible_category_typo", column, present & text.str.strip().eq(right), "consistency",
                                f"'{right}' is very similar to '{left}'; confirm the intended category.",
                                "warning", counts_as_error=False)
                            recommendations.append({"action": "replace_value", "column": column,
                                                    "from_value": right, "to_value": left,
                                                    "reason": f"Replace possible typo '{right}' with observed category '{left}'.",
                                                    "confidence": round(similarity, 2), "decision": "not_recommended",
                                                    "decision_reason": "Similar categories may have different meanings; confirm the intended value before changing it."})
        # Only a generic row id is assumed unique. Foreign keys such as customer_id may repeat.
        if column.strip().casefold() == "id":
            add("duplicate_identifier", column, present & series.duplicated(keep=False), "uniqueness",
                "Repeated values in the id column. Confirm whether this is a primary key.")
        if present.any() and series[present].nunique() == 1 and rows > 1:
            add("constant_column", column, present, None, "Only one observed value; informational, no score penalty.", "info")

    duplicates = df.duplicated(keep="first")
    add("duplicate_rows", None, duplicates, "uniqueness", "Exact duplicate record after the first occurrence.")
    if duplicates.any():
        recommendations.append({"action": "drop_duplicate_rows", "reason": "Remove exact duplicate records and keep the first occurrence.",
                                "confidence": 1.0, "decision": "recommended",
                                "decision_reason": "The full rows are identical, so one copy is redundant."})

    errors = np.logical_or.reduce([affected[key] for key in affected if key != "outliers"])
    warning_only = affected["outliers"] & ~errors
    score = round(100 * (1 - (int(errors.sum()) + 0.25 * int(warning_only.sum())) / rows), 1) if rows and len(df.columns) else 0.0
    recommendations_truncated = len(recommendations) > MAX_ACTIONS
    recommendations = recommendations[:MAX_ACTIONS]
    decisions = []
    action_map = {
        "missing_values": {"fill_missing", "drop_rows_with_missing"}, "surrounding_whitespace": {"trim_whitespace"},
        "inconsistent_category": {"replace_value"}, "possible_category_typo": {"replace_value"},
        "invalid_number": {"replace_value"}, "suspect_range": {"replace_value"},
        "duplicate_rows": {"drop_duplicate_rows"},
    }
    for issue in issues:
        matching = [index for index, action in enumerate(recommendations)
                    if action.get("action") in action_map.get(issue["type"], set())
                    and action.get("column") == issue.get("column")]
        if issue["type"] == "possible_category_typo":
            examples = {str(example["value"]) for example in issue.get("examples", [])}
            matching = [index for index, action in enumerate(recommendations)
                        if action.get("action") == "replace_value" and action.get("column") == issue.get("column")
                        and str(action.get("from_value")) in examples]
        if issue["type"] == "duplicate_rows":
            matching = [index for index, action in enumerate(recommendations) if action.get("action") == "drop_duplicate_rows"]
        if matching:
            decisions_for_actions = [recommendations[index].get("decision", "recommended") for index in matching]
            decision = "recommended" if "recommended" in decisions_for_actions else decisions_for_actions[0]
            reason = recommendations[matching[0]].get("decision_reason", recommendations[matching[0]].get("reason", ""))
        elif issue["type"] in {"numeric_outliers", "constant_column"}:
            decision, reason = "nothing", "The audit found evidence, but it is not enough to justify changing the value automatically."
        else:
            decision, reason = "not_recommended", "The value is suspicious, but the intended replacement cannot be proven from this CSV alone."
        decisions.append({"issue_type": issue["type"], "column": issue.get("column"), "decision": decision,
                          "reason": reason, "action_indexes": matching})
        issue["decision"] = decision
        issue["decision_reason"] = reason

    return {
        "rows": rows, "columns": len(df.columns), "score": max(0.0, score), "scoring_version": "2.0",
        "affected_rows": int((errors | warning_only).sum()), "error_rows": int(errors.sum()), "warning_only_rows": int(warning_only.sum()),
        "dimensions": {key: {"score": round(100 * (1 - int(indices.sum()) / rows), 1) if rows else 0,
                             "affected_rows": int(indices.sum())} for key, indices in affected.items()},
        "score_explanation": "100 × (1 − (error rows + 0.25 × outlier-only rows) / total rows). Each row counts once. Constant columns are informational.",
        "checks_note": "Range and type checks use column-name and value heuristics. A clean score is not proof of semantic correctness.",
        "recommendations_truncated": recommendations_truncated,
        "decisions": decisions,
        "issues": issues, "recommendations": recommendations,
    }


def _scalar(value):
    return value is None or type(value) in (str, int, bool) or (type(value) is float and math.isfinite(value))


def _apply(df, recommendations):
    """Validate in execution order on a copy, including action dependencies."""
    if not isinstance(recommendations, list) or len(recommendations) > MAX_ACTIONS:
        raise ValueError(f"A cleaning plan must be a list of at most {MAX_ACTIONS} actions.")
    cleaned = df.copy(deep=True).astype(object)
    for item in recommendations:
        if not isinstance(item, dict):
            raise ValueError("Each action must be an object.")
        action, column = item.get("action"), item.get("column")
        if action not in {"fill_missing", "replace_value", "drop_duplicate_rows", "drop_column", "trim_whitespace", "drop_rows_with_missing"}:
            raise ValueError(f"Unsupported cleaning action: {action}")
        if action != "drop_duplicate_rows" and (not isinstance(column, str) or column not in cleaned.columns):
            raise ValueError(f"Column '{column}' is missing or was already dropped.")
        if action == "fill_missing":
            if "value" not in item or not _scalar(item["value"]) or missing_mask(pd.Series([item["value"]])).iloc[0]:
                raise ValueError("fill_missing needs a non-missing finite scalar value.")
            cleaned.loc[missing_mask(cleaned[column]), column] = item["value"]
        elif action == "replace_value":
            if not {"from_value", "to_value"} <= item.keys() or not all(_scalar(item[key]) for key in ("from_value", "to_value")):
                raise ValueError("replace_value requires finite scalar from_value and to_value.")
            if missing_mask(pd.Series([item["to_value"]])).iloc[0]:
                raise ValueError("replace_value needs a non-missing to_value.")
            # CSV cells are strings; model numbers may be JSON numbers.
            mask = cleaned[column].astype(str).eq(str(item["from_value"]))
            if item["from_value"] is None:
                mask = missing_mask(cleaned[column])
            if not mask.any():
                raise ValueError(f"Replacement source was not found in '{column}'.")
            cleaned.loc[mask, column] = item["to_value"]
        elif action == "trim_whitespace":
            cleaned[column] = cleaned[column].map(lambda value: value.strip() if isinstance(value, str) else value)
        elif action == "drop_rows_with_missing":
            cleaned = cleaned.loc[~missing_mask(cleaned[column])].reset_index(drop=True)
        elif action == "drop_column":
            if len(cleaned.columns) == 1:
                raise ValueError("Cannot drop the last column.")
            cleaned = cleaned.drop(columns=[column])
        else:
            # Compare serialized values so a filled numeric value equals the same source CSV string.
            cleaned = cleaned.loc[~cleaned.fillna("").astype(str).duplicated()].reset_index(drop=True)
    return cleaned


def validate_recommendations(df, recommendations):
    _apply(df, recommendations)


def apply_recommendations(df, recommendations, *, approved=False):
    if not approved:
        raise PermissionError("Pass approved=True to apply data-quality recommendations.")
    return _apply(df, recommendations)


def report_text(report):
    return f"Data quality score: {report['score']}/100\nRows: {report['rows']}, columns: {report['columns']}\n" + "\n".join(
        f"- {issue['type']} / {issue.get('column') or 'row-level'}: {issue['count']} rows. {issue['message']}" for issue in report["issues"])


def summary_text(report):
    if not report["issues"]:
        return (f"The dataset scored {report['score']}/100 across {report['rows']} rows and {report['columns']} columns. "
                "No measured quality issues were found. Add explicit schema rules for business-specific checks.")
    decisions = {decision: sum(item["decision"] == decision for item in report.get("decisions", []))
                 for decision in ("recommended", "nothing", "not_recommended")}
    return (f"The dataset scored {report['score']}/100 across {report['rows']} rows and {report['columns']} columns. "
            f"{report['error_rows']} rows contain measured errors and {report['warning_only_rows']} rows contain outlier-only warnings. "
            f"The audit classified {decisions['recommended']} findings as recommended, {decisions['nothing']} as nothing to change, "
            f"and {decisions['not_recommended']} as not recommended. "
            f"{len(report['recommendations'])} validated action candidates are available for review. "
            "Recommended actions can be exported; the other classifications remain unchanged.")
