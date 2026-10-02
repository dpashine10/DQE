"""Deterministic data-quality auditing and conservative cleaning operations."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def audit_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Return a serializable audit report without changing ``df``."""
    rows = len(df)
    duplicate_rows = int(df.duplicated().sum())
    issues: list[dict[str, Any]] = []
    recommendations: list[dict[str, Any]] = []
    total_penalty = 0.0

    for column in df.columns:
        missing = int(df[column].isna().sum())
        if missing:
            rate = missing / max(rows, 1)
            total_penalty += min(rate * 30, 30)
            series = df[column].dropna()
            if pd.api.types.is_numeric_dtype(series):
                strategy = "median"
                value = _json_value(series.median()) if not series.empty else None
            else:
                modes = series.mode(dropna=True)
                strategy = "mode"
                value = _json_value(modes.iloc[0]) if not modes.empty else None
            issues.append({"type": "missing_values", "column": column, "count": missing, "rate": round(rate, 4)})
            if value is not None:
                recommendations.append({"action": "fill_missing", "column": column, "strategy": strategy, "value": value, "affected_rows": missing})

        if df[column].nunique(dropna=False) <= 1 and rows:
            issues.append({"type": "constant_column", "column": column})
            total_penalty += 5

        if pd.api.types.is_numeric_dtype(df[column]):
            values = df[column].dropna()
            if len(values) >= 4:
                first_quartile, third_quartile = values.quantile([0.25, 0.75])
                spread = third_quartile - first_quartile
                if spread > 0:
                    outliers = int(((values < first_quartile - 1.5 * spread) | (values > third_quartile + 1.5 * spread)).sum())
                    if outliers:
                        issues.append({"type": "numeric_outliers", "column": column, "count": outliers})
                        total_penalty += min(outliers / max(rows, 1) * 20, 20)

    if duplicate_rows:
        duplicate_rate = duplicate_rows / max(rows, 1)
        issues.append({"type": "duplicate_rows", "count": duplicate_rows, "rate": round(duplicate_rate, 4)})
        recommendations.append({"action": "drop_duplicate_rows", "affected_rows": duplicate_rows})
        total_penalty += min(duplicate_rate * 30, 30)

    return {
        "rows": rows,
        "columns": len(df.columns),
        "score": max(0.0, round(100.0 - total_penalty, 1)),
        "issues": issues,
        "recommendations": recommendations,
    }


def apply_recommendations(df: pd.DataFrame, recommendations: list[dict[str, Any]], *, approved: bool = False) -> pd.DataFrame:
    """Apply validated recommendations after explicit approval."""
    if not approved:
        raise PermissionError("Pass approved=True to apply data-quality recommendations.")

    validate_recommendations(df, recommendations)
    cleaned = df.copy()
    for recommendation in recommendations:
        action = recommendation.get("action")
        if action == "fill_missing":
            column = recommendation["column"]
            if column not in cleaned.columns:
                raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
            cleaned[column] = cleaned[column].fillna(recommendation["value"])
        elif action == "drop_duplicate_rows":
            cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        elif action == "replace_value":
            column = recommendation["column"]
            cleaned[column] = cleaned[column].replace(recommendation["from_value"], recommendation["to_value"])
        elif action == "drop_column":
            cleaned = cleaned.drop(columns=[recommendation["column"]])
        else:
            raise ValueError(f"Unsupported recommendation action: {action}")
    return cleaned


def validate_recommendations(df: pd.DataFrame, recommendations: list[dict[str, Any]]) -> None:
    """Reject malformed or out-of-scope model actions before any mutation."""
    supported_actions = {"fill_missing", "replace_value", "drop_duplicate_rows", "drop_column"}
    for recommendation in recommendations:
        action = recommendation.get("action")
        if action not in supported_actions:
            raise ValueError(f"Unsupported recommendation action: {action}")
        if action in {"fill_missing", "replace_value", "drop_column"}:
            column = recommendation.get("column")
            if not isinstance(column, str) or column not in df.columns:
                raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
        if action == "fill_missing" and "value" not in recommendation:
            raise ValueError("fill_missing requires a value.")
        if action == "replace_value" and not {"from_value", "to_value"}.issubset(recommendation):
            raise ValueError("replace_value requires from_value and to_value.")


def report_text(report: dict[str, Any]) -> str:
    """Create compact, model-friendly text from an audit report."""
    lines = [f"Data quality score: {report['score']}/100", f"Rows: {report['rows']}, columns: {report['columns']}"]
    lines.extend(f"- {issue['type']}: {issue}" for issue in report["issues"])
    return "\n".join(lines)