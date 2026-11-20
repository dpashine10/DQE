from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .csv_io import read_csv
    from .dataquality import apply_recommendations, audit_dataframe, report_text, summary_text
except ImportError:  # Support ``python backend/main.py`` as well as ``python -m backend.main``.
    from csv_io import read_csv
    from dataquality import apply_recommendations, audit_dataframe, report_text, summary_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and conservatively clean a CSV file.")
    parser.add_argument("csv", type=Path, help="Path to the input CSV")
    parser.add_argument("--apply", action="store_true", help="Apply every validated recommendation")
    parser.add_argument("--output", type=Path, help="Output CSV path when --apply is used")
    parser.add_argument("--llm", action="store_true", help="Ask the LLM for a structured cleaning plan")
    parser.add_argument("--summary", action="store_true", help="Print a plain-language summary without needing the frontend or an LLM")
    args = parser.parse_args()

    frame = read_csv(args.csv)
    report = audit_dataframe(frame)
    print(json.dumps(report, indent=2, default=str))
    print(report_text(report))

    recommendations = report["recommendations"]
    if args.llm:
        try:
            from .LLMtools import generate_cleaning_plan
        except ImportError:
            from LLMtools import generate_cleaning_plan

        recommendations = generate_cleaning_plan(frame, report)
        print("\nLLM cleaning plan:\n" + json.dumps(recommendations, indent=2, default=str))

    if args.summary:
        print("\nSummary:\n" + summary_text(report))

    if args.apply:
        output = args.output or args.csv.with_name(f"{args.csv.stem}.cleaned.csv")
        approved_recommendations = [recommendation for recommendation in recommendations
                                    if recommendation.get("decision", "recommended") == "recommended"]
        cleaned = apply_recommendations(frame, approved_recommendations, approved=True)
        cleaned.to_csv(output, index=False)
        print(f"Cleaned CSV written to {output}")


if __name__ == "__main__":
    main()
