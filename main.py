from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dataquality import apply_recommendations, audit_dataframe, report_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and conservatively clean a CSV file.")
    parser.add_argument("csv", type=Path, help="Path to the input CSV")
    parser.add_argument("--apply", action="store_true", help="Apply every validated recommendation")
    parser.add_argument("--output", type=Path, help="Output CSV path when --apply is used")
    parser.add_argument("--llm", action="store_true", help="Ask the LLM for a structured cleaning plan")
    parser.add_argument("--summary", action="store_true", help="Ask the configured LLM for a plain-language summary")
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    report = audit_dataframe(frame)
    print(json.dumps(report, indent=2, default=str))
    print(report_text(report))

    recommendations = report["recommendations"]
    if args.llm:
        from LLMtools import generate_cleaning_plan

        recommendations = generate_cleaning_plan(frame, report)
        print("\nLLM cleaning plan:\n" + json.dumps(recommendations, indent=2, default=str))

    if args.summary:
        from LLMtools import generatesummary

        print("\nLLM summary:\n" + generatesummary(report_text(report)))

    if args.apply:
        output = args.output or args.csv.with_name(f"{args.csv.stem}.cleaned.csv")
        cleaned = apply_recommendations(frame, recommendations, approved=True)
        cleaned.to_csv(output, index=False)
        print(f"Cleaned CSV written to {output}")


if __name__ == "__main__":
    main()
