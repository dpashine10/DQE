import unittest

import pandas as pd

from backend.dataquality import apply_recommendations, audit_dataframe


class DataQualityTests(unittest.TestCase):
    def test_bad_values_are_not_hidden_by_a_high_score(self):
        frame = pd.DataFrame({
            "id": ["1", "2", "2", "4"],
            "age": ["25", "?", "five", "200"],
            "price": ["10", "-3", "abc", "999"],
            "city": [" Mumbai ", "mumbai", "Pune", ""],
        })
        report = audit_dataframe(frame)
        types = {issue["type"] for issue in report["issues"]}
        self.assertLess(report["score"], 70)
        self.assertTrue({"missing_values", "invalid_number", "suspect_range", "duplicate_identifier"} <= types)
        self.assertTrue(any(issue["type"] == "inconsistent_category" for issue in report["issues"]))
        self.assertTrue(any(action["action"] == "replace_value" and action.get("column") == "price" for action in report["recommendations"]))

    def test_missing_placeholders_and_copy_safe_apply(self):
        frame = pd.DataFrame({"value": ["1", "?", "3"], "name": [" A ", "A", "B"]})
        report = audit_dataframe(frame)
        cleaned = apply_recommendations(frame, report["recommendations"], approved=True)
        self.assertEqual(frame.loc[1, "value"], "?")
        self.assertNotIn("?", cleaned["value"].tolist())
        self.assertEqual(cleaned.loc[0, "name"], "A")

    def test_empty_and_constant_inputs_have_bounded_scores(self):
        empty = pd.DataFrame(columns=["id", "value"])
        self.assertEqual(audit_dataframe(empty)["score"], 0.0)
        constant = pd.DataFrame({"id": ["1", "2"], "status": ["ok", "ok"]})
        report = audit_dataframe(constant)
        self.assertEqual(report["score"], 100.0)
        self.assertTrue(any(issue["type"] == "constant_column" for issue in report["issues"]))

    def test_rows_with_missing_identifiers_can_be_dropped_explicitly(self):
        frame = pd.DataFrame({"id": ["1", "?", "3"], "value": ["a", "b", "c"]})
        cleaned = apply_recommendations(frame, [{"action": "drop_rows_with_missing", "column": "id"}], approved=True)
        self.assertEqual(cleaned["id"].tolist(), ["1", "3"])

    def test_score_weights_outlier_only_rows_at_one_quarter(self):
        report = audit_dataframe(pd.DataFrame({"value": ["1", "2", "3", "100"]}))
        self.assertEqual(report["error_rows"], 0)
        self.assertEqual(report["warning_only_rows"], 1)
        self.assertEqual(report["score"], 93.8)

    def test_replacements_cannot_write_missing_values(self):
        frame = pd.DataFrame({"value": ["1", "2"]})
        with self.assertRaisesRegex(ValueError, "non-missing to_value"):
            apply_recommendations(frame, [{"action": "replace_value", "column": "value", "from_value": "1", "to_value": "?"}], approved=True)


if __name__ == "__main__":
    unittest.main()
