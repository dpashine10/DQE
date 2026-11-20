import unittest
from unittest.mock import patch

import pandas as pd

from backend.notebook import MAX_CATALOG_AGENT_BATCHES, _agent_batches, _catalog_batches
from backend.notebook_cells import preprocess_dataframe, train_baseline
from backend.providers import provider_summary


class NotebookProviderTests(unittest.TestCase):
    def test_role_configuration_falls_back_to_base_values(self):
        with patch.dict("os.environ", {
            "LLM_API_KEY": "base-key",
            "LLM_ENDPOINT": "https://example.test/v1/chat/completions",
            "LLM_MODEL": "base-model",
            "AGENT_CODE_MODEL": "code-model",
        }, clear=True):
            summary = provider_summary()
        self.assertEqual(summary["research"]["model"], "base-model")
        self.assertEqual(summary["code"]["model"], "code-model")
        self.assertTrue(summary["primary"]["configured"])

    def test_value_catalog_is_split_into_small_batches(self):
        frame = pd.DataFrame({"category": [f"value-{index}" for index in range(245)], "id": [str(index) for index in range(245)]})
        batches = _catalog_batches(frame, values_per_column=50, columns_per_batch=1)
        self.assertEqual(len(batches), 5)
        self.assertLessEqual(max(len(item["values"]) for batch in batches for item in batch["columns"]), 50)
        self.assertNotIn("id", {item["name"] for batch in batches for item in batch["columns"]})
        values = {value for batch in batches for item in batch["columns"] for value in item["values"]}
        self.assertEqual(values, {f"value-{index}" for index in range(245)})

    def test_model_catalog_calls_are_bounded_and_cover_the_range(self):
        batches = [{"batch": index, "columns": []} for index in range(100)]
        selected = _agent_batches(batches)
        self.assertEqual(len(selected), MAX_CATALOG_AGENT_BATCHES)
        self.assertEqual(selected[0]["batch"], 0)
        self.assertEqual(selected[-1]["batch"], 99)
        self.assertEqual(len({item["batch"] for item in selected}), MAX_CATALOG_AGENT_BATCHES)

    def test_preprocessing_removes_placeholder_strings_before_training(self):
        import pandas as pd

        raw = pd.DataFrame({"age": ["20", "?", "30", "40"], "city": ["A", "N/A", "B", "A"], "price": ["10", "20", "?", "40"]})
        clean, report = preprocess_dataframe(raw, {"age": "numeric", "city": "text", "price": "numeric"}, ["?", "N/A"], {"age": {"min": 0, "max": 120, "integer": True}, "price": {"min": 0}})
        self.assertTrue(pd.isna(clean.loc[1, "age"]))
        self.assertTrue(clean["price"].isna().sum() == 1)
        result = train_baseline(clean, "price", ["age", "city"], {"age": "numeric", "city": "text", "price": "numeric"})
        self.assertIn(result["status"], {"trained", "skipped"})


if __name__ == "__main__":
    unittest.main()
