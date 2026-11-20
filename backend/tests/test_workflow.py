import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from backend.graph import run_job
from backend.model_context import bounded_json, MAX_CONTEXT_BYTES
from backend.dataquality import audit_dataframe
import json
import pandas as pd
from backend.notebook import deterministic_notebook, run_notebook_job


class WorkflowTests(unittest.TestCase):
    def test_model_context_is_valid_json_and_bounded_for_large_unicode_values(self):
        payload = {"columns": [{"name": "x" * 10000, "values": ["😀" * 10000] * 200}] * 200}
        result = bounded_json(payload)
        self.assertLessEqual(len(result.encode("utf-8")), MAX_CONTEXT_BYTES)
        self.assertIn("truncation_notice", json.loads(result))

    def test_fuzzy_category_suggestions_are_not_automatically_recommended(self):
        report = audit_dataframe(pd.DataFrame({"city": ["Mumbai", "Mumbai", "Mumabi"]}))
        candidates = [x for x in report["recommendations"] if x.get("from_value") in {"Mumbai", "Mumabi"}]
        self.assertTrue(candidates)
        self.assertTrue(all(x["decision"] == "not_recommended" for x in candidates))

    def test_graph_follows_upload_profile_audit_plan_summary_order(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.csv"
            input_path.write_text("id,status\n1,ok\n2,?\n", encoding="utf-8")
            with patch.dict(os.environ, {"LLM_API_KEY": "", "LLM_MODEL": "", "LLM_ENDPOINT": ""}, clear=False):
                state = run_job(uuid.uuid4().hex, input_path, "input.csv")
            self.assertTrue(Path(state["catalog_path"]).exists())
        self.assertEqual([event["step"] for event in state["events"]], ["load", "profile", "audit", "plan", "complete"])
        self.assertEqual(state["status"], "complete")
        self.assertIn("report", state)
        self.assertIn("plan", state)

    def test_deterministic_notebook_has_safe_cells_and_does_not_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "training.csv"
            input_path.write_text("id,age,price,city\n1,20,10,A\n2,30,20,B\n3,40,30,A\n", encoding="utf-8")
            with patch("backend.notebook.agent_ready", return_value=False):
                state = run_notebook_job(uuid.uuid4().hex, input_path, "training.csv", "Create an ipynb file for training a Regression model on this dataset.")
            output_path = Path(state["notebook_path"])
            self.assertTrue(output_path.exists())
            notebook = __import__("json").loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(notebook["nbformat"], 4)
            self.assertTrue(any(cell["cell_type"] == "code" for cell in notebook["cells"]))
            source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
            self.assertIn("read_csv", source)
            self.assertIn("train_test_split", source)
            self.assertIn("na_values", source)
            self.assertIn("pd.to_numeric", source)
            self.assertIn("REQUIRED PREPROCESSING", source)
            self.assertLess(source.index("REQUIRED PREPROCESSING"), source.index("train_test_split"))
            self.assertNotIn("subprocess.", source)
            self.assertIn("research_subagent", " ".join(event["step"] for event in state["events"]))
            code_sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
            preprocessing_cell = next(source for source in code_sources if "REQUIRED PREPROCESSING" in source)
            training_cell = next(source for source in code_sources if "train_baseline" in source)
            self.assertGreater(code_sources.index(training_cell), code_sources.index(preprocessing_cell))
            self.assertGreaterEqual(len(code_sources), 4)


if __name__ == "__main__":
    unittest.main()
