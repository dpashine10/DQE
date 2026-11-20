import shutil
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.api import app
from backend.graph import STORAGE


class ApiFlowTests(unittest.TestCase):
    def test_health_upload_report_apply_and_notebook_flow(self):
        client = TestClient(app)
        with patch("backend.api.llm_ready", return_value=False), patch("backend.graph.llm_ready", return_value=False), patch("backend.notebook.agent_ready", return_value=False):
            health = client.get("/api/health")
            self.assertEqual(health.status_code, 200)

            response = client.post(
                "/api/jobs",
                files={"file": ("api-test.csv", b"id,value\n1,10\n2,?\n", "text/csv")},
            )
            self.assertEqual(response.status_code, 200)
            job_id = response.json()["job_id"]
            try:
                self.assertEqual(client.get(f"/api/jobs/{job_id}").status_code, 200)
                self.assertEqual(client.get(f"/api/jobs/{job_id}/report").status_code, 200)
                self.assertEqual(client.post(f"/api/jobs/{job_id}/summary").status_code, 200)
                impact = client.post(f"/api/jobs/{job_id}/preview", json={"selected_indexes": []})
                self.assertEqual(impact.status_code, 200)
                self.assertEqual(impact.json()["before"], impact.json()["after"])
                self.assertFalse((STORAGE / job_id / "cleaned.csv").exists())
                self.assertEqual(client.post(f"/api/jobs/{job_id}/preview", json={"selected_indexes": [999]}).status_code, 400)
                applied = client.post(f"/api/jobs/{job_id}/apply", json={"selected_indexes": []})
                self.assertEqual(applied.status_code, 200)
                self.assertEqual(applied.headers["content-type"].split(";")[0], "text/csv")
                queued = client.post(f"/api/jobs/{job_id}/notebook", json={"request": "Create an exploratory notebook."})
                self.assertEqual(queued.status_code, 200)
                status = client.get(f"/api/jobs/{job_id}/notebook/status")
                self.assertEqual(status.status_code, 200)
                self.assertEqual(status.json()["status"], "complete")
                self.assertEqual(client.get(f"/api/jobs/{job_id}/notebook").status_code, 200)
            finally:
                shutil.rmtree(STORAGE / job_id, ignore_errors=True)

            sample = client.post("/api/jobs/sample")
            self.assertEqual(sample.status_code, 200)
            sample_job_id = sample.json()["job_id"]
            try:
                retried = client.post(f"/api/jobs/{sample_job_id}/retry")
                self.assertEqual(retried.status_code, 200)
                self.assertEqual(retried.json()["status"], "complete")
            finally:
                shutil.rmtree(STORAGE / sample_job_id, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
