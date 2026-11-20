"""Very small API that connects the CSV workspace to the LangGraph pipeline."""

from __future__ import annotations

import json
import asyncio
import re
import shutil
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .csv_io import MAX_BYTES, read_csv
from .dataquality import apply_recommendations, audit_dataframe
from .graph import STORAGE, get_job, llm_ready, mark_agent_pending, retry_job, run_deferred_plan, run_job, summarize_job
from .notebook import run_notebook_job

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
app = FastAPI(title="DataQualityEng API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000", "null"],
    allow_methods=["*"], allow_headers=["*"],
)


class ApplySelection(BaseModel):
    selected_indexes: list[int] | None = None


class NotebookRequest(BaseModel):
    request: str


def job_folder(job_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise HTTPException(status_code=400, detail="Invalid job id.")
    return STORAGE / job_id


def _job_response(state):
    if not state:
        raise HTTPException(status_code=404, detail="Job not found.")
    safe = dict(state)
    safe.pop("input_path", None)
    safe.pop("catalog_path", None)
    return safe


def _notebook_result_path(job_id: str) -> Path:
    return job_folder(job_id) / "notebook-result.json"


def run_notebook_background(job_id: str, input_path: Path, file_name: str, request: str):
    result_path = _notebook_result_path(job_id)
    try:
        result = run_notebook_job(job_id, input_path, file_name, request)
        payload = {
            "job_id": job_id,
            "request": request,
            "status": "complete",
            "plan": result.get("plan", {}),
            "research": result.get("research", {}),
            "insights": result.get("insights", {}),
            "events": result.get("events", []),
            "provider_status": result.get("provider_status", {}),
            "llm_errors": result.get("llm_errors", []),
            "download_url": f"/api/jobs/{job_id}/notebook",
            "file_name": f"{Path(file_name).stem}.generated.ipynb",
        }
    except Exception as error:
        payload = {"job_id": job_id, "request": request, "status": "error", "detail": str(error)}
    result_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


@app.get("/api/health")
def health():
    return {"ok": True, "service": "DataQualityEng API"}


@app.post("/api/jobs")
async def create_job(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file.")
    job_id = uuid.uuid4().hex
    folder = STORAGE / job_id
    folder.mkdir(parents=True, exist_ok=False)
    input_path = folder / "input.csv"
    try:
        size = 0
        with input_path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(status_code=413, detail="CSV files must be 512 MiB or smaller.")
                output.write(chunk)
        state = await asyncio.to_thread(run_job, job_id, input_path, Path(file.filename).name, skip_llm=True)
        defer_agent = llm_ready()
        if defer_agent:
            mark_agent_pending(job_id)
            background_tasks.add_task(run_deferred_plan, job_id)
            state["status"] = "planning"
            state["agent_pending"] = True
        state["job_id"] = job_id
        return _job_response(state)
    except HTTPException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    except ValueError as error:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"The audit could not finish: {error}") from error


@app.post("/api/jobs/sample")
async def sample_job(background_tasks: BackgroundTasks):
    source = STORAGE.parent.parent / "sales_mock.csv"
    if not source.exists():
        raise HTTPException(status_code=404, detail="Sample CSV is missing.")
    job_id = uuid.uuid4().hex
    folder = STORAGE / job_id
    folder.mkdir(parents=True, exist_ok=False)
    input_path = folder / "input.csv"
    input_path.write_bytes(source.read_bytes())
    state = await asyncio.to_thread(run_job, job_id, input_path, source.name, skip_llm=True)
    if llm_ready():
        mark_agent_pending(job_id)
        background_tasks.add_task(run_deferred_plan, job_id)
        state["status"] = "planning"
        state["agent_pending"] = True
    return _job_response({**state, "job_id": job_id})


@app.get("/api/jobs/{job_id}")
def read_job(job_id: str):
    job_folder(job_id)
    return _job_response(get_job(job_id))


@app.post("/api/jobs/{job_id}/retry")
def retry(job_id: str):
    job_folder(job_id)
    try:
        return _job_response({**retry_job(job_id), "job_id": job_id})
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Retry failed: {error}") from error


def _selected_actions(state, selection):
    plan = state.get("plan", [])
    if selection is None or selection.selected_indexes is None:
        return [action for action in plan if action.get("decision", "recommended") == "recommended"]
    indexes = selection.selected_indexes
    if len(indexes) != len(set(indexes)) or any(index < 0 or index >= len(plan) for index in indexes):
        raise ValueError("Selected action indexes must be unique and belong to this plan.")
    selected = [plan[index] for index in indexes]
    if any(action.get("decision", "recommended") != "recommended" for action in selected):
        raise ValueError("Only actions classified as recommended can be applied.")
    return selected


@app.post("/api/jobs/{job_id}/preview")
def preview_cleaning(job_id: str, selection: ApplySelection | None = Body(default=None)):
    folder = job_folder(job_id)
    state = _job_response(get_job(job_id))
    try:
        frame = read_csv(folder / "input.csv")
        selected = _selected_actions(state, selection)
        cleaned = apply_recommendations(frame, selected, approved=True)
        after = audit_dataframe(cleaned)
        before = state["report"]
        return {"actions": len(selected), "before": {"rows": before["rows"], "columns": before["columns"], "score": before["score"], "error_rows": before["error_rows"]},
                "after": {"rows": after["rows"], "columns": after["columns"], "score": after["score"], "error_rows": after["error_rows"]}}
    except (ValueError, PermissionError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/jobs/{job_id}/apply")
def apply_job(job_id: str, selection: ApplySelection | None = Body(default=None)):
    folder = job_folder(job_id)
    state = _job_response(get_job(job_id))
    input_path = folder / "input.csv"
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="Original upload is no longer available.")
    try:
        frame = read_csv(input_path)
        selected = _selected_actions(state, selection)
        cleaned = apply_recommendations(frame, selected, approved=True)
        output_path = folder / "cleaned.csv"
        cleaned.to_csv(output_path, index=False)
        return FileResponse(output_path, filename=f"{Path(state.get('file_name', 'data.csv')).stem}.cleaned.csv", media_type="text/csv")
    except (ValueError, PermissionError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/jobs/{job_id}/report")
def download_report(job_id: str):
    folder = job_folder(job_id)
    state = _job_response(get_job(job_id))
    output_path = folder / "audit-report.json"
    output_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    return FileResponse(output_path, filename=f"{Path(state.get('file_name', 'data.csv')).stem}.audit.json", media_type="application/json")


@app.post("/api/jobs/{job_id}/summary")
def agent_summary(job_id: str):
    job_folder(job_id)
    try:
        return summarize_job(job_id)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"The summary could not be generated: {error}") from error


@app.post("/api/jobs/{job_id}/notebook")
def create_notebook(job_id: str, body: NotebookRequest, background_tasks: BackgroundTasks):
    folder = job_folder(job_id)
    state = _job_response(get_job(job_id))
    input_path = folder / "input.csv"
    request = body.request.strip()
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="Original upload is no longer available.")
    if not request:
        raise HTTPException(status_code=400, detail="Describe the notebook you want to generate.")
    if len(request) > 1_000:
        raise HTTPException(status_code=400, detail="Notebook requests must be 1,000 characters or fewer.")
    _notebook_result_path(job_id).unlink(missing_ok=True)
    background_tasks.add_task(run_notebook_background, job_id, input_path, state.get("file_name", "data.csv"), request)
    return {"job_id": job_id, "request": request, "status": "generating", "download_url": f"/api/jobs/{job_id}/notebook"}


@app.get("/api/jobs/{job_id}/notebook/status")
def notebook_status(job_id: str):
    if not get_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found.")
    result_path = _notebook_result_path(job_id)
    if not result_path.exists():
        return {"job_id": job_id, "status": "generating"}
    return json.loads(result_path.read_text(encoding="utf-8"))


@app.get("/api/jobs/{job_id}/notebook")
def download_notebook(job_id: str):
    folder = job_folder(job_id)
    state = _job_response(get_job(job_id))
    output_path = folder / "generated_notebook.ipynb"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Generate a notebook before downloading it.")
    return FileResponse(output_path, filename=f"{Path(state.get('file_name', 'data')).stem}.generated.ipynb", media_type="application/x-ipynb+json")


FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
