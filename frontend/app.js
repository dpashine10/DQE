const API_BASE = window.location.protocol === "file:" ? "http://127.0.0.1:8000" : window.location.origin;
const $ = (selector) => document.querySelector(selector);
const fileInput = $("#file-input");
const dropZone = $("#drop-zone");
const fileStatus = $("#file-status");
const downloadReport = $("#download-report");
const downloadCleaned = $("#download-cleaned");
const previewCleaning = $("#preview-cleaning");
const summaryButton = $("#summary-button");
const generateNotebook = $("#generate-notebook");
const notebookRequest = $("#notebook-request");
let issueFilter = "all";
let currentJob = null;
let agentPollTimer = null;
let notebookPollTimer = null;

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  }[character]));
}

function setStatus(message, ready = false) {
  fileStatus.classList.toggle("is-ready", ready);
  fileStatus.innerHTML = `<span class="status-dot" aria-hidden="true"></span><span>${escapeHtml(message)}</span>`;
}

async function responseJson(response) {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || "The backend could not finish the audit.");
  return body;
}

function renderActivity(events = []) {
  const finished = new Set(events.filter((event) => ["done", "fallback"].includes(event.status)).map((event) => event.step));
  const current = events.length ? events[events.length - 1].step : "load";
  document.querySelectorAll(".activity-item").forEach((item) => {
    const step = item.dataset.step;
    item.classList.toggle("is-done", finished.has(step) || step === "complete" && finished.has("complete"));
    item.classList.toggle("is-current", step === current && !finished.has(step));
  });
}

function scoreCopy(score) {
  if (score >= 90) return "Strong measured baseline";
  if (score >= 70) return "Review recommended";
  if (score >= 40) return "Needs attention";
  return "High-risk dataset";
}

function clearResults() {
  if (agentPollTimer) {
    clearTimeout(agentPollTimer);
    agentPollTimer = null;
  }
  if (notebookPollTimer) {
    clearTimeout(notebookPollTimer);
    notebookPollTimer = null;
  }
  currentJob = null;
  issueFilter = "all";
  document.querySelectorAll(".filter-button").forEach((button) => button.classList.toggle("is-active", button.dataset.filter === "all"));
  $("#score-value").textContent = "—";
  $("#score-note").textContent = "Waiting for a file";
  $("#row-count").textContent = "—";
  $("#column-count").textContent = "—";
  $("#issue-count").textContent = "—";
  $("#health-grid").innerHTML = ["completeness", "validity", "consistency", "uniqueness"].map((label) => `<div class="health-card"><span>${label}</span><strong>—</strong><small>Awaiting audit</small></div>`).join("");
  $("#issue-summary").textContent = "No audit yet";
  $("#agent-summary").textContent = "Upload a CSV. The backend will profile it, audit it, and build a validated plan.";
  $("#issues-list").innerHTML = `<div class="empty-state"><span class="empty-line"></span><strong>Your audit signals will appear here.</strong><span>Select a CSV above to inspect missing values, duplicates, and outliers.</span></div>`;
  $("#plan-summary").textContent = "No plan yet";
  $("#plan-list").innerHTML = `<span class="plan-empty">Actions proposed after the audit will appear here.</span>`;
  $("#table-file-name").textContent = "No CSV loaded";
  $("#preview-note").textContent = "Preview is limited to five rows.";
  $("#preview-head").innerHTML = "<tr><th>Upload a file to preview columns</th></tr>";
  $("#preview-body").innerHTML = "<tr><td class='table-empty'>Your first five records will show up here.</td></tr>";
  downloadReport.disabled = true;
  downloadCleaned.disabled = true;
  summaryButton.disabled = true;
  $("#summary-source").textContent = "Deterministic summary appears here after an audit.";
  $("#select-recommended").disabled = true;
  $("#clear-selection").disabled = true;
  generateNotebook.disabled = true;
  $("#notebook-result").innerHTML = `<div class="notebook-empty"><span class="notebook-icon" aria-hidden="true">ipynb</span><strong>No notebook yet.</strong><span>Upload a CSV, describe the plot or model, and download a ready-to-review Jupyter notebook.</span></div>`;
}

function renderHealth(dimensions = {}) {
  const labels = { completeness: "Completeness", validity: "Validity", consistency: "Consistency", uniqueness: "Uniqueness" };
  $("#health-grid").innerHTML = Object.entries(labels).map(([key, label]) => {
    const dimension = dimensions[key] || {};
    const score = dimension.score ?? "—";
    return `<div class="health-card"><span>${label}</span><strong>${score}${score === "—" ? "" : "<small>/100</small>"}</strong><div class="health-bar"><i style="width:${score === "—" ? 0 : score}%"></i></div><small>${dimension.affected_rows ?? 0} affected rows</small></div>`;
  }).join("");
}

function renderIssues(issues) {
  const visible = issueFilter === "all" ? issues : issues.filter((issue) => (issue.decision || "not_recommended") === issueFilter);
  $("#issues-list").innerHTML = visible.length ? visible.map((issue) => `
    <div class="issue-row" data-decision="${escapeHtml(issue.decision || "not_recommended")}">
      <span class="issue-check ${issue.severity === "warning" ? "is-warning" : "is-selected"}" aria-hidden="true"></span>
      <span><strong class="issue-title">${escapeHtml(issue.type.replaceAll("_", " "))}${issue.column ? ` · ${escapeHtml(issue.column)}` : ""}</strong><span class="issue-detail">${escapeHtml(issue.message)} ${issue.count} affected row${issue.count === 1 ? "" : "s"}.${issue.examples?.length ? ` Example: row ${escapeHtml(issue.examples[0].row)} (${escapeHtml(issue.examples[0].value)}).` : ""}</span></span>
      <span class="issue-tag">${escapeHtml(issue.decision || issue.severity || "review")}</span>
    </div>`).join("") : `<div class="empty-state"><span class="empty-line"></span><strong>No findings in this view.</strong><span>Choose another filter to see the rest of the audit evidence.</span></div>`;
}

function renderJob(job) {
  currentJob = job;
  const report = job.report || {};
  const issues = report.issues || [];
  $("#score-value").textContent = `${report.score ?? "—"}`;
  $("#score-note").textContent = scoreCopy(Number(report.score || 0));
  $("#row-count").textContent = Number(report.rows || 0).toLocaleString();
  $("#column-count").textContent = Number(report.columns || 0).toLocaleString();
  $("#issue-count").textContent = issues.length.toLocaleString();
  $("#issue-summary").textContent = issues.length ? `${issues.length} signal${issues.length === 1 ? "" : "s"}` : "No signals";
  $("#agent-summary").textContent = job.summary || "The audit is ready for review.";
  $("#summary-source").textContent = "Deterministic graph summary · ask the agent for a deeper read when configured.";
  summaryButton.disabled = false;
  renderHealth(report.dimensions || {});
  const plan = job.plan || [];
  const decisions = report.decisions || [];
  const actionRows = plan.map((action, index) => {
    const decision = action.decision || "recommended";
    const selectable = decision === "recommended";
    return `
    <label class="plan-item">
      <input class="plan-check" type="checkbox" data-plan-index="${index}" ${selectable ? "checked" : "disabled"} />
      <span><strong>${escapeHtml(action.action.replaceAll("_", " "))}${action.column ? ` · ${escapeHtml(action.column)}` : ""} <em class="decision-badge decision-${decision}">${escapeHtml(decision)}</em></strong><small>${escapeHtml(action.reason || "Validated against the uploaded data.")} · confidence ${Math.round(Number(action.confidence ?? 1) * 100)}%</small></span>
    </label>`;
  }).join("");
  const decisionRows = decisions.filter((decision) => !decision.action_indexes?.length).map((decision) => `
    <div class="plan-decision"><strong><em class="decision-badge decision-${escapeHtml(decision.decision)}">${escapeHtml(decision.decision)}</em> ${escapeHtml((decision.issue_type || "finding").replaceAll("_", " "))}${decision.column ? ` · ${escapeHtml(decision.column)}` : ""}</strong><small>${escapeHtml(decision.reason)}</small></div>`).join("");
  $("#plan-summary").textContent = plan.length ? `${plan.length} reviewed action${plan.length === 1 ? "" : "s"}` : "No action needed";
  $("#plan-list").innerHTML = actionRows || decisionRows ? `${actionRows}${decisionRows}` : `<span class="plan-empty">No validated cleaning action is waiting for approval.</span>`;
  updatePlanSelection();
  $("#table-file-name").textContent = job.file_name || "CSV loaded";
  $("#preview-note").textContent = `Showing ${(job.preview?.rows || []).length} of ${report.rows || 0} rows`;
  downloadReport.disabled = false;
  previewCleaning.disabled = false;
  downloadCleaned.disabled = !plan.some((action) => (action.decision || "recommended") === "recommended");
  $("#select-recommended").disabled = !plan.some((action) => (action.decision || "recommended") === "recommended");
  $("#clear-selection").disabled = !plan.length;
  generateNotebook.disabled = false;

  renderIssues(issues);

  const columns = job.preview?.columns || [];
  $("#preview-head").innerHTML = `<tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr>`;
  $("#preview-body").innerHTML = (job.preview?.rows || []).map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(cell) || "<span class='empty-cell'>—</span>"}</td>`).join("")}</tr>`).join("");
  renderActivity(job.events || []);

  if (job.agent_pending || job.status === "planning") {
    setStatus(`${job.file_name} is audited. The agent is still planning in the background…`, true);
  } else if (job.llm_error) {
    setStatus(`${job.file_name} audited. The deterministic plan is ready; the model was unavailable.`, true);
  } else if (job.llm_used) {
    setStatus(`${job.file_name} audited and the LangChain plan was validated.`, true);
  } else {
    setStatus(`${job.file_name} audited with the deterministic plan.`, true);
  }
}

function pollAgentPlan(jobId) {
  if (agentPollTimer) clearTimeout(agentPollTimer);
  agentPollTimer = setTimeout(async () => {
    if (!currentJob || currentJob.job_id !== jobId) return;
    try {
      const job = await responseJson(await fetch(`${API_BASE}/api/jobs/${jobId}`));
      renderJob(job);
      if (job.agent_pending || job.status === "planning") pollAgentPlan(jobId);
      else agentPollTimer = null;
    } catch (error) {
      // Keep the deterministic result visible if a status poll is interrupted.
      agentPollTimer = null;
    }
  }, 900);
}

async function uploadFile(file) {
  if (!file || !file.name.toLowerCase().endsWith(".csv")) {
    setStatus("Please choose a CSV file to start the audit.");
    return;
  }
  clearResults();
  setStatus(`Uploading ${file.name}. The graph is running…`);
  downloadReport.disabled = true;
  downloadCleaned.disabled = true;
  const form = new FormData();
  form.append("file", file);
  try {
    renderActivity([{ step: "load", status: "pending" }]);
    const job = await responseJson(await fetch(`${API_BASE}/api/jobs`, { method: "POST", body: form }));
    renderJob(job);
    if (job.agent_pending || job.status === "planning") pollAgentPlan(job.job_id);
  } catch (error) {
    setStatus(error.message);
    renderActivity([]);
  }
}

async function loadSample() {
  clearResults();
  setStatus("Loading the sample through the backend graph…");
  try {
    const job = await responseJson(await fetch(`${API_BASE}/api/jobs/sample`, { method: "POST" }));
    renderJob(job);
    if (job.agent_pending || job.status === "planning") pollAgentPlan(job.job_id);
  } catch (error) {
    setStatus(error.message);
  }
}

async function askAgentForSummary() {
  if (!currentJob) return;
  summaryButton.disabled = true;
  summaryButton.textContent = "Thinking…";
  try {
    const result = await responseJson(await fetch(`${API_BASE}/api/jobs/${currentJob.job_id}/summary`, { method: "POST" }));
    $("#agent-summary").textContent = result.summary;
    $("#summary-source").textContent = result.source === "langchain" ? "LangChain agent summary · grounded in the audit and proposed actions." : "Deterministic summary · model unavailable or not configured.";
  } catch (error) {
    setStatus(error.message);
  } finally {
    summaryButton.disabled = false;
    summaryButton.textContent = "Ask agent for summary";
  }
}

async function createNotebook() {
  if (!currentJob) return;
  const request = notebookRequest.value.trim();
  if (!request) {
    notebookRequest.focus();
    setStatus("Describe the notebook you want to generate first.");
    return;
  }
  generateNotebook.disabled = true;
  generateNotebook.textContent = "Agents are planning…";
  $("#notebook-result").innerHTML = `<div class="notebook-loading"><span class="status-dot" aria-hidden="true"></span><strong>Researching the task, reading the columns, and writing notebook cells…</strong><small>The code will be validated but not executed.</small></div>`;
  try {
    const queued = await responseJson(await fetch(`${API_BASE}/api/jobs/${currentJob.job_id}/notebook`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request }),
    }));
    const result = queued.status === "generating" ? await waitForNotebook(currentJob.job_id) : queued;
    renderNotebookResult(result);
    setStatus("Notebook generated and validated. Nothing was executed.", true);
  } catch (error) {
    $("#notebook-result").innerHTML = `<div class="notebook-error"><strong>Notebook generation failed.</strong><span>${escapeHtml(error.message)}</span></div>`;
    setStatus(error.message);
  } finally {
    generateNotebook.disabled = false;
    generateNotebook.textContent = "Generate notebook";
  }
}

function renderNotebookResult(result) {
    const plan = result.plan || {};
    const fallbacks = result.llm_errors?.length ? `<small class="notebook-fallback">Some roles used deterministic fallback: ${escapeHtml(result.llm_errors.join(" "))}</small>` : "";
    const target = plan.target_column || result.insights?.target_column || "exploration only";
    const features = (plan.feature_columns || result.insights?.feature_columns || []).slice(0, 6).join(", ") || "selected from the profile";
    const agents = (result.events || []).filter((event) => event.step.includes("subagent") || event.step === "primary_planner").map((event) => event.step.replaceAll("_", " ")).join(" · ");
    $("#notebook-result").innerHTML = `
      <div class="notebook-success">
        <div class="notebook-result-top"><span class="file-badge">IPYNB</span><span>ready to review</span></div>
        <h3>${escapeHtml(plan.title || "Generated data notebook")}</h3>
        <p>${escapeHtml(plan.rationale || "The agents selected a notebook structure from your request and the dataset profile.")}</p>
        <div class="notebook-insight"><strong>Agent read</strong><span>Target: ${escapeHtml(target)}</span><span>Features: ${escapeHtml(features)}</span><small>${escapeHtml(agents || "research · insights · planner · writer")}</small></div>
        <div class="notebook-meta"><span>${escapeHtml(plan.task || "exploration")}</span><span>${(result.events || []).length} workflow steps</span><span>${escapeHtml(result.provider_status?.research?.model || "deterministic fallback")}</span></div>
        <a class="button button-light notebook-download" href="${API_BASE}${escapeHtml(result.download_url)}" download>Download .ipynb</a>
        ${fallbacks}
      </div>`;
}

function waitForNotebook(jobId) {
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const result = await responseJson(await fetch(`${API_BASE}/api/jobs/${jobId}/notebook/status`));
        if (result.status === "complete") {
          notebookPollTimer = null;
          resolve(result);
          return;
        }
        if (result.status === "error") {
          notebookPollTimer = null;
          reject(new Error(result.detail || "The notebook worker failed."));
          return;
        }
        notebookPollTimer = setTimeout(poll, 900);
      } catch (error) {
        notebookPollTimer = null;
        reject(error);
      }
    };
    poll();
  });
}

async function downloadBlob(url, method, fallbackName) {
  const response = await fetch(url, { method });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || "The file could not be downloaded.");
  }
  const blob = await response.blob();
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = fallbackName;
  link.click();
  URL.revokeObjectURL(link.href);
}

function selectedPlanIndexes() {
  return [...document.querySelectorAll(".plan-check:checked")].map((input) => Number(input.dataset.planIndex));
}

function updatePlanSelection() {
  const total = document.querySelectorAll(".plan-check").length;
  const selected = selectedPlanIndexes().length;
  $("#plan-summary").textContent = total ? `${selected}/${total} selected` : "No action needed";
}

$("#choose-file").addEventListener("click", () => {
  // Clearing before opening also makes selecting the same CSV twice fire `change`.
  fileInput.value = "";
  fileInput.click();
});
fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  event.target.value = "";
  uploadFile(file);
});
$("#sample-file").addEventListener("click", loadSample);
$("#plan-list").addEventListener("change", updatePlanSelection);
summaryButton.addEventListener("click", askAgentForSummary);
generateNotebook.addEventListener("click", createNotebook);
document.querySelectorAll(".prompt-chip").forEach((chip) => chip.addEventListener("click", () => {
  notebookRequest.value = chip.dataset.prompt;
  notebookRequest.focus();
}));
$("#select-recommended").addEventListener("click", () => {
  document.querySelectorAll(".plan-check:not(:disabled)").forEach((input) => { input.checked = true; });
  updatePlanSelection();
});
$("#clear-selection").addEventListener("click", () => {
  document.querySelectorAll(".plan-check").forEach((input) => { input.checked = false; });
  updatePlanSelection();
});
document.querySelectorAll(".filter-button").forEach((button) => button.addEventListener("click", () => {
  issueFilter = button.dataset.filter;
  document.querySelectorAll(".filter-button").forEach((item) => item.classList.toggle("is-active", item === button));
  renderIssues(currentJob?.report?.issues || []);
}));
downloadReport.addEventListener("click", async () => {
  if (!currentJob) return;
  try { await downloadBlob(`${API_BASE}/api/jobs/${currentJob.job_id}/report`, "GET", "audit-report.json"); }
  catch (error) { setStatus(error.message); }
});
previewCleaning.addEventListener("click", async () => {
  if (!currentJob) return;
  previewCleaning.disabled = true;
  setStatus("Calculating the effect of the selected actions…");
  try {
    const result = await responseJson(await fetch(`${API_BASE}/api/jobs/${currentJob.job_id}/preview`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ selected_indexes: selectedPlanIndexes() }),
    }));
    setStatus(`${result.actions} actions: score ${result.before.score} → ${result.after.score}; rows ${result.before.rows} → ${result.after.rows}; error rows ${result.before.error_rows} → ${result.after.error_rows}. Preview only.`, true);
  } catch (error) { setStatus(error.message); }
  finally { previewCleaning.disabled = false; }
});
downloadCleaned.addEventListener("click", async () => {
  if (!currentJob) return;
  try {
    setStatus("Applying the validated plan to a copy…");
    const response = await fetch(`${API_BASE}/api/jobs/${currentJob.job_id}/apply`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ selected_indexes: selectedPlanIndexes() }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || "The selected actions could not be applied.");
    }
    const blob = await response.blob();
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "cleaned.csv";
    link.click();
    URL.revokeObjectURL(link.href);
    setStatus("Cleaned CSV downloaded. The uploaded source was not overwritten.", true);
  } catch (error) { setStatus(error.message); }
});

["dragenter", "dragover"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropZone.classList.add("is-dragging");
}));
["dragleave", "drop"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropZone.classList.remove("is-dragging");
}));
dropZone.addEventListener("drop", (event) => uploadFile(event.dataTransfer.files[0]));
dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});
