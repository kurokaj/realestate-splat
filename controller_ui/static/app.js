function fileRoleOptions() {
  return [
    ["auto", "auto"],
    ["coverage_video", "coverage video"],
    ["coverage_image", "coverage image"],
    ["hero_image", "hero image"],
  ];
}

function colmapPolicyOptions() {
  return [
    ["", "auto"],
    ["required", "required"],
    ["optional", "optional"],
    ["ignore", "ignore"],
  ];
}

function optionList(options) {
  return options.map(([value, label]) => `<option value="${value}">${label}</option>`).join("");
}

function setupRawUploadForm(form) {
  const fileInput = form.querySelector('input[type="file"]');
  const tableWrap = form.querySelector(".upload-table-wrap");
  const tbody = form.querySelector("#upload-metadata-table tbody");
  const result = form.querySelector("#upload-result");
  const metadataInput = form.querySelector('input[name="metadata_json"]');

  fileInput.addEventListener("change", () => {
    tbody.innerHTML = "";
    Array.from(fileInput.files).forEach((file) => {
      const row = document.createElement("tr");
      row.dataset.filename = file.name;
      row.innerHTML = `
        <td class="mono">${file.name}</td>
        <td><select data-field="role">${optionList(fileRoleOptions())}</select></td>
        <td><input data-field="location" placeholder="room / angle"></td>
        <td><select data-field="colmap_policy">${optionList(colmapPolicyOptions())}</select></td>
      `;
      tbody.appendChild(row);
    });
    tableWrap.hidden = fileInput.files.length === 0;
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    result.hidden = false;
    result.textContent = "Uploading...";

    const metadata = [];
    tbody.querySelectorAll("tr").forEach((row) => {
      const role = row.querySelector('[data-field="role"]').value;
      const location = row.querySelector('[data-field="location"]').value.trim();
      const colmapPolicy = row.querySelector('[data-field="colmap_policy"]').value;
      const item = { filename: row.dataset.filename };
      if (role && role !== "auto") item.role = role;
      if (location) item.location = location;
      if (colmapPolicy) item.colmap_policy = colmapPolicy;
      if (Object.keys(item).length > 1) metadata.push(item);
    });
    metadataInput.value = metadata.length ? JSON.stringify({ files: metadata }) : "";

    const payload = new FormData(form);
    try {
      const response = await fetch(`/projects/${form.dataset.projectId}/raw`, {
        method: "POST",
        body: payload,
      });
      const body = await response.json();
      result.textContent = JSON.stringify(body, null, 2);
      if (!response.ok) return;
      if (!payload.has("dry_run")) {
        window.setTimeout(() => window.location.reload(), 700);
      }
    } catch (error) {
      result.textContent = `Upload failed: ${error}`;
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("#raw-upload-form").forEach(setupRawUploadForm);
  document.querySelectorAll("#preprocess-queue-form").forEach(setupPreprocessProfileDefaults);
  setupTabs();
  setupAutoRefresh();
});

function setupTabs() {
  const buttons = document.querySelectorAll("[data-tab-target]");
  const panels = document.querySelectorAll("[data-tab-panel]");
  const projectRoot = document.querySelector("[data-project-id]");
  const storageKey = projectRoot ? `buildvision3d:${projectRoot.dataset.projectId}:active-tab` : "";

  function activate(target) {
    buttons.forEach((item) => item.classList.toggle("is-active", item.dataset.tabTarget === target));
    panels.forEach((panel) => panel.classList.toggle("is-active", panel.dataset.tabPanel === target));
    if (storageKey) sessionStorage.setItem(storageKey, target);
  }

  const requested = window.location.hash.replace("#", "");
  const stored = storageKey ? sessionStorage.getItem(storageKey) : "";
  const initial = requested || stored;
  if (initial && document.querySelector(`[data-tab-panel="${CSS.escape(initial)}"]`)) {
    activate(initial);
  }

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      activate(button.dataset.tabTarget);
    });
  });
}

function setupPreprocessProfileDefaults(form) {
  let defaultsByProfile = {};
  try {
    defaultsByProfile = JSON.parse(form.dataset.profileDefaults || "{}");
  } catch {
    defaultsByProfile = {};
  }
  const profileSelect = form.querySelector('select[name="profile"]');
  if (!profileSelect) return;

  profileSelect.addEventListener("change", () => {
    const defaults = defaultsByProfile[profileSelect.value] || {};
    Object.entries(defaults).forEach(([field, value]) => {
      const input = form.querySelector(`[name="${CSS.escape(field)}"]`);
      if (input) input.value = value;
    });
  });
}

function setupAutoRefresh() {
  const projectRoot = document.querySelector("[data-project-id]");
  if (!projectRoot) return;

  const projectId = projectRoot.dataset.projectId;
  let currentSignature = projectRoot.dataset.stageSignature || "";
  const activeStatuses = new Set([
    "preprocess_queued",
    "preprocess_running",
    "colmap_queued",
    "colmap_pod_starting",
    "colmap_running",
    "training_queued",
    "training_running",
  ]);

  async function poll() {
    try {
      const response = await fetch(`/stage-runs?project_id=${encodeURIComponent(projectId)}`, {
        headers: { accept: "application/json" },
      });
      if (!response.ok) return;
      const runs = await response.json();
      const nextSignature = runs
        .map((run) => {
          const progress = run.progress_json || {};
          return `${run.id || ""}:${run.status || ""}:${run.updated_at || ""}:${progress.percent || ""}:${progress.message || ""}`;
        })
        .join("|");
      const hasActiveRun = runs.some((run) => activeStatuses.has(run.status));
      if (nextSignature && nextSignature !== currentSignature) {
        window.location.reload();
        return;
      }
      if (!hasActiveRun) {
        window.clearInterval(timer);
      }
      currentSignature = nextSignature || currentSignature;
    } catch {
      window.clearInterval(timer);
    }
  }

  const timer = window.setInterval(poll, 5000);
}
