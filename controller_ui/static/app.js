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
  setupTabs();
});

function setupTabs() {
  const buttons = document.querySelectorAll("[data-tab-target]");
  const panels = document.querySelectorAll("[data-tab-panel]");
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const target = button.dataset.tabTarget;
      buttons.forEach((item) => item.classList.toggle("is-active", item === button));
      panels.forEach((panel) => panel.classList.toggle("is-active", panel.dataset.tabPanel === target));
    });
  });
}
