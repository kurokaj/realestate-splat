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

  document.querySelectorAll(".raw-remove-button").forEach((button) => {
    button.addEventListener("click", async () => {
      const relativePath = button.dataset.rawPath;
      if (!relativePath || !window.confirm(`Remove ${relativePath} from raw sources?`)) return;
      button.disabled = true;
      try {
        const response = await fetch(
          `/projects/${form.dataset.projectId}/raw?relative_path=${encodeURIComponent(relativePath)}`,
          { method: "DELETE" },
        );
        const body = await response.json();
        if (!response.ok) throw new Error(body.detail || "Removal failed");
        window.location.reload();
      } catch (error) {
        button.disabled = false;
        window.alert(`Remove failed: ${error}`);
      }
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("#raw-upload-form").forEach(setupRawUploadForm);
  document.querySelectorAll("#preprocess-queue-form").forEach((form) => {
    setupPreprocessProfileDefaults(form);
    setupPreprocessDependencyState(form);
  });
  document.querySelectorAll('form[action$="/colmap"]').forEach((form) => {
    setupColmapFormBehavior(form);
    setupProviderDependencyState(form);
  });
  document.querySelectorAll('form[action$="/training"]').forEach(setupProviderDependencyState);
  setupTabs();
  setupAutoRefresh();
  document.querySelectorAll("[data-colmap-viewer-url]").forEach(setupColmapViewer);
});

function setupColmapFormBehavior(form) {
  const matcherSelect = form.querySelector("[data-colmap-matcher-select]");
  const loopDetectionInput = form.querySelector("[data-colmap-loop-detection-input]");
  const loopDetectionRow = form.querySelector("[data-colmap-loop-detection-row]");
  const loopDetectionNote = form.querySelector("[data-colmap-loop-detection-note]");
  const vocabTreeInput = form.querySelector("[data-colmap-vocab-tree-input]");
  const vocabTreeRow = form.querySelector("[data-colmap-vocab-tree-row]");
  const vocabTreeNote = form.querySelector("[data-colmap-vocab-tree-note]");
  const matchingTypeSelect = form.querySelector("[data-colmap-matching-type-select]");
  const featureExtractorSelect = form.querySelector('[name="feature_extractor"]');
  if (!matcherSelect || !loopDetectionInput || !loopDetectionRow || !vocabTreeInput || !vocabTreeRow || !matchingTypeSelect || !featureExtractorSelect) return;

  function syncLoopDetectionState() {
    const sequentialEnabled = matcherSelect.value === "sequential";
    const vocabTreeEnabled = matcherSelect.value === "vocab_tree";
    loopDetectionInput.disabled = !sequentialEnabled;
    loopDetectionRow.classList.toggle("is-disabled", !sequentialEnabled);
    loopDetectionInput.closest(".checkbox-row")?.classList.toggle("is-disabled", !sequentialEnabled);
    vocabTreeInput.disabled = !vocabTreeEnabled;
    vocabTreeRow.classList.toggle("is-disabled", !vocabTreeEnabled);
    if (loopDetectionNote) {
      loopDetectionNote.textContent = sequentialEnabled
        ? "Recommended for sequential video-like runs."
        : "Locked because this only applies to sequential matching.";
    }
    if (vocabTreeNote) {
      vocabTreeNote.textContent = vocabTreeEnabled
        ? "Optional override; COLMAP supplies its default tree when empty."
        : "Locked because this only applies to vocabulary tree matching.";
    }
    const expectedPrefix = featureExtractorSelect.value.startsWith("ALIKED") ? "ALIKED_" : "SIFT_";
    const compatibleOptions = Array.from(matchingTypeSelect.options).filter((option) => option.value.startsWith(expectedPrefix));
    matchingTypeSelect.querySelectorAll("option").forEach((option) => {
      option.disabled = !option.value.startsWith(expectedPrefix);
    });
    if (!matchingTypeSelect.value.startsWith(expectedPrefix) && compatibleOptions.length) {
      matchingTypeSelect.value = compatibleOptions[0].value;
    }
    matchingTypeSelect.closest("label")?.classList.toggle("is-disabled", !matchingTypeSelect.value.startsWith(expectedPrefix));
  }

  matcherSelect.addEventListener("change", syncLoopDetectionState);
  featureExtractorSelect.addEventListener("change", syncLoopDetectionState);
  syncLoopDetectionState();
}

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

function setupPreprocessDependencyState(form) {
  if (form.dataset.hasCoverageVideo !== "false") return;
  form.querySelectorAll("[data-preprocess-video-only]").forEach((label) => {
    label.classList.add("is-disabled");
    label.querySelectorAll("input, select, textarea").forEach((control) => {
      control.disabled = true;
    });
    if (!label.querySelector(".dependency-note")) {
      const note = document.createElement("span");
      note.className = "muted dependency-note";
      note.textContent = "Video inputs only";
      label.appendChild(note);
    }
  });
}

function setupProviderDependencyState(form) {
  const provider = form.querySelector('[name="provider"]');
  if (!provider) return;
  const runpodFields = ["gpu_type_id", "container_disk_gb", "image", "repo_url", "git_ref", "endpoint_url"];

  function sync() {
    const enabled = provider.value !== "local_fake";
    runpodFields.forEach((name) => {
      const control = form.querySelector(`[name="${CSS.escape(name)}"]`);
      const label = control?.closest("label");
      if (!control || !label) return;
      control.disabled = !enabled;
      label.classList.toggle("is-disabled", !enabled);
    });
  }

  provider.addEventListener("change", sync);
  sync();
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
    "training_pod_starting",
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

async function setupColmapViewer(root) {
  const canvas = root.querySelector(".viewer-canvas");
  const status = root.querySelector(".viewer-status");
  const modeButtons = root.querySelectorAll("[data-viewer-mode]");
  const resetButton = root.querySelector("[data-viewer-reset]");
  if (!canvas || !status) return;

  try {
    const response = await fetch(root.dataset.colmapViewerUrl, { headers: { accept: "application/json" } });
    if (!response.ok) {
      status.textContent = "Viewer artifact not available yet.";
      return;
    }
    const scene = await response.json();
    const viewer = renderSparseViewer(canvas, scene);
    renderViewerLegend(root.querySelector("[data-viewer-legend]"), scene.camera_group_colors || {});
    modeButtons.forEach((button) => {
      button.addEventListener("click", () => {
        viewer.setMode(button.dataset.viewerMode || "orbit");
        modeButtons.forEach((item) => item.classList.toggle("is-active", item === button));
      });
    });
    resetButton?.addEventListener("click", () => viewer.reset());
    const pointCount = scene.point_sample_count || (scene.points || []).length || 0;
    const cameraCount = scene.camera_count || (scene.cameras || []).length || 0;
    status.textContent = `${pointCount} sampled points · ${cameraCount} cameras`;
  } catch (error) {
    status.textContent = `Viewer load failed: ${error}`;
  }
}

function renderViewerLegend(legend, colors) {
  if (!legend) return;
  legend.replaceChildren();
  Object.entries(colors).forEach(([group, color]) => {
    const item = document.createElement("span");
    item.className = "viewer-legend-item";
    const swatch = document.createElement("span");
    swatch.className = "viewer-legend-swatch";
    swatch.style.backgroundColor = `rgb(${(color.fill || color.stroke || [180, 190, 200]).join(", ")})`;
    const label = document.createElement("span");
    label.textContent = group;
    item.append(swatch, label);
    legend.append(item);
  });
}

function renderSparseViewer(canvas, scene) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return { setMode() {}, reset() {} };

  const points = Array.isArray(scene.points) ? scene.points : [];
  const cameras = Array.isArray(scene.cameras) ? scene.cameras : [];
  const bounds = scene.bounds || { center: [0, 0, 0], radius: 1 };
  const center = Array.isArray(bounds.center) ? bounds.center : [0, 0, 0];
  const radius = Number(bounds.radius || 1);
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  let mode = "orbit";
  let dragging = false;
  let dragMode = "orbit";
  let lastX = 0;
  let lastY = 0;
  let animationFrame = 0;
  let lastTick = 0;
  const pressed = new Set();

  const orbitDefaults = {
    yaw: 0.85,
    pitch: -0.45,
    distance: radius * 2.2,
    target: [...center],
  };
  const freeDefaults = {
    yaw: 0.85,
    pitch: -0.3,
    position: [center[0] - radius * 1.2, center[1] + radius * 0.35, center[2] + radius * 1.2],
    speed: Math.max(0.08, radius * 0.02),
  };
  const orbitState = { ...orbitDefaults };
  const freeState = { ...freeDefaults };

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function add(a, b) {
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  }

  function subtract(a, b) {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  }

  function scale(vector, factor) {
    return [vector[0] * factor, vector[1] * factor, vector[2] * factor];
  }

  function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  function cross(a, b) {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
  }

  function length(vector) {
    return Math.sqrt(dot(vector, vector));
  }

  function normalize(vector) {
    const vectorLength = length(vector) || 1;
    return [vector[0] / vectorLength, vector[1] / vectorLength, vector[2] / vectorLength];
  }

  function directionFromAngles(yaw, pitch) {
    const cosPitch = Math.cos(pitch);
    return normalize([
      Math.sin(yaw) * cosPitch,
      Math.sin(pitch),
      Math.cos(yaw) * cosPitch,
    ]);
  }

  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const nextWidth = Math.max(1, Math.round(rect.width * dpr));
    const nextHeight = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== nextWidth || canvas.height !== nextHeight) {
      canvas.width = nextWidth;
      canvas.height = nextHeight;
    }
  }

  function cameraFrame() {
    if (mode === "free") {
      const forward = directionFromAngles(freeState.yaw, freeState.pitch);
      const worldUp = [0, 1, 0];
      const right = normalize(cross(forward, worldUp));
      const up = normalize(cross(right, forward));
      return {
        position: freeState.position,
        forward,
        right,
        up,
      };
    }

    const forward = directionFromAngles(orbitState.yaw, orbitState.pitch);
    const worldUp = [0, 1, 0];
    const position = subtract(orbitState.target, scale(forward, orbitState.distance));
    const right = normalize(cross(forward, worldUp));
    const up = normalize(cross(right, forward));
    return {
      position,
      forward,
      right,
      up,
    };
  }

  function project(position) {
    const frame = cameraFrame();
    const relative = subtract(position, frame.position);
    const cameraX = dot(relative, frame.right);
    const cameraY = dot(relative, frame.up);
    const cameraZ = dot(relative, frame.forward);
    if (cameraZ <= radius * 0.02) return null;

    const focal = Math.min(canvas.width, canvas.height) * 0.72;
    return {
      x: canvas.width * 0.5 + (cameraX / cameraZ) * focal,
      y: canvas.height * 0.5 - (cameraY / cameraZ) * focal,
      depth: cameraZ,
    };
  }

  function draw() {
    resizeCanvas();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#0b1012";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const projectedPoints = points
      .map((point) => ({ point, projection: project(point.position) }))
      .filter((entry) => entry.projection)
      .sort((a, b) => b.projection.depth - a.projection.depth);

    for (const entry of projectedPoints) {
      const [r, g, b] = entry.point.color || [180, 200, 210];
      const alpha = clamp(1.1 - entry.projection.depth / (radius * 7), 0.18, 0.95);
      const size = clamp((radius * 0.5) / Math.max(entry.projection.depth, radius * 0.1), 1.2, 3.6) * dpr;
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
      ctx.fillRect(entry.projection.x, entry.projection.y, size, size);
    }

    for (const camera of cameras) {
      const origin = project(camera.position);
      if (!origin) continue;
      const forward = camera.forward || [0, 0, 1];
      const tip = project([
        camera.position[0] + forward[0] * radius * 0.08,
        camera.position[1] + forward[1] * radius * 0.08,
        camera.position[2] + forward[2] * radius * 0.08,
      ]);
      if (!tip) continue;
      const stroke = camera.stroke_color || [163, 178, 194];
      const fill = camera.fill_color || stroke;
      ctx.strokeStyle = `rgba(${stroke[0]}, ${stroke[1]}, ${stroke[2]}, 0.95)`;
      ctx.lineWidth = 1.2 * dpr;
      ctx.beginPath();
      ctx.moveTo(origin.x, origin.y);
      ctx.lineTo(tip.x, tip.y);
      ctx.stroke();
      ctx.fillStyle = `rgb(${fill[0]}, ${fill[1]}, ${fill[2]})`;
      ctx.beginPath();
      ctx.arc(origin.x, origin.y, 2.6 * dpr, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function panOrbit(dx, dy) {
    const frame = cameraFrame();
    const factor = orbitState.distance * 0.0016;
    const offset = add(scale(frame.right, -dx * factor), scale(frame.up, dy * factor));
    orbitState.target = add(orbitState.target, offset);
  }

  function moveFree(dx, dy, dz) {
    const frame = cameraFrame();
    const next = add(
      add(scale(frame.right, dx), scale(frame.up, dy)),
      scale(frame.forward, dz),
    );
    freeState.position = add(freeState.position, next);
  }

  function reset() {
    mode = "orbit";
    orbitState.yaw = orbitDefaults.yaw;
    orbitState.pitch = orbitDefaults.pitch;
    orbitState.distance = orbitDefaults.distance;
    orbitState.target = [...orbitDefaults.target];
    freeState.yaw = freeDefaults.yaw;
    freeState.pitch = freeDefaults.pitch;
    freeState.position = [...freeDefaults.position];
    freeState.speed = freeDefaults.speed;
    draw();
  }

  function setMode(nextMode) {
    mode = nextMode === "free" ? "free" : "orbit";
    draw();
  }

  function tick(timestamp) {
    animationFrame = 0;
    const deltaSeconds = lastTick ? Math.min(0.05, (timestamp - lastTick) / 1000) : 0.016;
    lastTick = timestamp;
    let changed = false;
    if (mode === "free") {
      const stride = freeState.speed * deltaSeconds * (pressed.has("shift") ? 3.5 : 1);
      if (pressed.has("w")) {
        moveFree(0, 0, stride);
        changed = true;
      }
      if (pressed.has("s")) {
        moveFree(0, 0, -stride);
        changed = true;
      }
      if (pressed.has("a")) {
        moveFree(-stride, 0, 0);
        changed = true;
      }
      if (pressed.has("d")) {
        moveFree(stride, 0, 0);
        changed = true;
      }
      if (pressed.has("q")) {
        moveFree(0, -stride, 0);
        changed = true;
      }
      if (pressed.has("e")) {
        moveFree(0, stride, 0);
        changed = true;
      }
    }
    if (changed) draw();
    if (pressed.size) animationFrame = window.requestAnimationFrame(tick);
  }

  function ensureAnimation() {
    if (!animationFrame) {
      animationFrame = window.requestAnimationFrame(tick);
    }
  }

  canvas.addEventListener("mousedown", (event) => {
    if (event.button === 2) event.preventDefault();
    canvas.focus();
    dragging = true;
    dragMode = event.shiftKey || event.button === 2 ? "pan" : "look";
    lastX = event.clientX;
    lastY = event.clientY;
  });
  window.addEventListener("mouseup", () => {
    dragging = false;
  });
  window.addEventListener("mousemove", (event) => {
    if (!dragging) return;
    const dx = event.clientX - lastX;
    const dy = event.clientY - lastY;
    lastX = event.clientX;
    lastY = event.clientY;
    if (mode === "free") {
      if (dragMode === "pan") {
        moveFree(-dx * freeState.speed * 0.02, dy * freeState.speed * 0.02, 0);
      } else {
        freeState.yaw += dx * 0.008;
        freeState.pitch = clamp(freeState.pitch - dy * 0.008, -1.5, 1.5);
      }
    } else {
      if (dragMode === "pan") {
        panOrbit(dx, dy);
      } else {
        orbitState.yaw += dx * 0.008;
        orbitState.pitch = clamp(orbitState.pitch - dy * 0.008, -1.5, 1.5);
      }
    }
    draw();
  });
  canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      if (mode === "free") {
        moveFree(0, 0, (event.deltaY > 0 ? -1 : 1) * freeState.speed * 1.8);
      } else {
        const factor = event.deltaY > 0 ? 1.12 : 0.89;
        orbitState.distance = clamp(orbitState.distance * factor, radius * 0.08, radius * 10);
      }
      draw();
    },
    { passive: false },
  );
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());
  canvas.addEventListener("keydown", (event) => {
    const key = event.key.toLowerCase();
    if (!["w", "a", "s", "d", "q", "e", "shift"].includes(key)) return;
    pressed.add(key);
    ensureAnimation();
    event.preventDefault();
  });
  canvas.addEventListener("keyup", (event) => {
    pressed.delete(event.key.toLowerCase());
  });
  canvas.addEventListener("blur", () => {
    pressed.clear();
  });
  if (typeof ResizeObserver !== "undefined") {
    const observer = new ResizeObserver(() => draw());
    observer.observe(canvas);
  }

  draw();
  return { setMode, reset };
}
