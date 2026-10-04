import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TrackballControls } from "three/addons/controls/TrackballControls.js";

const COLORS = {
  colmap: [90, 200, 250],
  arkit: [255, 209, 102],
  hybrid: [190, 142, 255],
  accepted: [91, 214, 132],
  rejected: [255, 92, 92],
  missing: [166, 177, 186],
  depthColmap: [66, 206, 255],
  depthArkit: [255, 158, 64],
};

function color(values) {
  const [red, green, blue] = values || [190, 200, 210];
  return new THREE.Color(red / 255, green / 255, blue / 255);
}

function matrixFromRows(rows) {
  if (!Array.isArray(rows) || rows.length !== 4 || rows.some((row) => !Array.isArray(row) || row.length !== 4)) {
    return new THREE.Matrix4();
  }
  return new THREE.Matrix4().set(...rows.flat().map(Number));
}

function vector(values, transform) {
  if (!Array.isArray(values) || values.length !== 3) return null;
  return new THREE.Vector3(Number(values[0]), Number(values[1]), Number(values[2])).applyMatrix4(transform);
}

function direction(values, transform) {
  if (!Array.isArray(values) || values.length !== 3) return null;
  return new THREE.Vector3(Number(values[0]), Number(values[1]), Number(values[2])).transformDirection(transform);
}

function materialLine(rgb, opacity = 1) {
  return new THREE.LineBasicMaterial({
    color: color(rgb),
    transparent: opacity < 1,
    opacity,
    depthTest: true,
    depthWrite: opacity >= 1,
    vertexColors: false,
  });
}

function makeLineSegments(segmentPositions, rgb, opacity = 1) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(segmentPositions, 3));
  const object = new THREE.LineSegments(geometry, materialLine(rgb, opacity));
  object.frustumCulled = false;
  return object;
}

function trajectorySegments(frames, key, transform) {
  const positions = [];
  for (let index = 1; index < frames.length; index += 1) {
    const previous = vector(frames[index - 1]?.[key], transform);
    const current = vector(frames[index]?.[key], transform);
    if (!previous || !current) continue;
    positions.push(previous.x, previous.y, previous.z, current.x, current.y, current.z);
  }
  return positions;
}

function makePointObject(entries, radius, fallbackColor, opacity = 1) {
  const positions = [];
  const colors = [];
  const items = [];
  entries.forEach((entry) => {
    if (!entry.position) return;
    positions.push(entry.position.x, entry.position.y, entry.position.z);
    const entryColor = color(entry.color || fallbackColor);
    colors.push(entryColor.r, entryColor.g, entryColor.b);
    items.push(entry.item);
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  const material = new THREE.PointsMaterial({
    size: Math.max(radius * 0.012, 0.008),
    sizeAttenuation: true,
    vertexColors: true,
    transparent: opacity < 1,
    opacity,
    depthTest: true,
    depthWrite: opacity >= 1,
  });
  const object = new THREE.Points(geometry, material);
  object.userData.pickItems = items;
  object.frustumCulled = false;
  return object;
}

function addPointCloud(group, rows, transform, radius) {
  const positions = new Float32Array(rows.length * 3);
  const colors = new Float32Array(rows.length * 3);
  let writeIndex = 0;
  rows.forEach((row) => {
    const point = vector(row.position, transform);
    if (!point) return;
    const pointColor = color(row.color);
    positions[writeIndex] = point.x;
    positions[writeIndex + 1] = point.y;
    positions[writeIndex + 2] = point.z;
    colors[writeIndex] = pointColor.r;
    colors[writeIndex + 1] = pointColor.g;
    colors[writeIndex + 2] = pointColor.b;
    writeIndex += 3;
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions.slice(0, writeIndex), 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors.slice(0, writeIndex), 3));
  const material = new THREE.PointsMaterial({
    size: Math.max(radius * 0.0022, 0.002),
    sizeAttenuation: true,
    vertexColors: true,
    opacity: 0.92,
    transparent: true,
    depthWrite: true,
  });
  const cloud = new THREE.Points(geometry, material);
  cloud.frustumCulled = false;
  group.add(cloud);
  return cloud;
}

function appendCameraFrustum(segments, origin, forward, upHint, length) {
  if (!origin || !forward) return;
  const normalizedForward = forward.clone().normalize();
  let normalizedUp = upHint?.clone().normalize() || new THREE.Vector3(0, 1, 0);
  if (Math.abs(normalizedForward.dot(normalizedUp)) > 0.96) normalizedUp = new THREE.Vector3(0, 0, 1);
  const right = new THREE.Vector3().crossVectors(normalizedForward, normalizedUp).normalize();
  const up = new THREE.Vector3().crossVectors(right, normalizedForward).normalize();
  const planeCenter = origin.clone().addScaledVector(normalizedForward, length);
  const halfWidth = length * 0.46;
  const halfHeight = length * 0.32;
  const corners = [
    planeCenter.clone().addScaledVector(right, -halfWidth).addScaledVector(up, halfHeight),
    planeCenter.clone().addScaledVector(right, halfWidth).addScaledVector(up, halfHeight),
    planeCenter.clone().addScaledVector(right, halfWidth).addScaledVector(up, -halfHeight),
    planeCenter.clone().addScaledVector(right, -halfWidth).addScaledVector(up, -halfHeight),
  ];
  const appendSegment = (start, end) => segments.push(start.x, start.y, start.z, end.x, end.y, end.z);
  appendSegment(origin, planeCenter);
  corners.forEach((corner) => appendSegment(origin, corner));
  corners.forEach((corner, index) => appendSegment(corner, corners[(index + 1) % corners.length]));
}

function makeCameraFrustums(rows, positionKey, forwardKey, upKey, transform, length, rgb) {
  const segments = [];
  rows.forEach((row) => {
    const origin = vector(row[positionKey], transform);
    const forward = direction(row[forwardKey], transform);
    const up = direction(row[upKey], transform);
    appendCameraFrustum(segments, origin, forward, up, length);
  });
  return makeLineSegments(segments, rgb, 0.9);
}

function calculateBounds(sceneData, transform) {
  const box = new THREE.Box3();
  (sceneData.points || []).forEach((row) => {
    const point = vector(row.position, transform);
    if (point) box.expandByPoint(point);
  });
  (sceneData.cameras || []).forEach((row) => {
    const point = vector(row.position, transform);
    if (point) box.expandByPoint(point);
  });
  (sceneData.depth_diagnostic?.points || []).forEach((row) => {
    const point = vector(row.position, new THREE.Matrix4());
    if (point) box.expandByPoint(point);
  });
  const captures = sceneData.alignment?.captures || [];
  captures.forEach((capture) => {
    (capture.frames || []).forEach((frame) => {
      ["arkit_position", "colmap_position", "hybrid_position"].forEach((key) => {
        const point = vector(frame[key], transform);
        if (point) box.expandByPoint(point);
      });
    });
  });
  if (box.isEmpty()) box.expandByPoint(new THREE.Vector3()).expandByPoint(new THREE.Vector3(1, 1, 1));
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  sphere.radius = Math.max(sphere.radius, 0.5);
  return { box, sphere };
}

function percentile(values, fraction) {
  if (!values.length) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * fraction)))];
}

export function renderSparseViewer(canvas, sceneData) {
  const alignment = sceneData.alignment && typeof sceneData.alignment === "object" ? sceneData.alignment : null;
  const transform = matrixFromRows(alignment?.display_transform_row_major);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
  renderer.setClearColor(0x0b1012, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  const world = new THREE.Scene();
  const { box, sphere } = calculateBounds(sceneData, transform);
  const radius = sphere.radius;
  const camera = new THREE.PerspectiveCamera(48, 1, Math.max(radius / 10000, 0.001), radius * 100);
  camera.up.set(0, 1, 0);

  const orbit = new OrbitControls(camera, canvas);
  orbit.enableDamping = true;
  orbit.dampingFactor = 0.075;
  orbit.screenSpacePanning = true;
  orbit.zoomToCursor = true;
  orbit.minDistance = radius * 0.015;
  orbit.maxDistance = radius * 20;

  const trackball = new TrackballControls(camera, canvas);
  trackball.enabled = false;
  trackball.rotateSpeed = 3.0;
  trackball.zoomSpeed = 1.25;
  trackball.panSpeed = 0.8;
  trackball.staticMoving = false;
  trackball.dynamicDampingFactor = 0.12;

  const layers = {
    points: new THREE.Group(),
    depth_colmap: new THREE.Group(),
    depth_arkit: new THREE.Group(),
    colmap: new THREE.Group(),
    arkit: new THREE.Group(),
    links: new THREE.Group(),
    hybrid: new THREE.Group(),
  };
  const outlierLayers = {
    colmap: new THREE.Group(),
    arkit: new THREE.Group(),
    links: new THREE.Group(),
    hybrid: new THREE.Group(),
  };
  Object.values(layers).forEach((group) => world.add(group));
  Object.values(outlierLayers).forEach((group) => world.add(group));

  addPointCloud(layers.points, Array.isArray(sceneData.points) ? sceneData.points : [], transform, radius);
  const depthRows = Array.isArray(sceneData.depth_diagnostic?.points) ? sceneData.depth_diagnostic.points : [];
  const identity = new THREE.Matrix4();
  addPointCloud(
    layers.depth_colmap,
    depthRows.filter((row) => row.pose_source === "colmap_registered"),
    identity,
    radius,
  );
  addPointCloud(
    layers.depth_arkit,
    depthRows.filter((row) => row.pose_source === "arkit_propagated"),
    identity,
    radius,
  );

  const cameraRows = Array.isArray(sceneData.cameras) ? sceneData.cameras : [];
  const cameraByName = new Map(cameraRows.map((row) => [String(row.name || row.image_name || ""), row]));
  const cameraEntries = cameraRows.map((row) => ({
    position: vector(row.position, transform),
    color: alignment ? COLORS.colmap : (row.fill_color || row.stroke_color),
    item: row,
  }));
  const cameraPoints = makePointObject(cameraEntries, radius, COLORS.colmap);
  layers.colmap.add(cameraPoints);
  layers.colmap.add(makeCameraFrustums(cameraRows, "position", "forward", "up", transform, radius * 0.022, COLORS.colmap));

  const trajectoryPickObjects = [];
  if (alignment) {
    const captures = Array.isArray(alignment.captures) ? alignment.captures : [];
    captures.forEach((capture) => {
      const frames = Array.isArray(capture.frames) ? capture.frames : [];
      layers.colmap.add(makeLineSegments(trajectorySegments(frames, "colmap_position", transform), COLORS.colmap, 0.92));
      layers.arkit.add(makeLineSegments(trajectorySegments(frames, "arkit_position", transform), COLORS.arkit, 0.88));
      layers.hybrid.add(makeLineSegments(trajectorySegments(frames, "hybrid_position", transform), COLORS.hybrid, 1));
      layers.arkit.add(makeCameraFrustums(frames, "arkit_position", "arkit_forward", "arkit_up", transform, radius * 0.022, COLORS.arkit));

      const colmapEntries = [];
      const arkitEntries = [];
      const hybridEntries = [];
      const outlierColmapEntries = [];
      const outlierArkitEntries = [];
      const replacementEntries = [];
      const acceptedLinks = [];
      const rejectedLinks = [];
      const hybridFrustums = [];

      frames.forEach((frame) => {
        const colmapPosition = vector(frame.colmap_position, transform);
        const arkitPosition = vector(frame.arkit_position, transform);
        const hybridPosition = vector(frame.hybrid_position, transform);
        if (colmapPosition) {
          const entry = { position: colmapPosition, color: COLORS.colmap, item: { frame, source: "COLMAP" } };
          colmapEntries.push(entry);
          if (frame.classification === "outlier") outlierColmapEntries.push(entry);
        }
        if (arkitPosition) {
          const arkitColor = frame.classification === "outlier"
            ? COLORS.rejected
            : frame.classification === "missing"
              ? COLORS.missing
              : COLORS.arkit;
          const entry = { position: arkitPosition, color: arkitColor, item: { frame, source: "ARKit" } };
          arkitEntries.push(entry);
          if (frame.classification === "outlier") outlierArkitEntries.push(entry);
        }
        if (hybridPosition) {
          const source = frame.hybrid_pose_source === "arkit" ? "Hybrid (ARKit)" : "Hybrid (COLMAP)";
          const entry = { position: hybridPosition, color: COLORS.hybrid, item: { frame, source } };
          hybridEntries.push(entry);
          if (frame.hybrid_pose_source === "arkit") replacementEntries.push(entry);

          const sourceRow = frame.hybrid_pose_source === "arkit" ? frame : cameraByName.get(String(frame.image_name || ""));
          const forwardValues = frame.hybrid_pose_source === "arkit" ? frame.arkit_forward : sourceRow?.forward;
          const upValues = frame.hybrid_pose_source === "arkit" ? frame.arkit_up : sourceRow?.up;
          const forward = direction(forwardValues, transform);
          const up = direction(upValues, transform);
          appendCameraFrustum(hybridFrustums, hybridPosition, forward, up, radius * 0.022);
        }
        if (colmapPosition && arkitPosition) {
          const target = frame.classification === "outlier" ? rejectedLinks : acceptedLinks;
          target.push(
            colmapPosition.x, colmapPosition.y, colmapPosition.z,
            arkitPosition.x, arkitPosition.y, arkitPosition.z,
          );
        }
      });

      const colmapObject = makePointObject(colmapEntries, radius, COLORS.colmap);
      const arkitObject = makePointObject(arkitEntries, radius, COLORS.arkit);
      const hybridObject = makePointObject(hybridEntries, radius, COLORS.hybrid);
      const outlierColmapObject = makePointObject(outlierColmapEntries, radius, COLORS.colmap);
      const outlierArkitObject = makePointObject(outlierArkitEntries, radius, COLORS.rejected);
      const replacementObject = makePointObject(replacementEntries, radius, COLORS.hybrid);
      layers.colmap.add(colmapObject);
      layers.arkit.add(arkitObject);
      layers.hybrid.add(hybridObject, makeLineSegments(hybridFrustums, COLORS.hybrid, 0.9));
      layers.links.add(makeLineSegments(acceptedLinks, COLORS.accepted, 0.28));
      layers.links.add(makeLineSegments(rejectedLinks, COLORS.rejected, 0.95));
      outlierLayers.colmap.add(outlierColmapObject);
      outlierLayers.arkit.add(outlierArkitObject);
      outlierLayers.links.add(makeLineSegments(rejectedLinks, COLORS.rejected, 1));
      outlierLayers.hybrid.add(replacementObject);
      trajectoryPickObjects.push(
        colmapObject,
        arkitObject,
        hybridObject,
        outlierColmapObject,
        outlierArkitObject,
        replacementObject,
      );
    });
  }

  const axes = new THREE.AxesHelper(radius * 0.22);
  world.add(axes);
  if (alignment?.display_transform_row_major) {
    const pointHeights = (sceneData.points || [])
      .map((row) => vector(row.position, transform)?.y)
      .filter((value) => Number.isFinite(value));
    const floorY = percentile(pointHeights, 0.02);
    const grid = new THREE.GridHelper(radius * 3, 24, 0x52646d, 0x26343a);
    grid.position.y = floorY;
    grid.material.transparent = true;
    grid.material.opacity = 0.34;
    world.add(grid);
  }

  let mode = "orbit";
  let outliersOnly = false;
  const visibleLayers = {
    points: true,
    depth_colmap: true,
    depth_arkit: true,
    colmap: true,
    arkit: true,
    links: true,
    hybrid: true,
  };
  const raycaster = new THREE.Raycaster();
  raycaster.params.Points.threshold = Math.max(radius * 0.018, 0.015);
  const pointer = new THREE.Vector2();

  function syncVisibility() {
    layers.points.visible = visibleLayers.points;
    layers.depth_colmap.visible = visibleLayers.depth_colmap;
    layers.depth_arkit.visible = visibleLayers.depth_arkit;
    layers.colmap.visible = visibleLayers.colmap && !outliersOnly;
    layers.arkit.visible = visibleLayers.arkit && !outliersOnly;
    layers.links.visible = visibleLayers.links && !outliersOnly;
    layers.hybrid.visible = visibleLayers.hybrid && !outliersOnly;
    outlierLayers.colmap.visible = visibleLayers.colmap && outliersOnly;
    outlierLayers.arkit.visible = visibleLayers.arkit && outliersOnly;
    outlierLayers.links.visible = visibleLayers.links && outliersOnly;
    outlierLayers.hybrid.visible = visibleLayers.hybrid && outliersOnly;
  }

  function frameScene() {
    const center = sphere.center.clone();
    camera.position.copy(center).add(new THREE.Vector3(radius * 1.35, radius * 0.8, radius * 1.35));
    camera.near = Math.max(radius / 10000, 0.001);
    camera.far = radius * 100;
    camera.updateProjectionMatrix();
    orbit.target.copy(center);
    orbit.update();
    trackball.target.copy(center);
    trackball.update();
  }

  function resize() {
    const width = Math.max(1, canvas.clientWidth);
    const height = Math.max(1, canvas.clientHeight);
    const pixelRatio = renderer.getPixelRatio();
    if (canvas.width !== Math.round(width * pixelRatio) || canvas.height !== Math.round(height * pixelRatio)) {
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      trackball.handleResize();
    }
  }

  function animate() {
    resize();
    if (mode === "free") trackball.update();
    else orbit.update();
    renderer.render(world, camera);
    window.requestAnimationFrame(animate);
  }

  function setMode(nextMode) {
    mode = nextMode === "free" ? "free" : "orbit";
    orbit.enabled = mode === "orbit";
    trackball.enabled = mode === "free";
    if (mode === "free") trackball.target.copy(orbit.target);
    else orbit.target.copy(trackball.target);
  }

  function setLayer(layer, visible) {
    if (Object.prototype.hasOwnProperty.call(visibleLayers, layer)) {
      visibleLayers[layer] = Boolean(visible);
      syncVisibility();
    }
  }

  function setOutliersOnly(enabled) {
    outliersOnly = Boolean(enabled);
    syncVisibility();
  }

  function reset() {
    setMode("orbit");
    frameScene();
  }

  function setPointer(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
  }

  function pickFromObjects(clientX, clientY, objects) {
    setPointer(clientX, clientY);
    const hit = raycaster.intersectObjects(objects, false).find((entry) => {
      const items = entry.object.userData.pickItems;
      return Array.isArray(items) && Number.isInteger(entry.index) && items[entry.index];
    });
    return hit ? hit.object.userData.pickItems[hit.index] : null;
  }

  function pickCamera(clientX, clientY) {
    return pickFromObjects(clientX, clientY, [cameraPoints]);
  }

  function pickTrajectory(clientX, clientY) {
    return pickFromObjects(clientX, clientY, trajectoryPickObjects.filter((object) => object.visible && object.parent?.visible));
  }

  syncVisibility();
  frameScene();
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(resize).observe(canvas);
  animate();
  return {
    rendererName: "Three.js r186",
    setMode,
    setLayer,
    setOutliersOnly,
    reset,
    pickCamera,
    pickTrajectory,
  };
}
