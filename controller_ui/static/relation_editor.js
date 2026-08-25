(function () {
  "use strict";

  function safeJson(value, fallback) {
    try {
      const parsed = JSON.parse(value || "");
      return parsed ?? fallback;
    } catch (_error) {
      return fallback;
    }
  }

  function nodeMarkup(entry) {
    const locality = entry.locality || (entry.locations || [])[0] || "Unassigned";
    return `
      <div class="relation-node-card">
        <strong>${escapeHtml(locality)}</strong>
        <span>${escapeHtml(entry.image_count || 0)} images</span>
        <span>${escapeHtml(entry.hero_count || 0)} heroes</span>
      </div>`;
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>\"']/g, (character) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;",
    }[character]));
  }

  function createRelationEditor({ container, entries, relations, projectId, onChange }) {
    if (!window.Drawflow) {
      container.innerHTML = "<p class='empty'>The relation editor library could not be loaded.</p>";
      return { getRelations: () => relations || [], destroy: () => {} };
    }
    const editor = new Drawflow(container);
    editor.reroute = true;
    editor.reroute_fix_curvature = true;
    editor.start();

    const appToDrawflow = new Map();
    const drawflowToApp = new Map();
    const stateKey = `buildvision3d:${projectId}:matching-editor-state`;
    const savedState = safeJson(sessionStorage.getItem(stateKey), null);

    entries.forEach((entry, index) => {
      const nodeId = editor.addNode(
        `source_${index}`,
        1,
        1,
        40 + (index % 3) * 260,
        40 + Math.floor(index / 3) * 170,
        "source-group",
        { applicationId: entry.id },
        nodeMarkup(entry),
      );
      const drawflowId = String(nodeId);
      appToDrawflow.set(String(entry.id), drawflowId);
      drawflowToApp.set(drawflowId, String(entry.id));
    });

    if (savedState?.drawflow?.Home?.data) {
      Object.values(savedState.drawflow.Home.data).forEach((savedNode) => {
        const applicationId = savedNode?.data?.applicationId;
        const currentNodeId = appToDrawflow.get(String(applicationId));
        const currentNode = currentNodeId && container.querySelector(`#node-${currentNodeId}`);
        if (currentNode && Number.isFinite(Number(savedNode.pos_x))) {
          currentNode.style.left = `${Number(savedNode.pos_x)}px`;
          currentNode.style.top = `${Number(savedNode.pos_y)}px`;
        }
      });
    }

    function getRelations() {
      const exported = editor.export();
      const nodes = exported?.drawflow?.Home?.data || {};
      const result = [];
      Object.entries(nodes).forEach(([nodeId, node]) => {
        const sourceId = drawflowToApp.get(String(nodeId));
        if (!sourceId) return;
        Object.values(node.outputs || {}).forEach((output) => {
          (output.connections || []).forEach((connection) => {
            const targetId = drawflowToApp.get(String(connection.node));
            if (targetId && sourceId !== targetId) {
              result.push({ sourceId, targetId, from: sourceId, to: targetId, matching_style: "exhaustive" });
            }
          });
        });
      });
      return result;
    }

    function saveState() {
      try { sessionStorage.setItem(stateKey, JSON.stringify(editor.export())); } catch (_error) { /* best effort */ }
    }

    (relations || []).forEach((relation) => {
      const source = appToDrawflow.get(String(relation.sourceId || relation.from));
      const target = appToDrawflow.get(String(relation.targetId || relation.to));
      if (source && target) {
        try { editor.addConnection(source, target, "output_1", "input_1"); } catch (_error) { /* stale relation */ }
      }
    });
    function notifyChange() {
      saveState();
      onChange?.(getRelations());
    }
    editor.on("connectionCreated", notifyChange);
    editor.on("connectionRemoved", notifyChange);
    editor.on("nodeMoved", saveState);
    editor.on("nodeRemoved", notifyChange);

    return {
      getRelations,
      saveEditorState: saveState,
      destroy: () => editor.clear(),
    };
  }

  window.BuildvisionRelationEditor = { createRelationEditor };
})();
