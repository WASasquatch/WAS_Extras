import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXT_NAME = "WAS_Extras.PowerLoraMergerUI";
const NODE_NAME = "WASPowerLoraMerger";

function toBool(v, defaultValue = true) {
  if (v == null) return defaultValue;
  if (typeof v === "boolean") return v;
  if (typeof v === "number") return !!v;
  if (typeof v === "string") {
    const s = v.trim().toLowerCase();
    if (s === "true" || s === "1" || s === "yes" || s === "y" || s === "on") return true;
    if (s === "false" || s === "0" || s === "no" || s === "n" || s === "off" || s === "") return false;
    return defaultValue;
  }
  return !!v;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function refreshNodeDefsAndUpdate(node) {
  try {
    const command = app?.extensionManager?.command;
    if (command && typeof command.execute === "function") {
      try {
        await command.execute("Comfy.RefreshNodeDefinitions");
      } catch (e) {
      }
    }

    const prev = getLoraOptions(node);
    let fromBackend = null;

    for (let attempt = 0; attempt < 8; attempt++) {
      const defs = await api.getNodeDefs();
      const def = defs?.[NODE_NAME] ?? (Array.isArray(defs) ? defs.find((d) => d?.name === NODE_NAME) : null);
      const cat = def?.input?.hidden?.was_lora_catalog;
      if (Array.isArray(cat) && cat.length) {
        const next = normalizeLoraOptions(cat);
        const changed = JSON.stringify(next) !== JSON.stringify(prev);
        if (changed || attempt === 7) {
          fromBackend = next;
          break;
        }
      }

      await sleep(150);
    }

    if (Array.isArray(fromBackend) && fromBackend.length) {
      setLoraOptions(node, fromBackend);
      rebuildLoraRows(node, fromBackend);
      node.setDirtyCanvas(true, true);
      return;
    }
  } catch (e) {
  }
}

function showLoraChooser(event, callback, parentMenu, loras) {
  const canvas = app.canvas;
  const safeLoras = normalizeLoraOptions(loras);
  const nestedMenuValues = buildNestedLoraMenuValues(safeLoras, callback);

  let safeEvent = event;
  if (!(safeEvent instanceof MouseEvent) && !(safeEvent instanceof CustomEvent)) {
    try {
      const canvasEl = canvas?.canvas;
      const rect = canvasEl?.getBoundingClientRect?.();
      const mx = canvas?.mouse?.[0] ?? canvas?.last_mouse?.[0] ?? 0;
      const my = canvas?.mouse?.[1] ?? canvas?.last_mouse?.[1] ?? 0;
      const clientX = (rect?.left ?? 0) + mx;
      const clientY = (rect?.top ?? 0) + my;
      safeEvent = new MouseEvent("contextmenu", {
        bubbles: true,
        cancelable: true,
        clientX,
        clientY,
      });
    } catch (e) {
      safeEvent = new MouseEvent("contextmenu", { bubbles: true, cancelable: true });
    }
  }

  new LiteGraph.ContextMenu(nestedMenuValues, {
    event: safeEvent,
    parentMenu: parentMenu ?? undefined,
    title: "WAS LoRA Picker",
    scale: Math.max(1, canvas?.ds?.scale ?? 1),
    className: "dark",
    callback,
  });
}

function splitLoraPath(path) {
  const normalized = String(path ?? "").replace(/\\/g, "/");
  return normalized.split("/").filter((p) => p.length > 0);
}

function buildLoraTree(loras) {
  const root = { folders: new Map(), files: [], all: [] };

  for (const raw of loras) {
    if (typeof raw !== "string") continue;
    if (raw === "None") continue;

    const parts = splitLoraPath(raw);
    if (!parts.length) continue;

    let node = root;
    node.all.push(raw);
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i];
      const isLeaf = i === parts.length - 1;
      if (isLeaf) {
        node.files.push({ name: part, full: raw });
      } else {
        if (!node.folders.has(part)) {
          node.folders.set(part, { folders: new Map(), files: [], all: [] });
        }
        node = node.folders.get(part);
        node.all.push(raw);
      }
    }
  }

  return root;
}

function filterLoras(loras, query) {
  const q = String(query ?? "").trim().toLowerCase();
  if (!q) return Array.isArray(loras) ? loras.slice() : [];
  const arr = Array.isArray(loras) ? loras : [];
  return arr.filter((x) => typeof x === "string" && x.toLowerCase().includes(q));
}

function makeSearchMenuItem(title, allLoras, onPick) {
  return {
    content: title,
    callback: (_value, _options, event, parentMenu, node) => {
      let q = "";
      try {
        q = window?.prompt?.("Filter LoRAs", "") ?? "";
      } catch (e) {
        q = "";
      }

      const filtered = filterLoras(allLoras, q);
      const t = buildLoraTree(filtered);
      const menuValues = treeToMenuValues(t, onPick, filtered);

      new LiteGraph.ContextMenu(menuValues, {
        event,
        parentMenu: parentMenu ?? undefined,
        title: "WAS LoRA Picker",
        scale: Math.max(1, app?.canvas?.ds?.scale ?? 1),
        className: "dark",
        callback: (_v, _o, _e, _pm, _n) => {
        },
      }, node);
    },
  };
}

function treeToMenuValues(treeNode, onPick, allLorasForNode) {
  const values = [];

  // Folder-aware search: show a filter action for the current folder/subtree.
  const folderAll = Array.isArray(allLorasForNode) ? allLorasForNode : (treeNode?.all ?? []);
  if (Array.isArray(folderAll) && folderAll.length) {
    values.push(makeSearchMenuItem("🔎 Filter in this folder", folderAll, onPick));
    values.push(null);
  }

  const folderNames = Array.from(treeNode.folders.keys()).sort((a, b) =>
    a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" }),
  );

  for (const folderName of folderNames) {
    const child = treeNode.folders.get(folderName);
    values.push({
      content: `📁 ${folderName}`,
      has_submenu: true,
      callback: () => {
      },
      submenu: {
        options: treeToMenuValues(child, onPick, child?.all ?? []),
      },
    });
  }

  const files = Array.isArray(treeNode.files) ? treeNode.files.slice() : [];
  files.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: "base" }));
  for (const f of files) {
    values.push({
      content: f.name,
      rgthree_originalValue: f.full,
      callback: (_value, options, event, parentMenu, node) => {
        onPick?.(f.full, options, event, parentMenu, node);
      },
    });
  }

  return values;
}

function buildNestedLoraMenuValues(loras, onPick) {
  const safe = normalizeLoraOptions(loras);

  const out = [];

  // Root-level search across the entire LoRA catalog.
  const safeNoNone = safe.filter((x) => typeof x === "string" && x !== "None");
  if (safeNoNone.length) {
    out.push(makeSearchMenuItem("🔎 Filter LoRAs", safeNoNone, onPick));
    out.push(null);
  }

  if (safe.includes("None")) {
    out.push({
      content: "None",
      rgthree_originalValue: "None",
      callback: (_value, options, event, parentMenu, node) => {
        onPick?.("None", options, event, parentMenu, node);
      },
    });
  }

  const tree = buildLoraTree(safe);
  const nested = treeToMenuValues(tree, onPick, tree?.all ?? []);
  out.push(...nested);

  return out;
}

function pickedLoraValue(value) {
  if (typeof value === "string") return value;
  if (value && typeof value === "object") {
    // Folder items should never be treated as a picked LoRA.
    if (value.has_submenu || value.submenu) return null;
    const orig = value.rgthree_originalValue;
    if (typeof orig === "string") return orig;
    const c = value.content;
    if (typeof c === "string") {
      if (c.startsWith("📁 ")) return null;
      return c;
    }
  }
  return null;
}

function ensureState(node) {
  if (!node.properties) node.properties = {};
  if (!Array.isArray(node.properties.was_lora_rows)) {
    node.properties.was_lora_rows = [];
  }
  return node.properties.was_lora_rows;
}

function normalizeLoraOptions(options) {
  let out = options;
  if (Array.isArray(out) && out.length === 1 && Array.isArray(out[0])) {
    out = out[0];
  }
  if (!Array.isArray(out)) return ["None"];
  const filtered = out.filter((x) => typeof x === "string");
  if (!filtered.length) return ["None"];
  if (filtered[0] !== "None") {
    if (!filtered.includes("None")) filtered.unshift("None");
  }
  return filtered;
}

function setLoraOptions(node, options) {
  if (!node.properties) node.properties = {};
  node.properties.was_lora_options = normalizeLoraOptions(options);
}

function getLoraOptions(node) {
  const opts = node?.properties?.was_lora_options;
  return normalizeLoraOptions(opts);
}

function makeHiddenPayloadWidget(name, stateRef) {
  return {
    name,
    type: "custom",
    value: stateRef,
    computeSize() {
      return [0, 0];
    },
    draw() {
    },
    serializeValue() {
      return { ...stateRef };
    },
  };
}

function makeSectionHeaderWidget(name, title) {
  return {
    name,
    type: "custom",
    value: title,
    computeSize(width) {
      return [width ?? 0, 22];
    },
    draw(ctx, node, width, y, height) {
      try {
        const h = height ?? 22;
        ctx.save();
        ctx.globalAlpha = 1.0;
        ctx.fillStyle = "rgba(120, 170, 255, 0.15)";
        ctx.fillRect(0, y, width, h);
        ctx.fillStyle = "rgba(210, 230, 255, 0.95)";
        ctx.font = "12px sans-serif";
        ctx.textBaseline = "middle";
        ctx.fillText(title, 10, y + h / 2);
        ctx.restore();
      } catch (e) {
      }
    },
    serializeValue() {
      return title;
    },
  };
}

function clearWasWidgets(node) {
  if (!node.widgets) return;
  node.widgets = node.widgets.filter((w) => {
    const n = w?.name;
    const rowId = w?.options?.id;
    if (rowId === "row_add") return false;
    if (rowId === "row_remove_last") return false;
    if (rowId === "row_clear") return false;
    if (rowId === "row_refresh") return false;
    if (!n) return true;
    return !(n.startsWith("was_row_") || n.startsWith("lora_"));
  });
}

function syncRowsFromWidgets(node) {
  const rows = ensureState(node);
  const widgets = Array.isArray(node?.widgets) ? node.widgets : [];

  let maxIdxSeen = -1;

  for (const w of widgets) {
    const name = w?.name;
    if (typeof name !== "string") continue;

    const mCombo = /^lora_(\d+)$/.exec(name);
    if (mCombo) {
      const idx = Math.max(0, Number(mCombo[1]) - 1);
      maxIdxSeen = Math.max(maxIdxSeen, idx);
      while (rows.length <= idx) rows.push({ on: true, lora: null, weight: 1.0 });
      const v = w?.value;
      rows[idx].lora = typeof v === "string" && v !== "None" ? v : null;
      continue;
    }

    const mWeight = /^lora_(\d+)_weight$/.exec(name);
    if (mWeight) {
      const idx = Math.max(0, Number(mWeight[1]) - 1);
      maxIdxSeen = Math.max(maxIdxSeen, idx);
      while (rows.length <= idx) rows.push({ on: true, lora: null, weight: 1.0 });
      const n = Number(w?.value);
      rows[idx].weight = Number.isFinite(n) ? n : 1.0;
    }

    const mEnabled = /^lora_(\d+)_enabled$/.exec(name);
    if (mEnabled) {
      const idx = Math.max(0, Number(mEnabled[1]) - 1);
      maxIdxSeen = Math.max(maxIdxSeen, idx);
      while (rows.length <= idx) rows.push({ on: true, lora: null, weight: 1.0 });
      rows[idx].on = toBool(w?.value, true);
    }
  }

  const targetLen = Math.max(1, maxIdxSeen + 1);
  if (rows.length > targetLen) {
    rows.length = targetLen;
  }

  return rows;
}

function rebuildLoraRows(node, loraOptions, sync = true) {
  const rows = sync ? syncRowsFromWidgets(node) : ensureState(node);
  clearWasWidgets(node);

  node.addCustomWidget(makeSectionHeaderWidget("was_row_header", "Selected LoRA's"));

  const resolvedOptions = normalizeLoraOptions(loraOptions);

  for (const row of rows) {
    if (row && typeof row.lora === "string" && row.lora !== "None" && !resolvedOptions.includes(row.lora)) {
      resolvedOptions.push(row.lora);
    }
  }
  setLoraOptions(node, resolvedOptions);

  rows.forEach((row, idx) => {
    const rowIndex = idx + 1;
    const payloadName = `lora_payload_${rowIndex}`;
    node.addCustomWidget(makeHiddenPayloadWidget(payloadName, row));

    if (typeof row.lora !== "string") row.lora = row.lora == null ? null : String(row.lora);
    if (!Number.isFinite(row.weight)) row.weight = 1.0;
    if (typeof row.on !== "boolean") row.on = toBool(row.on, true);

    node.addWidget(
      "toggle",
      `lora_${rowIndex}_enabled`,
      !!row.on,
      (v) => {
        row.on = !!v;
      },
      { label: `lora_${rowIndex}_enabled` },
    );

    const comboWidget = node.addWidget(
      "combo",
      `lora_${rowIndex}`,
      row.lora ?? "None",
      (v) => {
        row.lora = v === "None" ? null : v;
      },
      { values: resolvedOptions, label: `lora_${rowIndex}` },
    );

    if (comboWidget && comboWidget.options) {
      comboWidget.options.values = resolvedOptions;
    }

    node.addWidget(
      "number",
      `lora_${rowIndex}_weight`,
      Number.isFinite(row.weight) ? row.weight : 1.0,
      (v) => {
        const n = Number(v);
        row.weight = Number.isFinite(n) ? n : 1.0;
      },
      { min: -10.0, max: 10.0, step: 0.01, precision: 3, label: `lora_${rowIndex}_strength` },
    );
  });

  node.addWidget(
    "button",
    "➕ Add LoRA",
    null,
    (...args) => {
      const event = args?.[1] ?? args?.[0];
      const opts = getLoraOptions(node);
      showLoraChooser(
        event,
        (value) => {
          const picked = pickedLoraValue(value);
          if (typeof picked === "string" && picked !== "None") {
            syncRowsFromWidgets(node);
            const curRows = ensureState(node);
            curRows.push({ on: true, lora: picked, weight: 1.0 });
            rebuildLoraRows(node, getLoraOptions(node), false);
            const computed = node.computeSize?.() ?? [node.size[0], node.size[1]];
            node.size[1] = Math.max(node.size[1], computed[1]);
            node.setDirtyCanvas(true, true);
          }
        },
        null,
        opts,
      );
    },
    { id: "row_add" },
  );

  node.addWidget(
    "button",
    "➖ Remove Last LoRA",
    null,
    () => {
      syncRowsFromWidgets(node);
      const curRows = ensureState(node);
      if (!curRows.length) return;
      curRows.pop();
      rebuildLoraRows(node, getLoraOptions(node), false);
      const computed = node.computeSize?.() ?? [node.size[0], node.size[1]];
      node.size[1] = Math.max(node.size[1], computed[1]);
      node.setDirtyCanvas(true, true);
    },
    { id: "row_remove_last" },
  );

  node.addWidget(
    "button",
    "🧹 Clear LoRAs",
    null,
    () => {
      syncRowsFromWidgets(node);
      const curRows = ensureState(node);
      if (!curRows.length) return;
      curRows.length = 0;
      curRows.push({ on: true, lora: null, weight: 1.0 });
      rebuildLoraRows(node, getLoraOptions(node), false);
      const computed = node.computeSize?.() ?? [node.size[0], node.size[1]];
      node.size[1] = Math.max(node.size[1], computed[1]);
      node.setDirtyCanvas(true, true);
    },
    { id: "row_clear" },
  );

  node.addWidget(
    "button",
    "♻️ Refresh LoRA List",
    null,
    async () => {
      await refreshNodeDefsAndUpdate(node);
    },
    { id: "row_refresh" },
  );

  try {
    const computed = node.computeSize?.() ?? null;
    if (computed && Array.isArray(computed) && computed.length >= 2) {
      node.size[1] = computed[1];
    }
  } catch (e) {
  }
}

app.registerExtension({
  name: EXT_NAME,
  getNodeMenuItems(node) {
    if (node?.comfyClass !== NODE_NAME) return [];

    const rows = ensureState(node);
    const opts = getLoraOptions(node);

    return [
      null,
      {
        content: "➕ Add LoRA",
        callback: (_value, _options, event) => {
          showLoraChooser(
            event,
            (picked) => {
              const lora = pickedLoraValue(picked);
              if (typeof lora === "string" && lora !== "None") {
                syncRowsFromWidgets(node);
                const curRows = ensureState(node);
                curRows.push({ on: true, lora: lora, weight: 1.0 });
                rebuildLoraRows(node, getLoraOptions(node), false);
                node.setDirtyCanvas(true, true);
              }
            },
            null,
            opts,
          );
        },
      },
      {
        content: "➖ Remove Last LoRA",
        disabled: rows.length === 0,
        callback: () => {
          syncRowsFromWidgets(node);
          const curRows = ensureState(node);
          curRows.pop();
          rebuildLoraRows(node, getLoraOptions(node), false);
          node.setDirtyCanvas(true, true);
        },
      },
      {
        content: "❌ Clear LoRAs",
        disabled: rows.length === 0,
        callback: () => {
          syncRowsFromWidgets(node);
          const curRows = ensureState(node);
          curRows.length = 0;
          curRows.push({ on: true, lora: null, weight: 1.0 });
          rebuildLoraRows(node, getLoraOptions(node), false);
          node.setDirtyCanvas(true, true);
        },
      },
      {
        content: "♻️ Refresh LoRA List",
        callback: async () => {
          await refreshNodeDefsAndUpdate(node);
        },
      },
    ];
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const backendCatalog = nodeData?.input?.hidden?.was_lora_catalog;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);

      if (!this.widgets) this.widgets = [];

      if (Array.isArray(backendCatalog) && backendCatalog.length) {
        setLoraOptions(this, backendCatalog);
      }

      const rows = ensureState(this);
      if (!rows.length) {
        rows.push({ on: true, lora: null, weight: 1.0 });
      }

      const opts = getLoraOptions(this);
      rebuildLoraRows(this, opts);
      const computed = this.computeSize?.() ?? [this.size[0], this.size[1]];
      this.size[1] = computed[1];
      this.setDirtyCanvas(true, true);

      try {
        if (!this.properties) this.properties = {};
        if (!this.properties._was_lora_catalog_refresh_pending) {
          this.properties._was_lora_catalog_refresh_pending = true;
          setTimeout(async () => {
            try {
              await refreshNodeDefsAndUpdate(this);
            } catch (e) {
            }
            try {
              this.properties._was_lora_catalog_refresh_pending = false;
            } catch (e) {
            }
          }, 0);
        }
      } catch (e) {
      }
    };

    const configure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      const widgetValues = info?.widgets_values || [];
      const rows = ensureState(this);
      rows.length = 0;

      for (const v of widgetValues) {
        if (v && typeof v === "object" && Object.prototype.hasOwnProperty.call(v, "lora")) {
          rows.push({
            on: toBool(v.on, true),
            lora: v.lora ?? null,
            weight: Number.isFinite(v.weight) ? v.weight : 1.0,
          });
        }
      }

      if (!rows.length) {
        rows.push({ on: true, lora: null, weight: 1.0 });
      }

      if (Array.isArray(backendCatalog) && backendCatalog.length) {
        setLoraOptions(this, backendCatalog);
      }

      const opts = getLoraOptions(this);
      rebuildLoraRows(this, opts);
      const computed = this.computeSize?.() ?? [this.size[0], this.size[1]];
      this.size[1] = computed[1];
      this.setDirtyCanvas(true, true);

      try {
        if (!this.properties) this.properties = {};
        if (!this.properties._was_lora_catalog_refresh_pending) {
          this.properties._was_lora_catalog_refresh_pending = true;
          setTimeout(async () => {
            try {
              await refreshNodeDefsAndUpdate(this);
            } catch (e) {
            }
            try {
              this.properties._was_lora_catalog_refresh_pending = false;
            } catch (e) {
            }
          }, 0);
        }
      } catch (e) {
      }

      configure?.apply(this, arguments);
    };

    const onSerialize = nodeType.prototype.onSerialize;
    nodeType.prototype.onSerialize = function (o) {
      try {
        const rows = ensureState(this);
        const safeRows = rows.map((r) => {
          const lora = typeof r?.lora === "string" ? r.lora : null;
          const weight = Number.isFinite(Number(r?.weight)) ? Number(r.weight) : 1.0;
          return { on: !!r?.on, lora, weight };
        });
        o.properties = o.properties || {};
        o.properties.was_lora_rows = safeRows;
      } catch (e) {
      }

      return onSerialize?.apply(this, arguments);
    };

    const refreshComboInNode = nodeType.prototype.refreshComboInNode;
    nodeType.prototype.refreshComboInNode = function (defs) {
      const fromBackend = defs?.input?.hidden?.was_lora_catalog;
      if (Array.isArray(fromBackend) && fromBackend.length) {
        setLoraOptions(this, fromBackend);
        rebuildLoraRows(this, fromBackend);
        this.setDirtyCanvas(true, true);
      }
      return refreshComboInNode?.apply(this, arguments);
    };
  },
});
