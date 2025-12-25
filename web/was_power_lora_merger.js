import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXT_NAME = "WAS_Extras.PowerLoraMergerUI";
const NODE_NAME = "WASPowerLoraMerger";

const WAS_OVERRIDDEN_SERVER_NODES = new Map();
let WAS_OVERRIDE_REGISTERED = false;

function registerWasNodeOverrideSystem() {
  if (WAS_OVERRIDE_REGISTERED) return;
  if (!globalThis?.LiteGraph?.registerNodeType) return;
  WAS_OVERRIDE_REGISTERED = true;

  const oldregisterNodeType = LiteGraph.registerNodeType;
  LiteGraph.registerNodeType = async function (nodeId, baseClass) {
    const clazz = WAS_OVERRIDDEN_SERVER_NODES.get(baseClass) || baseClass;
    return oldregisterNodeType.call(LiteGraph, nodeId, clazz);
  };
}

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

function rowToPayloadString(row) {
  const on = !!row?.on;
  const lora = typeof row?.lora === "string" ? row.lora : null;
  const weight = Number.isFinite(Number(row?.weight)) ? Number(row.weight) : 1.0;
  return JSON.stringify({ on, lora, weight });
}

function updatePayloadWidget(payloadWidget, row) {
  if (!payloadWidget) return;
  payloadWidget.value = rowToPayloadString(row);
}

function rowsToPayloadAllString(rows) {
  const arr = Array.isArray(rows) ? rows : [];
  const safe = arr.map((r) => {
    const on = !!r?.on;
    const lora = typeof r?.lora === "string" ? r.lora : null;
    const weight = Number.isFinite(Number(r?.weight)) ? Number(r.weight) : 1.0;
    return { on, lora, weight };
  });
  return JSON.stringify(safe);
}

function updateAllPayloadWidget(payloadWidget, rows) {
  if (!payloadWidget) return;
  payloadWidget.value = rowsToPayloadAllString(rows);
}

function findWidgetByName(node, name) {
  const widgets = Array.isArray(node?.widgets) ? node.widgets : [];
  for (const w of widgets) {
    if (w?.name === name) return w;
  }
  return null;
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

function removeWidgetSafe(node, index) {
  const widgets = Array.isArray(node?.widgets) ? node.widgets : null;
  const w = widgets?.[index];
  if (!w) return;
  try {
    node.removeWidget(index);
    return;
  } catch (e) {
  }
  try {
    node.removeWidget(w);
    return;
  } catch (e) {
  }
  try {
    widgets.splice(index, 1);
  } catch (e) {
  }
}

function clearWasWidgets(node) {
  if (!node.widgets) return;

  // Do not mutate node.widgets directly (e.g. filter/assign). LiteGraph tracks widget values
  // by index; direct mutation can desync widgets <-> widgets_values and break persistence.
  for (let i = node.widgets.length - 1; i >= 0; i--) {
    const w = node.widgets[i];
    const n = w?.name;
    const rowId = w?.options?.id;

    if (rowId === "row_add" || rowId === "row_remove_last" || rowId === "row_clear" || rowId === "row_refresh") {
      removeWidgetSafe(node, i);
      continue;
    }

    if (typeof n !== "string") continue;
    if (n === "lora_payload_all") continue;

    if (n.startsWith("was_row_") || n.startsWith("lora_")) {
      removeWidgetSafe(node, i);
    }
  }
}

function syncRowsFromWidgets(node) {
  const rows = ensureState(node);
  const widgets = Array.isArray(node?.widgets) ? node.widgets : [];

  let maxIdxSeen = -1;

  for (const w of widgets) {
    const name = w?.name;
    if (typeof name !== "string") continue;

    if (name === "lora_payload_all") {
      continue;
    }

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

  if (maxIdxSeen >= 0) {
    const targetLen = Math.max(1, maxIdxSeen + 1);
    if (rows.length > targetLen) {
      rows.length = targetLen;
    }
  }

  return rows;
}

function rebuildLoraRows(node, loraOptions, sync = true) {
  const rows = sync ? syncRowsFromWidgets(node) : ensureState(node);
  clearWasWidgets(node);

  node.addCustomWidget(makeSectionHeaderWidget("was_row_header", "Selected LoRA's"));

  // A single stable serialized widget that persists the entire row list.
  const payloadAllName = "lora_payload_all";
  let payloadAllWidget = findWidgetByName(node, payloadAllName);
  if (!payloadAllWidget) {
    payloadAllWidget = node.addWidget(
      "text",
      payloadAllName,
      rowsToPayloadAllString(rows),
      () => {
      },
      { multiline: true, label: payloadAllName },
    );
  }
  if (payloadAllWidget) {
    payloadAllWidget.computeSize = () => [0, 0];
    payloadAllWidget.draw = () => {
    };
    updateAllPayloadWidget(payloadAllWidget, rows);
  }

  const resolvedOptions = normalizeLoraOptions(loraOptions);

  for (const row of rows) {
    if (row && typeof row.lora === "string" && row.lora !== "None" && !resolvedOptions.includes(row.lora)) {
      resolvedOptions.push(row.lora);
    }
  }
  setLoraOptions(node, resolvedOptions);

  rows.forEach((row, idx) => {
    const rowIndex = idx + 1;

    if (typeof row.lora !== "string") row.lora = row.lora == null ? null : String(row.lora);
    if (!Number.isFinite(row.weight)) row.weight = 1.0;
    if (typeof row.on !== "boolean") row.on = toBool(row.on, true);

    node.addWidget(
      "toggle",
      `lora_${rowIndex}_enabled`,
      !!row.on,
      (v) => {
        row.on = !!v;
        updateAllPayloadWidget(payloadAllWidget, rows);
      },
      { label: `lora_${rowIndex}_enabled` },
    );

    const comboWidget = node.addWidget(
      "combo",
      `lora_${rowIndex}`,
      row.lora ?? "None",
      (v) => {
        row.lora = v === "None" ? null : v;
        updateAllPayloadWidget(payloadAllWidget, rows);
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
        updateAllPayloadWidget(payloadAllWidget, rows);
      },
      { min: -10.0, max: 10.0, step: 0.01, precision: 3, label: `lora_${rowIndex}_strength` },
    );
  });

  // Ensure payload-all matches any normalization performed above.
  updateAllPayloadWidget(payloadAllWidget, rows);

  node.addWidget(
    "button",
    "➕ Add LoRA",
    null,
    (...args) => {
      syncRowsFromWidgets(node);
      const curRows = ensureState(node);
      curRows.push({ on: true, lora: null, weight: 1.0 });
      rebuildLoraRows(node, getLoraOptions(node), false);
      const computed = node.computeSize?.() ?? [node.size[0], node.size[1]];
      node.size[1] = Math.max(node.size[1], computed[1]);
      node.setDirtyCanvas(true, true);
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
          syncRowsFromWidgets(node);
          const curRows = ensureState(node);
          curRows.push({ on: true, lora: null, weight: 1.0 });
          rebuildLoraRows(node, getLoraOptions(node), false);
          node.setDirtyCanvas(true, true);
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

    registerWasNodeOverrideSystem();

    const backendCatalog = nodeData?.input?.hidden?.was_lora_catalog;

    class WASPowerLoraMergerOverride extends nodeType {
      constructor(...args) {
        super(...args);
        this.serialize_widgets = true;
        try {
          this.title = nodeType.title ?? this.title;
        } catch (e) {
        }
      }

      onNodeCreated() {
        super.onNodeCreated?.();

        if (Array.isArray(backendCatalog) && backendCatalog.length) {
          setLoraOptions(this, backendCatalog);
        }

        const rows = ensureState(this);
        if (!rows.length) {
          rows.push({ on: true, lora: null, weight: 1.0 });
        }

        // Avoid double rebuild when loading from a workflow: ComfyUI will call configure(info)
        // shortly after creation, and configure will rebuild from serialized state.
        setTimeout(() => {
          try {
            if (this._was_plm_configured) return;
            rebuildLoraRows(this, getLoraOptions(this), false);
            const computed = this.computeSize?.() ?? [this.size[0], this.size[1]];
            this.size[1] = computed[1];
            this.setDirtyCanvas(true, true);
          } catch (e) {
          }
        }, 0);
      }

      configure(info) {
        this._was_plm_configured = true;
        super.configure?.(info);

        const prevRows = ensureState(this);
        const parsed = [];
        let restoredFromProps = false;

        const parseRows = (arr) => {
          if (!Array.isArray(arr)) return;
          for (const v of arr) {
            let obj = null;

            if (v && typeof v === "object") {
              obj = v;
            } else if (typeof v === "string") {
              const s = v.trim();
              if (s && (s.startsWith("{") || s.startsWith("["))) {
                try {
                  const parsedObj = JSON.parse(s);
                  if (parsedObj && typeof parsedObj === "object") obj = parsedObj;
                } catch (e) {
                }
              }
            }

            if (obj && typeof obj === "object" && Object.prototype.hasOwnProperty.call(obj, "lora")) {
              parsed.push({
                on: toBool(obj.on, true),
                lora: obj.lora ?? null,
                weight: Number.isFinite(Number(obj.weight)) ? Number(obj.weight) : 1.0,
              });
            }
          }
        };

        try {
          const propRows = info?.properties?.was_lora_rows;
          if (Array.isArray(propRows) && propRows.length) {
            parseRows(propRows);
            restoredFromProps = true;
          }
        } catch (e) {
        }

        // Only fall back to widgets_values if we didn't already restore from properties.
        if (!restoredFromProps && Array.isArray(info?.widgets_values)) {
          for (const v of info.widgets_values) {
            if (typeof v === "string") {
              const s = v.trim();
              if (s.startsWith("[")) {
                try {
                  const arr = JSON.parse(s);
                  parseRows(arr);
                } catch (e) {
                }
              }
            } else if (Array.isArray(v)) {
              parseRows(v);
            }
          }
        }

        if (!parsed.length) {
          parseRows(prevRows);
        }

        prevRows.length = 0;
        prevRows.push(...parsed);
        if (!prevRows.length) {
          prevRows.push({ on: true, lora: null, weight: 1.0 });
        }

        if (Array.isArray(backendCatalog) && backendCatalog.length) {
          setLoraOptions(this, backendCatalog);
        }

        rebuildLoraRows(this, getLoraOptions(this), false);
        const computed = this.computeSize?.() ?? [this.size[0], this.size[1]];
        this.size[1] = computed[1];
        this.setDirtyCanvas(true, true);
      }

      refreshComboInNode(defs) {
        const fromBackend = defs?.input?.hidden?.was_lora_catalog;
        if (Array.isArray(fromBackend) && fromBackend.length) {
          setLoraOptions(this, fromBackend);
          rebuildLoraRows(this, fromBackend, false);
          this.setDirtyCanvas(true, true);
        }
        return super.refreshComboInNode?.(defs);
      }
    }

    WASPowerLoraMergerOverride.title = nodeType.title;
    WASPowerLoraMergerOverride.category = nodeType.category;
    WASPowerLoraMergerOverride.type = nodeType.type;

    WAS_OVERRIDDEN_SERVER_NODES.set(nodeType, WASPowerLoraMergerOverride);
  },
});
