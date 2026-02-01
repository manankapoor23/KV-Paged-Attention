/**
 * Visualization logic for paged KV cache
 */

class KVVisualizer {
    constructor() {
        this.data = null;
        this.currentStep = 0;
        this.maxSteps = 0;
        this.prefixTokenIndices = [];

        this._cacheCtxStep = null;
        this._cacheCtx = null;
    }

    setData(data, prefixTokenIndices = []) {
        this.data = data;
        this.currentStep = 0;
        this.maxSteps = data.events.length;
        this.prefixTokenIndices = Array.isArray(prefixTokenIndices) ? prefixTokenIndices : [];

        this._cacheCtxStep = null;
        this._cacheCtx = null;
    }

    hashToken(text) {
        // FNV-1a 32-bit
        let hash = 0x811c9dc5;
        for (let i = 0; i < text.length; i++) {
            hash ^= text.charCodeAt(i);
            hash = Math.imul(hash, 0x01000193);
        }
        // Unsigned + compact hex
        return (hash >>> 0).toString(16).padStart(8, "0");
    }

    makeTokenKey(tokenText, tokenIdx) {
        const safeText = typeof tokenText === "string" ? tokenText : "";
        return `${this.hashToken(safeText)}:${tokenIdx}`;
    }

    computeCacheContext(step) {
        if (this._cacheCtxStep === step && this._cacheCtx) return this._cacheCtx;

        const empty = {
            requestStartIndex: 0,
            priorCache: new Map(),
            currentWrites: new Set(),
            currentTokenIdx: null,
            cacheHit: false,
            cacheHitTokenIdx: null,
            cacheHitPageId: null,
        };

        if (!this.data) {
            this._cacheCtxStep = step;
            this._cacheCtx = empty;
            return empty;
        }

        const events = this.getEventsUpToStep(step);
        const tokens = Array.isArray(this.data.tokens) ? this.data.tokens : [];

        // Find the most recent request_start (treat everything before that as the global cache).
        let requestStartIndex = 0;
        for (let i = events.length - 1; i >= 0; i--) {
            if (events[i].event_type === "request_start") {
                requestStartIndex = i;
                break;
            }
        }

        const priorCache = new Map();
        for (let i = 0; i < requestStartIndex; i++) {
            const e = events[i];
            if (e.event_type !== "kv_write") continue;
            if (!this.isLayer0(e.details)) continue;
            const tokenIdx = e.details?.token_idx;
            if (typeof tokenIdx !== "number") continue;
            const tokenText = tokens[tokenIdx] ?? "";
            const key = this.makeTokenKey(tokenText, tokenIdx);
            priorCache.set(key, { pageId: e.details.page_id, slot: e.details.slot });
        }

        const currentWrites = new Set();
        for (let i = requestStartIndex; i < events.length; i++) {
            const e = events[i];
            if (e.event_type !== "kv_write") continue;
            if (!this.isLayer0(e.details)) continue;
            const tokenIdx = e.details?.token_idx;
            if (typeof tokenIdx !== "number") continue;
            const tokenText = tokens[tokenIdx] ?? "";
            const key = this.makeTokenKey(tokenText, tokenIdx);
            currentWrites.add(key);
        }

        const currentTokenIdx = this.getCurrentTokenIdx(events);

        // Cache-hit moment: only fire on the exact token_step event.
        const currentEvent = this.data?.events?.[step] || null;
        const tokenStepNow =
            currentEvent?.event_type === "token_step" && typeof currentEvent.details?.token_idx === "number";
        const cacheHitTokenIdx = tokenStepNow ? currentEvent.details.token_idx : null;

        let cacheHit = false;
        let cacheHitPageId = null;
        if (cacheHitTokenIdx !== null && cacheHitTokenIdx >= 0 && cacheHitTokenIdx < tokens.length) {
            const key = this.makeTokenKey(tokens[cacheHitTokenIdx] ?? "", cacheHitTokenIdx);
            if (priorCache.has(key) && !currentWrites.has(key)) {
                cacheHit = true;
                cacheHitPageId = priorCache.get(key).pageId;
            }
        }

        const ctx = {
            requestStartIndex,
            priorCache,
            currentWrites,
            currentTokenIdx,
            cacheHit,
            cacheHitTokenIdx,
            cacheHitPageId,
        };

        this._cacheCtxStep = step;
        this._cacheCtx = ctx;
        return ctx;
    }

    getEventsUpToStep(step) {
        if (!this.data) return [];
        return this.data.events.slice(0, step + 1);
    }

    getCurrentTokenIdx(events) {
        for (let i = events.length - 1; i >= 0; i--) {
            const e = events[i];
            if (e.event_type === "token_step" && typeof e.details?.token_idx === "number") {
                return e.details.token_idx;
            }
        }
        return null;
    }

    getCurrentRequestId(events) {
        for (let i = events.length - 1; i >= 0; i--) {
            const e = events[i];
            if (e.event_type === "request_start" && typeof e.details?.request_id === "number") {
                return e.details.request_id;
            }
        }
        return 0;
    }

    isLayer0(details) {
        const layer = details?.layer ?? details?.layer_idx;
        if (layer === undefined || layer === null) return true;
        return layer === 0;
    }

    renderPages(step) {
        const grid = document.getElementById("pages-grid");
        grid.innerHTML = "";

        if (!this.data) return;

        const events = this.getEventsUpToStep(step);
        const cacheCtx = this.computeCacheContext(step);
        const pageMap = new Map();

        const currentEvent = this.data?.events?.[step] || null;
        const highlightWrite =
            currentEvent?.event_type === "kv_write" && this.isLayer0(currentEvent.details)
                ? { pageId: currentEvent.details.page_id, slot: currentEvent.details.slot }
                : null;

        // Find the most recent COW event near this step to highlight the moment.
        let cowActive = null;
        for (let i = events.length - 1; i >= 0; i--) {
            const e = events[i];
            if (e.event_type === "copy_on_write") {
                cowActive = {
                    sourceId: e.details.source_page_id,
                    newId: e.details.new_page_id,
                    step: i,
                };
                break;
            }
        }

        // Track page states
        events.forEach((event) => {
            if (event.event_type === "page_fault") {
                const pageId = event.details.page_id;
                if (!pageMap.has(pageId)) {
                    pageMap.set(pageId, {
                        id: pageId,
                        slots: this.data.model.page_size || 16,
                        usedSlots: 0,
                        refCount: 1,
                        isFreed: false,
                    });
                }
            } else if (event.event_type === "kv_write") {
                if (!this.isLayer0(event.details)) return;
                const pageId = event.details.page_id;
                if (pageMap.has(pageId)) {
                    pageMap.get(pageId).usedSlots = Math.max(
                        pageMap.get(pageId).usedSlots,
                        event.details.slot + 1
                    );
                }
            } else if (event.event_type === "copy_on_write") {
                const sourceId = event.details.source_page_id;
                const newId = event.details.new_page_id;
                const slots = this.data.model.page_size || 16;

                if (!pageMap.has(sourceId)) {
                    pageMap.set(sourceId, {
                        id: sourceId,
                        slots,
                        usedSlots: 0,
                        refCount: 1,
                        isFreed: false,
                    });
                }

                const source = pageMap.get(sourceId);
                source.refCount = 1;

                if (!pageMap.has(newId)) {
                    pageMap.set(newId, {
                        id: newId,
                        slots,
                        usedSlots: source.usedSlots,
                        refCount: 1,
                        isFreed: false,
                        cowFrom: sourceId,
                    });
                }
            } else if (event.event_type === "page_freed") {
                const pageId = event.details.page_id;
                if (pageMap.has(pageId)) {
                    pageMap.get(pageId).isFreed = true;
                }
            } else if (event.event_type === "prefix_reuse") {
                // Attempt to reflect sharing in the visualization when the trace provides page ids.
                const candidates = [
                    event.details?.page_ids,
                    event.details?.pages,
                    event.details?.reused_pages,
                    event.details?.prefix_pages,
                ];
                const pageIds = candidates.find((x) => Array.isArray(x));
                if (Array.isArray(pageIds)) {
                    pageIds.forEach((pid) => {
                        if (typeof pid !== "number") return;
                        if (!pageMap.has(pid)) {
                            pageMap.set(pid, {
                                id: pid,
                                slots: this.data.model.page_size || 16,
                                usedSlots: 0,
                                refCount: 1,
                                isFreed: false,
                            });
                        }
                        pageMap.get(pid).refCount = Math.max(2, pageMap.get(pid).refCount + 1);
                    });
                }
            }
        });

        // Render pages
        const pages = Array.from(pageMap.values()).sort((a, b) => a.id - b.id);
        pages.forEach((page) => {
            const pageEl = document.createElement("div");
            pageEl.className = "page";

            if (page.isFreed) {
                pageEl.classList.add("freed");
            } else if (page.refCount > 1) {
                pageEl.classList.add("shared");
            } else {
                pageEl.classList.add("allocated");
            }

            if (cowActive && step - cowActive.step <= 2) {
                if (page.id === cowActive.sourceId) pageEl.classList.add("cow-source");
                if (page.id === cowActive.newId) pageEl.classList.add("cow-target");
            }

            if (cacheCtx.cacheHit && cacheCtx.cacheHitPageId === page.id) {
                pageEl.classList.add("reuse-glow");
            }

            // Page header
            const header = document.createElement("div");
            header.className = "page-header";
            const cowFrom =
                page.cowFrom !== undefined
                    ? ` <span class="page-subtitle">(copied from Page ${page.cowFrom})</span>`
                    : "";
            const hitBadge =
                cacheCtx.cacheHit && cacheCtx.cacheHitPageId === page.id
                    ? `<span class="badge cache-hit">cache hit</span>`
                    : "";
            header.innerHTML = `
                <span class="page-title"><strong>Page ${page.id}</strong>${cowFrom}</span>
                <span class="page-badges">${hitBadge}</span>
            `;
            pageEl.appendChild(header);

            // Slots
            const slotsContainer = document.createElement("div");
            slotsContainer.className = "page-slots";

            for (let i = 0; i < page.slots; i++) {
                const slot = document.createElement("div");
                slot.className = "slot";
                if (i < page.usedSlots) {
                    slot.classList.add("filled");
                }

                if (highlightWrite && page.id === highlightWrite.pageId && i === highlightWrite.slot) {
                    slot.classList.add("just-written");
                }

                slot.title = `Slot ${i}`;
                slotsContainer.appendChild(slot);
            }

            pageEl.appendChild(slotsContainer);

            // Info
            const info = document.createElement("div");
            info.className = "page-info";
            info.textContent = `${page.usedSlots}/${page.slots} • ref=${page.refCount}`;
            pageEl.appendChild(info);

            grid.appendChild(pageEl);
        });
    }

    renderEvents(step) {
        const list = document.getElementById("events-list");
        list.innerHTML = "";

        if (!this.data) return;

        const events = this.getEventsUpToStep(step);

        // Show last 25 events
        const recentEvents = events.slice(Math.max(0, events.length - 25));

        recentEvents.forEach((event, idx) => {
            const eventEl = document.createElement("div");
            eventEl.className = `event event-${event.event_type}`;

            const details = JSON.stringify(event.details);
            const id = event.event_id ?? idx;

            eventEl.innerHTML = `
                <span class="event-type ${event.event_type}">${event.event_type}</span>
                <span class="event-details">#${id} ${details}</span>
            `;

            list.appendChild(eventEl);
        });
    }

    renderModelInfo() {
        const text = document.getElementById("model-info-text");
        if (!text) return;
        if (!this.data) return;

        const m = this.data.model;
        text.innerHTML = `
            <strong>Model:</strong> DistilGPT2<br>
            <strong>Layers:</strong> ${m.layers}<br>
            <strong>Heads:</strong> ${m.heads}<br>
            <strong>Hidden Size:</strong> ${m.hidden_size}<br>
            <strong>Head Dim:</strong> ${m.head_dim}
        `;
    }

    renderSummary() {
        const text = document.getElementById("summary-text");
        if (!text) return;
        if (!this.data) return;

        const s = this.data.summary;
        text.innerHTML = `
            <strong>Total Events:</strong> ${s.total_events}<br>
            <strong>Total Requests:</strong> ${s.num_requests}<br>
            <strong>Event Breakdown:</strong><br>
            <div style="margin-left: 20px;">
                ${Object.entries(s.event_counts)
                    .map(([k, v]) => `${k}: ${v}`)
                    .join("<br>")}
            </div>
        `;
    }

    renderNarration(step) {
        const narrationDiv = document.getElementById("narration-text");
        if (!this.data || !this.data.events[step]) {
            narrationDiv.innerHTML = "<p>No events yet.</p>";
            return;
        }

        const event = this.data.events[step];
        const narration = event.narration || "Event occurred.";
        
        // Render narration as HTML with proper line breaks
        const lines = narration.split("\n");

        const cacheCtx = this.computeCacheContext(step);
        let extra = "";
        
        // Add ASCII graphics based on event type
        if (event.event_type === "token_step") {
            const tokenIdx = event.details?.token_idx;
            const tokens = this.data.tokens || [];
            const token = tokens[tokenIdx] || "";
            
            if (cacheCtx.cacheHit && cacheCtx.cacheHitPageId !== null) {
                // Cache hit ASCII celebration
                extra += `
<pre class="ascii-graphic">
  ✓ CACHE HIT!
  ┌─────────────────────┐
  │ Token "${token}"      │ ⟶ Page ${cacheCtx.cacheHitPageId}
  └─────────────────────┘
  No recompute needed!
</pre>`;
            } else {
                // New computation
                extra += `
<pre class="ascii-graphic">
  📝 NEW TOKEN: "${token}"
  ┌──────────────────────────────┐
  │ Attention computation        │
  │ (reading all cached KV)      │
  └──────────────────────────────┘
</pre>`;
            }
        } else if (event.event_type === "kv_write") {
            const { page_id, slot, token_idx } = event.details;
            extra += `
<pre class="ascii-graphic">
  [KV Write] Token ${token_idx} → Page ${page_id}:${slot}
  ┌────────────────┐
  │ ■■■  ░░░░  │  ← Slot ${slot}
  │ ■■■  ░░░░  │
  │ ■■■  ░░░░  │
  └────────────────┘
  Page ${page_id}
</pre>`;
        } else if (event.event_type === "page_fault") {
            const { page_id } = event.details;
            extra += `
<pre class="ascii-graphic">
  ⚠ PAGE FAULT
  ┌─────────────────────────┐
  │ Allocate Page ${page_id}       │
  │ [████████████████]      │
  │ 16 slots available      │
  └─────────────────────────┘
</pre>`;
        } else if (event.event_type === "copy_on_write") {
            const { source_page_id, new_page_id } = event.details;
            extra += `
<pre class="ascii-graphic">
  📋 COPY-ON-WRITE
  Page ${source_page_id}            Page ${new_page_id}
  ┌─────────┐    ┌─────────┐
  │ ■■■■■■  │ ⟶  │ ■■■■■■  │
  │ ■■■■■■  │    │ ■■■■■■  │
  │ ░░░░░░  │    │ ░░░░░░  │
  └─────────┘    └─────────┘
  (shared→private transition)
</pre>`;
        } else if (event.event_type === "prefix_reuse") {
            extra += `
<pre class="ascii-graphic">
  🔄 PREFIX REUSE
  ┌─────────────────────────┐
  │ Request 1: "you are"    │
  │ Request 2: "you are ... │
  │            ─────────    │ ← reused!
  │            (no compute) │
  └─────────────────────────┘
</pre>`;
        }
        
        const extraMessage = cacheCtx.cacheHit && event.event_type === "token_step"
            ? `<p><span class="badge cache-hit">cache hit</span> KV reused from Page ${cacheCtx.cacheHitPageId} (no recompute)</p>`
            : "";

        narrationDiv.innerHTML =
            lines.map((line) => `<p>${line}</p>`).join("") +
            extra +
            extraMessage;
    }

    renderTokens(step) {
        const container = document.getElementById("tokens-list");
        if (!container) return;
        container.innerHTML = "";

        if (!this.data) return;

        const events = this.getEventsUpToStep(step);
        const currentTokenIdx = this.getCurrentTokenIdx(events);
        const cacheCtx = this.computeCacheContext(step);

        const tokens = Array.isArray(this.data.tokens) ? this.data.tokens : [];
        const tokenCount = Math.max(tokens.length, currentTokenIdx !== null ? currentTokenIdx + 1 : 0);

        const list = document.createElement("div");
        list.className = "tokens-list";

        for (let i = 0; i < tokenCount; i++) {
            const row = document.createElement("div");
            row.className = "token-row";

            if (currentTokenIdx !== null) {
                if (i < currentTokenIdx) row.classList.add("past");
                if (i === currentTokenIdx) row.classList.add("current");
            }

            // Check if this token is part of the prefix from a previous simulation
            const isPrefixReused = this.prefixTokenIndices.includes(i);
            if (isPrefixReused) {
                row.classList.add("prefix-reused");
            }

            const tokenText = i < tokens.length ? tokens[i] : "(generated token)";

            const key = i < tokens.length ? this.makeTokenKey(tokens[i] ?? "", i) : null;
            const isCached = key ? cacheCtx.priorCache.has(key) : false;
            const isHit = cacheCtx.cacheHit && cacheCtx.cacheHitTokenIdx === i;

            if (isCached && !isPrefixReused) row.classList.add("cached");

            const badge = isHit ? `<span class="badge cache-hit">cache hit</span>` : "";

            row.innerHTML = `
                <div class="token-idx">${i}</div>
                <div class="token-text"><span class="token-str">${tokenText}</span>${badge}</div>
            `;
            list.appendChild(row);
        }

        container.appendChild(list);
    }

    renderPageTable(step) {
        const container = document.getElementById("page-table");
        if (!container) return;
        container.innerHTML = "";

        if (!this.data) return;

        const events = this.getEventsUpToStep(step);
        const currentTokenIdx = this.getCurrentTokenIdx(events);
        const cacheCtx = this.computeCacheContext(step);
        const tokens = Array.isArray(this.data.tokens) ? this.data.tokens : [];
        const tokenCount = Math.max(tokens.length, currentTokenIdx !== null ? currentTokenIdx + 1 : 0);

        const mapping = new Map();
        events.forEach((e) => {
            if (e.event_type !== "kv_write") return;
            if (!this.isLayer0(e.details)) return;
            const tokenIdx = e.details?.token_idx;
            if (typeof tokenIdx !== "number") return;
            mapping.set(tokenIdx, { pageId: e.details.page_id, slot: e.details.slot });
        });

        const currentEvent = this.data.events?.[step] || null;
        if (currentEvent?.event_type === "page_fault") {
            const fault = document.createElement("div");
            fault.className = "pt-row fault";
            fault.innerHTML = `
                <div class="pt-left">fault</div>
                <div class="pt-right">allocated Page ${currentEvent.details.page_id}</div>
            `;
            container.appendChild(fault);
        }

        const table = document.createElement("div");
        table.className = "page-table";

        for (let i = 0; i < tokenCount; i++) {
            const row = document.createElement("div");
            row.className = "pt-row";
            if (currentTokenIdx !== null && i === currentTokenIdx) row.classList.add("current");

            const m = mapping.get(i);
            const isPrefixReused = this.prefixTokenIndices.includes(i);
            if (isPrefixReused) row.classList.add("prefix-reused");

            const tokenText = i < tokens.length ? tokens[i] : "";
            const key = i < tokens.length ? this.makeTokenKey(tokenText, i) : null;
            const isCached = key ? cacheCtx.priorCache.has(key) : false;
            const isHit = cacheCtx.cacheHit && cacheCtx.cacheHitTokenIdx === i;
            const badge = isHit ? `<span class="badge cache-hit">cache hit</span>` : "";

            row.innerHTML = `
                <div class="pt-left">Token ${i}</div>
                <div class="pt-right"><span class="pt-mapping">${m ? `Page ${m.pageId}, Slot ${m.slot}` : "—"}</span>${badge}</div>
            `;
            table.appendChild(row);
        }

        container.appendChild(table);
    }

    renderAttentionProxy(step) {
        const container = document.getElementById("attention-bars");
        if (!container) return;
        container.innerHTML = "";
        if (!this.data) return;

        const events = this.getEventsUpToStep(step);
        const pageMap = new Map();

        events.forEach((event) => {
            if (event.event_type === "page_fault") {
                const pageId = event.details.page_id;
                if (!pageMap.has(pageId)) {
                    pageMap.set(pageId, {
                        id: pageId,
                        slots: this.data.model.page_size || 16,
                        usedSlots: 0,
                    });
                }
            } else if (event.event_type === "kv_write") {
                if (!this.isLayer0(event.details)) return;
                const pageId = event.details.page_id;
                if (pageMap.has(pageId)) {
                    pageMap.get(pageId).usedSlots = Math.max(pageMap.get(pageId).usedSlots, event.details.slot + 1);
                }
            }
        });

        const pages = Array.from(pageMap.values()).sort((a, b) => a.id - b.id);
        pages.forEach((p) => {
            const ratio = p.slots > 0 ? p.usedSlots / p.slots : 0;
            const row = document.createElement("div");
            row.className = "att-row";
            row.innerHTML = `
                <div class="att-label">Page ${p.id}</div>
                <div class="att-bar"><span style="width:${Math.round(ratio * 100)}%"></span></div>
            `;
            container.appendChild(row);
        });
    }

    render(step) {
        this.currentStep = Math.min(step, this.maxSteps - 1);
        this.renderPages(this.currentStep);
        this.renderEvents(this.currentStep);
        this.renderNarration(this.currentStep);
        this.renderTokens(this.currentStep);
        this.renderPageTable(this.currentStep);
        this.renderAttentionProxy(this.currentStep);
        this.renderModelInfo();
        this.renderSummary();
    }
}

const visualizer = new KVVisualizer();
