import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";


const NODE_NAME = "DvD_IOPaint_Interactive_Eraser";
const MIN_NODE_WIDTH = 520;
const EDITOR_MAX_HEIGHT = 560;
const COMPARISON_MAX_HEIGHT = 300;
const BRUSH_MIN_SIZE = 1;
const BRUSH_MAX_SIZE = 512;
const editors = new Map();


function addStylesheet() {
    const id = "dvd-iopaint-styles";
    if (document.getElementById(id)) return;
    const link = document.createElement("link");
    link.id = id;
    link.rel = "stylesheet";
    link.href = new URL("./dvd_iopaint.css", import.meta.url).href;
    document.head.appendChild(link);
}


function hideWidget(widget) {
    widget.origType = widget.type;
    widget.origComputeSize = widget.computeSize;
    widget.computeSize = () => [0, -4];
    widget.type = "converted-widget:dvd_iopaint";
}


function iconButton(icon, title, callback) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "dvd-iopaint-icon-button";
    button.title = title;
    button.setAttribute("aria-label", title);
    const iconElement = document.createElement("i");
    iconElement.className = `pi ${icon}`;
    button.appendChild(iconElement);
    button.addEventListener("click", callback);
    return button;
}


function inputImageUrl(filename) {
    let normalized = String(filename || "").replace(/\\/g, "/");
    normalized = normalized.replace(/\s*\[input\]\s*$/, "");
    const slash = normalized.lastIndexOf("/");
    const name = slash >= 0 ? normalized.slice(slash + 1) : normalized;
    const subfolder = slash >= 0 ? normalized.slice(0, slash) : "";
    const params = new URLSearchParams({
        filename: name,
        type: "input",
        subfolder,
        t: String(Date.now()),
    });
    return api.apiURL(`/view?${params.toString()}`);
}


function outputImageUrl(image) {
    const params = new URLSearchParams({
        filename: image.filename,
        type: image.type,
        subfolder: image.subfolder || "",
        t: String(Date.now()),
    });
    return api.apiURL(`/view?${params.toString()}`);
}


async function loadOutputImage(image) {
    const response = await fetch(outputImageUrl(image));
    if (!response.ok) throw new Error(`Result fetch failed: ${response.status}`);
    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);
    try {
        return await loadImage(objectUrl);
    } finally {
        URL.revokeObjectURL(objectUrl);
    }
}


async function uploadImage(blob, filename, subfolder) {
    const form = new FormData();
    form.append("image", blob, filename);
    form.append("type", "input");
    form.append("subfolder", subfolder);
    form.append("overwrite", "true");
    const response = await api.fetchApi("/upload/image", {
        method: "POST",
        body: form,
    });
    if (!response.ok) {
        throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
    }
    const uploaded = await response.json();
    return uploaded.subfolder ? `${uploaded.subfolder}/${uploaded.name}` : uploaded.name;
}


function canvasBlob(canvas) {
    return new Promise((resolve, reject) => {
        canvas.toBlob((blob) => {
            if (blob) resolve(blob);
            else reject(new Error("Could not encode the mask canvas."));
        }, "image/png");
    });
}


function loadImage(url) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error(`Could not load image: ${url}`));
        image.src = url;
    });
}


function isImageFile(file) {
    return Boolean(
        file && (
            file.type?.startsWith("image/") ||
            /\.(avif|bmp|gif|jpe?g|png|webp)$/i.test(file.name || "")
        )
    );
}


function executionIds(detail) {
    const values = (detail && typeof detail === "object")
        ? [
            detail.node,
            detail.display_node,
            detail.node_id,
            detail.display_node_id,
        ]
        : [detail];
    const ids = new Set();
    for (const value of values) {
        if (value === undefined || value === null || value === "") continue;
        const id = String(value);
        ids.add(id);
        if (id.includes(":")) ids.add(id.split(":").at(-1));
    }
    return [...ids];
}


function findEditor(detail) {
    const ids = executionIds(detail);
    for (const id of ids) {
        const editor = editors.get(id);
        if (editor) return editor;
    }
    for (const id of ids) {
        const numericId = Number(id);
        const node = app.graph?.getNodeById?.(
            Number.isNaN(numericId) ? id : numericId,
        ) || app.graph?.getNodeById?.(id);
        const editor = node?.dvdIOPaintEditor || node?.dvdSAMEditor;
        if (editor) {
            editor.register();
            return editor;
        }
    }
    const promptId = detail?.prompt_id || detail?.promptId;
    if (promptId !== undefined && promptId !== null && promptId !== "") {
        const normalizedPromptId = String(promptId);
        for (const editor of uniqueEditors()) {
            if (editor.promptId && String(editor.promptId) === normalizedPromptId) {
                return editor;
            }
        }
    }
    return null;
}


function uniqueEditors() {
    return new Set(editors.values());
}


function finishExecutionEditors(detail, message) {
    const editor = findEditor(detail);
    if (editor) {
        editor.finishWithError(message);
        return;
    }
    // Older ComfyUI frontends did not always include a node id on lifecycle
    // errors.  Clear a lone active editor in that case so it cannot remain in
    // the Running state forever; when several are active, only editors that
    // have received an executing event are unblocked.
    const active = [...uniqueEditors()].filter((candidate) => candidate.busy);
    if (active.length === 1) {
        active[0].finishWithError(message);
        return;
    }
    for (const candidate of active) {
        if (candidate.executionSeen) candidate.finishWithError(message);
    }
}


class DvDIOPaintEditor {
    constructor(node) {
        this.node = node;
        this.imageWidget = node.widgets.find((widget) => widget.name === "image");
        this.maskWidget = node.widgets.find((widget) => widget.name === "mask");
        this.brushWidget = node.widgets.find((widget) => widget.name === "brush_size");
        this.autoRunWidget = node.widgets.find((widget) => widget.name === "auto_run");
        this.history = [];
        this.processHistory = [];
        this.mode = "paint";
        this.drawing = false;
        this.busy = false;
        this.loadGeneration = 0;
        this.layoutQueued = false;
        this.layoutInProgress = false;
        this.widgetHeight = 300;
        this.hasComparison = false;
        this.lastBrushPointer = null;
        this.executionSeen = false;
        this.resultProcessing = false;
        this.resultVersion = 0;
        this.promptId = null;

        hideWidget(this.maskWidget);
        this.element = this.buildElement();
        this.domWidget = node.addDOMWidget(
            "dvd_iopaint_canvas",
            "dvd_iopaint_canvas",
            this.element,
            {
                serialize: false,
                hideOnZoom: false,
                margin: 0,
                getMinHeight: () => this.widgetHeight,
                getMaxHeight: () => this.widgetHeight,
                getHeight: () => this.widgetHeight,
            },
        );

        const originalImageCallback = this.imageWidget.callback;
        this.imageWidget.callback = (...args) => {
            const result = originalImageCallback?.apply(this.imageWidget, args);
            this.clearProcessHistory();
            this.clearComparison();
            this.loadInput(this.imageWidget.value, true);
            return result;
        };

        const originalBrushCallback = this.brushWidget.callback;
        this.brushWidget.callback = (...args) => {
            const result = originalBrushCallback?.apply(this.brushWidget, args);
            this.updateBrushCursor();
            return result;
        };

        this.bindPointerEvents();
        this.bindDropEvents();
        this.restore();
    }

    buildElement() {
        const root = document.createElement("div");
        root.className = "dvd-iopaint-editor";

        const toolbar = document.createElement("div");
        toolbar.className = "dvd-iopaint-toolbar";

        this.paintButton = iconButton("pi-pencil", "Paint removal mask", () => this.setMode("paint"));
        this.eraseButton = iconButton("pi-eraser", "Erase removal mask", () => this.setMode("erase"));
        this.undoButton = iconButton("pi-undo", "Undo mask stroke", () => this.undo());
        this.processUndoButton = iconButton(
            "pi-history",
            "Undo last completed removal",
            () => this.undoProcessedEdit(),
        );
        this.clearButton = iconButton("pi-trash", "Clear mask", () => this.clearMask(true));
        this.paintButton.classList.add("active");

        this.status = document.createElement("span");
        this.status.className = "dvd-iopaint-status";
        this.status.textContent = "Choose an image";

        toolbar.append(
            this.paintButton,
            this.eraseButton,
            this.undoButton,
            this.processUndoButton,
            this.clearButton,
            this.status,
        );

        this.stage = document.createElement("div");
        this.stage.className = "dvd-iopaint-stage";
        this.stage.style.setProperty("--dvd-aspect", "1");

        this.baseCanvas = document.createElement("canvas");
        this.baseCanvas.className = "dvd-iopaint-base";
        this.maskCanvas = document.createElement("canvas");
        this.maskCanvas.className = "dvd-iopaint-mask";
        this.brushCursor = document.createElement("div");
        this.brushCursor.className = "dvd-iopaint-brush-cursor";
        this.stage.append(this.baseCanvas, this.maskCanvas, this.brushCursor);

        this.comparison = document.createElement("div");
        this.comparison.className = "dvd-iopaint-comparison";
        this.comparison.style.setProperty("--dvd-split", "50%");
        this.compareBeforeCanvas = document.createElement("canvas");
        this.compareBeforeCanvas.className = "dvd-iopaint-comparison-before";
        this.compareAfterLayer = document.createElement("div");
        this.compareAfterLayer.className = "dvd-iopaint-comparison-after";
        this.compareAfterCanvas = document.createElement("canvas");
        this.compareAfterLayer.appendChild(this.compareAfterCanvas);
        this.compareDivider = document.createElement("div");
        this.compareDivider.className = "dvd-iopaint-comparison-divider";
        this.comparison.append(
            this.compareBeforeCanvas,
            this.compareAfterLayer,
            this.compareDivider,
        );

        root.append(toolbar, this.stage, this.comparison);
        return root;
    }

    bindPointerEvents() {
        this.maskCanvas.addEventListener("contextmenu", (event) => event.preventDefault());
        this.maskCanvas.addEventListener("pointerdown", (event) => this.pointerDown(event));
        this.maskCanvas.addEventListener("pointerenter", (event) => {
            this.updateBrushCursor(event);
        });
        this.maskCanvas.addEventListener("pointermove", (event) => {
            this.updateBrushCursor(event);
            this.pointerMove(event);
        });
        this.maskCanvas.addEventListener("pointerleave", () => {
            this.brushCursor.classList.remove("visible");
        });
        this.maskCanvas.addEventListener("pointerup", (event) => this.pointerUp(event));
        this.maskCanvas.addEventListener("pointercancel", (event) => this.pointerUp(event, false));
        this.element.addEventListener("wheel", (event) => this.handleWheel(event), {
            passive: false,
        });
        this.comparison.addEventListener("pointerenter", (event) => this.moveComparison(event));
        this.comparison.addEventListener("pointermove", (event) => this.moveComparison(event));
        this.comparison.addEventListener("pointerdown", (event) => {
            event.preventDefault();
            this.comparison.classList.add("dragging");
            this.comparison.setPointerCapture(event.pointerId);
            this.moveComparison(event);
        });
        this.comparison.addEventListener("pointerup", (event) => {
            this.comparison.classList.remove("dragging");
            if (this.comparison.hasPointerCapture(event.pointerId)) {
                this.comparison.releasePointerCapture(event.pointerId);
            }
        });
        this.comparison.addEventListener("pointercancel", () => {
            this.comparison.classList.remove("dragging");
        });
    }

    bindDropEvents() {
        this.element.addEventListener("dragover", (event) => {
            if (!this.canReplaceWithDrop(event)) return;
            event.preventDefault();
            event.stopPropagation();
            if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
            this.element.classList.add("drop-ready");
        });
        this.element.addEventListener("dragleave", (event) => {
            if (event.relatedTarget && this.element.contains(event.relatedTarget)) return;
            this.element.classList.remove("drop-ready");
        });
        this.element.addEventListener("drop", (event) => {
            const file = this.droppedImageFile(event);
            this.element.classList.remove("drop-ready");
            if (!file) return;
            event.preventDefault();
            event.stopPropagation();
            void this.replaceImageFile(file);
        });
    }

    register() {
        for (const [id, editor] of editors) {
            if (editor === this) editors.delete(id);
        }
        for (const id of executionIds({
            node: this.node.id,
            display_node: this.node.display_id,
        })) {
            editors.set(id, this);
        }
    }

    remove() {
        for (const [id, editor] of editors) {
            if (editor === this) editors.delete(id);
        }
    }

    setStatus(message, kind = "") {
        this.status.textContent = message;
        this.status.dataset.kind = kind;
    }

    setBusy(busy) {
        this.busy = busy;
        this.element.classList.toggle("busy", busy);
        this.updateHistoryButtons();
        if (busy) this.brushCursor.classList.remove("visible");
        else this.updateBrushCursor();
    }

    updateBrushCursor(event) {
        if (event) {
            this.lastBrushPointer = {
                clientX: event.clientX,
                clientY: event.clientY,
            };
        }
        if (!this.lastBrushPointer || !this.maskCanvas.width || this.busy) {
            this.brushCursor.classList.remove("visible");
            return;
        }
        const bounds = this.maskCanvas.getBoundingClientRect();
        if (!bounds.width || !bounds.height) return;
        const screenX = this.lastBrushPointer.clientX - bounds.left;
        const screenY = this.lastBrushPointer.clientY - bounds.top;
        const inside = (
            screenX >= 0 && screenX <= bounds.width &&
            screenY >= 0 && screenY <= bounds.height
        );
        this.brushCursor.classList.toggle("visible", inside);
        if (!inside) return;
        const localWidth = this.maskCanvas.clientWidth;
        const localHeight = this.maskCanvas.clientHeight;
        if (!localWidth || !localHeight) return;
        const x = screenX * localWidth / bounds.width;
        const y = screenY * localHeight / bounds.height;
        const brushSize = Number(this.brushWidget.value) || 48;
        const diameter = Math.max(
            2,
            brushSize * localWidth / this.maskCanvas.width,
        );
        this.brushCursor.style.left = `${x}px`;
        this.brushCursor.style.top = `${y}px`;
        this.brushCursor.style.width = `${diameter}px`;
        this.brushCursor.style.height = `${diameter}px`;
    }

    adjustBrushSize(event) {
        if (!event.altKey || event.deltaY === 0) return;
        event.preventDefault();
        event.stopPropagation();
        const current = Number(this.brushWidget.value) || 48;
        const step = Math.max(1, Math.round(current * 0.1));
        const direction = event.deltaY < 0 ? 1 : -1;
        const next = Math.max(
            BRUSH_MIN_SIZE,
            Math.min(BRUSH_MAX_SIZE, current + direction * step),
        );
        if (next === current) return;
        this.brushWidget.value = next;
        this.brushWidget.callback?.(next);
        this.updateBrushCursor(event);
        this.node.setDirtyCanvas?.(true, true);
    }

    handleWheel(event) {
        const bounds = this.maskCanvas.getBoundingClientRect();
        const overPaintCanvas = (
            event.clientX >= bounds.left && event.clientX <= bounds.right &&
            event.clientY >= bounds.top && event.clientY <= bounds.bottom
        );
        if (event.altKey && overPaintCanvas) {
            this.adjustBrushSize(event);
            return;
        }
        event.preventDefault();
        event.stopPropagation();
        app.canvas?.processMouseWheel?.(event);
    }

    moveComparison(event) {
        const bounds = this.comparison.getBoundingClientRect();
        if (!bounds.width) return;
        const percent = Math.max(0, Math.min(100, (
            (event.clientX - bounds.left) / bounds.width
        ) * 100));
        this.comparison.style.setProperty("--dvd-split", `${percent}%`);
    }

    setMode(mode) {
        this.mode = mode;
        this.paintButton.classList.toggle("active", mode === "paint");
        this.eraseButton.classList.toggle("active", mode === "erase");
    }

    point(event) {
        const bounds = this.maskCanvas.getBoundingClientRect();
        return {
            x: Math.max(
                0,
                Math.min(
                    this.maskCanvas.width - 1,
                    (event.clientX - bounds.left) * this.maskCanvas.width / bounds.width,
                ),
            ),
            y: Math.max(
                0,
                Math.min(
                    this.maskCanvas.height - 1,
                    (event.clientY - bounds.top) * this.maskCanvas.height / bounds.height,
                ),
            ),
        };
    }

    maskContext() {
        const context = this.maskCanvas.getContext("2d", { willReadFrequently: true });
        context.lineCap = "round";
        context.lineJoin = "round";
        context.lineWidth = Number(this.brushWidget.value) || 48;
        context.strokeStyle = "#ffffff";
        context.fillStyle = "#ffffff";
        context.globalCompositeOperation = this.mode === "erase" ? "destination-out" : "source-over";
        return context;
    }

    pointerDown(event) {
        if (this.busy || !this.baseCanvas.width || event.button !== 0) return;
        event.preventDefault();
        this.pushHistory();
        this.drawing = true;
        this.maskCanvas.setPointerCapture(event.pointerId);
        const point = this.point(event);
        const context = this.maskContext();
        context.beginPath();
        context.arc(point.x, point.y, context.lineWidth / 2, 0, Math.PI * 2);
        context.fill();
        context.beginPath();
        context.moveTo(point.x, point.y);
    }

    pointerMove(event) {
        if (!this.drawing) return;
        event.preventDefault();
        const point = this.point(event);
        const context = this.maskContext();
        context.lineTo(point.x, point.y);
        context.stroke();
    }

    async pointerUp(event, submit = true) {
        if (!this.drawing) return;
        event.preventDefault();
        this.drawing = false;
        if (this.maskCanvas.hasPointerCapture(event.pointerId)) {
            this.maskCanvas.releasePointerCapture(event.pointerId);
        }
        if (submit) await this.submitMask();
    }

    pushHistory() {
        if (!this.maskCanvas.width) return;
        this.history.push(this.maskCanvas.toDataURL("image/png"));
        if (this.history.length > 12) this.history.shift();
        this.updateHistoryButtons();
    }

    async undo() {
        const snapshot = this.history.pop();
        if (!snapshot) return;
        const image = await loadImage(snapshot);
        const context = this.maskCanvas.getContext("2d");
        context.clearRect(0, 0, this.maskCanvas.width, this.maskCanvas.height);
        context.drawImage(image, 0, 0);
        this.updateHistoryButtons();
        this.setStatus("Mask restored");
    }

    updateHistoryButtons() {
        this.undoButton.disabled = this.busy || this.history.length === 0;
        this.processUndoButton.disabled = this.busy || this.processHistory.length === 0;
    }

    captureProcessedState() {
        return {
            image: this.imageWidget.value,
            comparison: this.hasComparison ? {
                before: this.compareBeforeCanvas.toDataURL("image/png"),
                after: this.compareAfterCanvas.toDataURL("image/png"),
                split: this.comparison.style.getPropertyValue("--dvd-split") || "50%",
            } : null,
        };
    }

    async restoreComparison(snapshot) {
        if (!snapshot) {
            this.clearComparison();
            return;
        }
        const [beforeImage, afterImage] = await Promise.all([
            loadImage(snapshot.before),
            loadImage(snapshot.after),
        ]);
        const width = afterImage.naturalWidth || afterImage.width;
        const height = afterImage.naturalHeight || afterImage.height;
        this.comparisonCanvases(width, height);
        this.compareBeforeCanvas.getContext("2d").drawImage(beforeImage, 0, 0, width, height);
        this.compareAfterCanvas.getContext("2d").drawImage(afterImage, 0, 0, width, height);
        this.comparison.style.setProperty("--dvd-split", snapshot.split || "50%");
        this.comparison.classList.add("visible");
        this.hasComparison = true;
        this.fitNodeToImage();
    }

    clearProcessHistory() {
        this.processHistory = [];
        this.updateHistoryButtons();
    }

    pushProcessHistory(state) {
        this.processHistory.push(state);
        if (this.processHistory.length > 12) this.processHistory.shift();
        this.updateHistoryButtons();
    }

    async undoProcessedEdit() {
        if (this.busy || this.processHistory.length === 0) return;
        const currentState = this.captureProcessedState();
        const previousState = this.processHistory.pop();
        this.setBusy(true);
        this.setStatus("Restoring previous edit...", "working");
        try {
            this.maskWidget.value = "";
            this.setImageWidget(previousState.image);
            if (!await this.loadInput(previousState.image, true)) {
                throw new Error("Could not restore the previous image.");
            }
            await this.restoreComparison(previousState.comparison);
            this.node.imgs = [];
            this.node.images = [];
            app.graph.setDirtyCanvas(true, true);
            this.setStatus("Previous edit restored", "ready");
        } catch (error) {
            this.processHistory.push(previousState);
            try {
                this.setImageWidget(currentState.image);
                await this.loadInput(currentState.image, true);
                await this.restoreComparison(currentState.comparison);
            } catch (restoreError) {
                console.error("DvD IOPaint: could not restore current edit:", restoreError);
            }
            this.setStatus(error.message || "Undo failed", "error");
            console.error("DvD IOPaint:", error);
        } finally {
            this.setBusy(false);
        }
    }

    clearMask(recordHistory = false) {
        if (recordHistory) this.pushHistory();
        this.maskCanvas.getContext("2d").clearRect(
            0,
            0,
            this.maskCanvas.width,
            this.maskCanvas.height,
        );
        if (!recordHistory) this.history = [];
        this.updateHistoryButtons();
        this.setStatus("Mask cleared");
    }

    resizeCanvases(width, height) {
        this.baseCanvas.width = width;
        this.baseCanvas.height = height;
        this.maskCanvas.width = width;
        this.maskCanvas.height = height;
        this.stage.style.setProperty("--dvd-aspect", String(width / height));
        this.updateBrushCursor();
    }

    comparisonCanvases(width, height) {
        for (const canvas of [this.compareBeforeCanvas, this.compareAfterCanvas]) {
            canvas.width = width;
            canvas.height = height;
        }
    }

    showComparison(beforeCanvas, afterImage) {
        const width = afterImage.naturalWidth || afterImage.width;
        const height = afterImage.naturalHeight || afterImage.height;
        this.comparisonCanvases(width, height);
        this.compareBeforeCanvas.getContext("2d").drawImage(
            beforeCanvas,
            0,
            0,
            width,
            height,
        );
        this.compareAfterCanvas.getContext("2d").drawImage(afterImage, 0, 0, width, height);
        this.comparison.style.setProperty("--dvd-split", "50%");
        this.comparison.classList.add("visible");
        this.hasComparison = true;
        this.fitNodeToImage();
    }

    clearComparison() {
        this.hasComparison = false;
        this.comparison?.classList.remove("visible", "dragging");
        this.fitNodeToImage();
    }

    fitNodeToImage() {
        if (this.layoutQueued) return;
        this.layoutQueued = true;
        requestAnimationFrame(() => {
            this.layoutQueued = false;
            if (this.layoutInProgress) return;
            this.layoutInProgress = true;
            try {
                const nodeWidth = Math.max(MIN_NODE_WIDTH, this.node.size?.[0] || MIN_NODE_WIDTH);
                const availableWidth = Math.max(100, nodeWidth - 30);
                const width = this.baseCanvas.width || 1;
                const height = this.baseCanvas.height || 1;
                const aspect = width / height;
                const editorWidth = Math.min(availableWidth, EDITOR_MAX_HEIGHT * aspect);
                const editorHeight = Math.max(100, editorWidth / aspect);
                this.stage.style.width = `${Math.round(editorWidth)}px`;
                this.stage.style.height = `${Math.round(editorHeight)}px`;

                let comparisonHeight = 0;
                if (this.hasComparison) {
                    const comparisonWidth = Math.min(
                        availableWidth,
                        COMPARISON_MAX_HEIGHT * aspect,
                    );
                    comparisonHeight = comparisonWidth / aspect;
                    this.comparison.style.width = `${Math.round(comparisonWidth)}px`;
                    this.comparison.style.height = `${Math.round(comparisonHeight)}px`;
                }

                const toolbarHeight = 38;
                const comparisonGap = this.hasComparison ? 10 : 0;
                this.widgetHeight = Math.ceil(
                    14 + toolbarHeight + editorHeight + comparisonGap + comparisonHeight
                );
                this.element.style.setProperty("--dvd-widget-height", `${this.widgetHeight}px`);

                const rowHeight = (globalThis.LiteGraph?.NODE_WIDGET_HEIGHT ?? 20) + 4;
                const titleHeight = globalThis.LiteGraph?.NODE_TITLE_HEIGHT ?? 30;
                const nativeRows = (this.node.widgets || []).filter((widget) => (
                    widget !== this.domWidget &&
                    !widget.hidden &&
                    !String(widget.type || "").startsWith("converted-widget")
                )).length;
                const nodeHeight = titleHeight + nativeRows * rowHeight + this.widgetHeight + 12;
                if (
                    Math.abs((this.node.size?.[0] || 0) - nodeWidth) > 0.5 ||
                    Math.abs((this.node.size?.[1] || 0) - nodeHeight) > 0.5
                ) {
                    this.node.setSize([nodeWidth, nodeHeight]);
                }
                this.node.graph?.setDirtyCanvas(true, true);
            } finally {
                this.layoutInProgress = false;
            }
        });
    }

    async loadInput(filename, clearMask) {
        if (!filename) {
            this.setStatus("Choose an image");
            return false;
        }
        const generation = ++this.loadGeneration;
        try {
            const image = await loadImage(inputImageUrl(filename));
            if (generation !== this.loadGeneration) return false;
            this.resizeCanvases(image.naturalWidth, image.naturalHeight);
            this.baseCanvas.getContext("2d").drawImage(image, 0, 0);
            if (clearMask) this.clearMask(false);
            this.setStatus(`${image.naturalWidth} x ${image.naturalHeight}`, "ready");
            this.fitNodeToImage();
            this.node.setDirtyCanvas?.(true, true);
            return true;
        } catch (error) {
            this.setStatus("Image load failed", "error");
            console.error("DvD IOPaint:", error);
            return false;
        }
    }

    async loadMask(filename) {
        if (!filename || !this.maskCanvas.width) return;
        try {
            const image = await loadImage(inputImageUrl(filename));
            const temp = document.createElement("canvas");
            temp.width = this.maskCanvas.width;
            temp.height = this.maskCanvas.height;
            const tempContext = temp.getContext("2d", { willReadFrequently: true });
            tempContext.drawImage(image, 0, 0, temp.width, temp.height);
            const pixels = tempContext.getImageData(0, 0, temp.width, temp.height);
            for (let index = 0; index < pixels.data.length; index += 4) {
                const value = pixels.data[index];
                pixels.data[index] = 255;
                pixels.data[index + 1] = 255;
                pixels.data[index + 2] = 255;
                pixels.data[index + 3] = value;
            }
            this.maskCanvas.getContext("2d").putImageData(pixels, 0, 0);
        } catch (error) {
            console.error("DvD IOPaint: could not restore mask:", error);
        }
    }

    async restore() {
        this.register();
        await this.loadInput(this.imageWidget.value, true);
        await this.loadMask(this.maskWidget.value);
    }

    async maskBlob() {
        const exportCanvas = document.createElement("canvas");
        exportCanvas.width = this.maskCanvas.width;
        exportCanvas.height = this.maskCanvas.height;
        const context = exportCanvas.getContext("2d");
        context.fillStyle = "#000000";
        context.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
        context.drawImage(this.maskCanvas, 0, 0);
        return canvasBlob(exportCanvas);
    }

    async submitMask() {
        this.register();
        this.executionSeen = false;
        this.resultProcessing = false;
        this.promptId = null;
        this.setBusy(true);
        this.setStatus("Uploading mask...", "working");
        try {
            const blob = await this.maskBlob();
            const filename = `dvd_iopaint_${this.node.id}_mask.png`;
            this.maskWidget.value = await uploadImage(blob, filename, "dvd_iopaint");
            app.graph.setDirtyCanvas(true, true);
            if (this.autoRunWidget.value) {
                this.setStatus("Queued...", "working");
                const accepted = await app.queuePrompt();
                this.promptId = accepted?.prompt_id || accepted?.promptId || this.promptId;
                if (!this.busy) return;
                if (accepted === false && !app.processingQueue) {
                    this.setBusy(false);
                    this.setStatus("Queue validation failed", "error");
                    return;
                }
                this.setStatus(
                    this.executionSeen ? "Running..." : "Queued...",
                    "working",
                );
            } else {
                this.setBusy(false);
                this.setStatus("Mask ready", "ready");
            }
        } catch (error) {
            this.setBusy(false);
            this.setStatus(error.message || "Queue failed", "error");
            console.error("DvD IOPaint:", error);
        }
    }

    setImageWidget(filename) {
        const values = this.imageWidget.options?.values;
        if (Array.isArray(values) && !values.includes(filename)) values.push(filename);
        this.imageWidget.value = filename;
    }

    droppedImageFile(event) {
        const file = [...(event.dataTransfer?.files || [])].find(isImageFile);
        if (file) return file;
        const item = [...(event.dataTransfer?.items || [])].find((candidate) => (
            candidate.kind === "file" && (
                candidate.type?.startsWith("image/") ||
                !candidate.type
            )
        ));
        return item?.getAsFile?.() || null;
    }

    canReplaceWithDrop(event) {
        const files = [...(event.dataTransfer?.files || [])];
        if (files.some(isImageFile)) return true;
        return [...(event.dataTransfer?.items || [])].some((item) => (
            item.kind === "file" && (
                item.type?.startsWith("image/") ||
                !item.type
            )
        ));
    }

    async replaceImageFile(file) {
        if (!isImageFile(file) || this.busy) return false;
        this.setBusy(true);
        this.setStatus("Uploading image...", "working");
        try {
            const inputPath = await uploadImage(file, file.name, "dvd_iopaint");
            this.clearProcessHistory();
            this.clearComparison();
            this.setImageWidget(inputPath);
            this.maskWidget.value = "";
            await this.loadInput(inputPath, true);
            app.graph.setDirtyCanvas(true, true);
            this.setStatus("Image replaced", "ready");
            return true;
        } catch (error) {
            this.setStatus(error.message || "Image upload failed", "error");
            console.error("DvD IOPaint:", error);
            return false;
        } finally {
            this.setBusy(false);
        }
    }

    markExecuting(detail) {
        this.executionSeen = true;
        this.promptId = detail?.prompt_id || detail?.promptId || this.promptId;
        this.setBusy(true);
        this.setStatus("Running...", "working");
    }

    finishWithError(message) {
        this.executionSeen = false;
        this.resultProcessing = false;
        this.promptId = null;
        this.setBusy(false);
        this.setStatus(message, "error");
    }

    async handleExecuted(output) {
        const results = (
            output?.dvd_iopaint_result ||
            output?.output?.dvd_iopaint_result ||
            output?.images
        );
        const result = Array.isArray(results) ? results[0] : results;
        if (!result) {
            this.finishWithError("Finished without result");
            return;
        }
        this.executionSeen = true;
        this.resultProcessing = true;
        this.setBusy(true);
        let previousState = null;
        try {
            previousState = this.captureProcessedState();
            this.maskWidget.value = "";
            this.clearMask(false);
            this.setStatus("Updating canvas...", "working");
            const afterImage = await loadOutputImage(result);
            const beforeResults = (
                output?.dvd_iopaint_before ||
                output?.output?.dvd_iopaint_before
            );
            const beforeResult = Array.isArray(beforeResults)
                ? beforeResults[0]
                : beforeResults;
            // The connected IMAGE input is not necessarily the image currently
            // shown by the node's file widget.  The backend now returns an
            // explicit before-preview from the tensor used for this execution;
            // use it whenever available and fall back to the canvas for the
            // original no-connection workflow.
            const beforeImage = beforeResult
                ? await loadOutputImage(beforeResult)
                : this.baseCanvas;
            this.showComparison(beforeImage, afterImage);
            const response = await fetch(outputImageUrl(result));
            if (!response.ok) throw new Error(`Result fetch failed: ${response.status}`);
            const blob = await response.blob();
            const filename = (
                `dvd_iopaint_${this.node.id}_result_${Date.now()}_${++this.resultVersion}.png`
            );
            const inputPath = await uploadImage(blob, filename, "dvd_iopaint");
            this.setImageWidget(inputPath);
            if (!await this.loadInput(inputPath, true)) {
                throw new Error("Could not load the processed image.");
            }
            this.node.imgs = [];
            this.node.images = [];
            this.pushProcessHistory(previousState);
            app.graph.setDirtyCanvas(true, true);
            this.setStatus("Removal complete", "ready");
        } catch (error) {
            if (previousState) {
                try {
                    this.setImageWidget(previousState.image);
                    await this.loadInput(previousState.image, true);
                    await this.restoreComparison(previousState.comparison);
                } catch (restoreError) {
                    console.error("DvD IOPaint: could not restore previous edit:", restoreError);
                }
            }
            this.setStatus(error.message || "Result update failed", "error");
            console.error("DvD IOPaint:", error);
        } finally {
            this.executionSeen = false;
            this.resultProcessing = false;
            this.promptId = null;
            this.setBusy(false);
        }
    }
}


class DvDSAMEditor {
    constructor(node) {
        this.node = node;
        this.imageWidget = node.widgets.find((widget) => widget.name === "image");
        this.pointsWidget = node.widgets.find((widget) => widget.name === "points");
        this.autoRunWidget = node.widgets.find((widget) => widget.name === "auto_run");
        this.clicks = [];
        this.history = [];
        this.busy = false;
        this.loadGeneration = 0;
        this.layoutQueued = false;
        this.widgetHeight = 300;
        this.executionSeen = false;
        this.promptId = null;
        this.registered = false;
        hideWidget(this.pointsWidget);
        this.element = this.buildElement();
        this.domWidget = node.addDOMWidget(
            "dvd_sam_canvas",
            "dvd_sam_canvas",
            this.element,
            {
                serialize: false,
                hideOnZoom: false,
                margin: 0,
                getMinHeight: () => this.widgetHeight,
                getMaxHeight: () => this.widgetHeight,
                getHeight: () => this.widgetHeight,
            },
        );
        const originalImageCallback = this.imageWidget.callback;
        this.imageWidget.callback = (...args) => {
            const result = originalImageCallback?.apply(this.imageWidget, args);
            this.clearPoints(false);
            void this.loadInput(this.imageWidget.value);
            return result;
        };
        this.bindPointerEvents();
        this.restore();
    }

    buildElement() {
        const root = document.createElement("div");
        root.className = "dvd-iopaint-editor dvd-sam-editor";
        const toolbar = document.createElement("div");
        toolbar.className = "dvd-iopaint-toolbar";
        this.foregroundButton = iconButton("pi-plus", "Add foreground point (left click)", () => this.setMode(1));
        this.backgroundButton = iconButton("pi-minus", "Add background point (right click)", () => this.setMode(0));
        this.undoButton = iconButton("pi-undo", "Undo last point", () => this.undoPoint());
        this.clearButton = iconButton("pi-trash", "Clear all points and mask", () => this.clearPoints(true));
        this.runButton = iconButton("pi-play", "Run SAM segmentation", () => this.submitPoints());
        this.status = document.createElement("span");
        this.status.className = "dvd-iopaint-status";
        this.status.textContent = "Choose an image; left click foreground, right click background";
        toolbar.append(this.foregroundButton, this.backgroundButton, this.undoButton, this.clearButton, this.runButton, this.status);
        this.stage = document.createElement("div");
        this.stage.className = "dvd-iopaint-stage dvd-sam-stage";
        this.baseCanvas = document.createElement("canvas");
        this.baseCanvas.className = "dvd-iopaint-base";
        this.maskCanvas = document.createElement("canvas");
        this.maskCanvas.className = "dvd-sam-mask";
        this.pointsCanvas = document.createElement("canvas");
        this.pointsCanvas.className = "dvd-sam-points";
        this.stage.append(this.baseCanvas, this.maskCanvas, this.pointsCanvas);
        root.append(toolbar, this.stage);
        this.setMode(1);
        return root;
    }

    bindPointerEvents() {
        this.pointsCanvas.addEventListener("contextmenu", (event) => event.preventDefault());
        this.pointsCanvas.addEventListener("pointerdown", (event) => this.pointerDown(event));
        this.element.addEventListener("wheel", (event) => {
            event.preventDefault();
            event.stopPropagation();
            app.canvas?.processMouseWheel?.(event);
        }, { passive: false });
    }

    register() {
        for (const [id, editor] of editors) if (editor === this) editors.delete(id);
        for (const id of executionIds({ node: this.node.id, display_node: this.node.display_id })) editors.set(id, this);
        this.registered = true;
    }

    remove() {
        for (const [id, editor] of editors) if (editor === this) editors.delete(id);
        this.registered = false;
    }

    setStatus(message, kind = "") {
        this.status.textContent = message;
        this.status.dataset.kind = kind;
    }

    setBusy(value) {
        this.busy = value;
        this.element.classList.toggle("busy", value);
        this.updateButtons();
    }

    updateButtons() {
        this.undoButton.disabled = this.busy || this.history.length === 0;
        this.clearButton.disabled = this.busy || this.clicks.length === 0;
        this.runButton.disabled = this.busy || this.clicks.length === 0;
    }

    setMode(label) {
        this.mode = label;
        this.foregroundButton.classList.toggle("active", label === 1);
        this.backgroundButton.classList.toggle("active", label === 0);
    }

    point(event) {
        const bounds = this.pointsCanvas.getBoundingClientRect();
        return {
            x: Math.max(
                0,
                Math.min(
                    this.pointsCanvas.width - 1,
                    (event.clientX - bounds.left) * this.pointsCanvas.width / bounds.width,
                ),
            ),
            y: Math.max(
                0,
                Math.min(
                    this.pointsCanvas.height - 1,
                    (event.clientY - bounds.top) * this.pointsCanvas.height / bounds.height,
                ),
            ),
        };
    }

    pointerDown(event) {
        if (this.busy || !this.baseCanvas.width || (event.button !== 0 && event.button !== 2)) return;
        event.preventDefault();
        this.history.push(JSON.stringify(this.clicks));
        if (this.history.length > 24) this.history.shift();
        const point = this.point(event);
        const label = event.button === 2 ? 0 : this.mode;
        this.clicks.push([Number(point.x.toFixed(2)), Number(point.y.toFixed(2)), label]);
        this.pointsWidget.value = JSON.stringify(this.clicks);
        this.renderPoints();
        this.updateButtons();
        this.setStatus(`${this.clicks.length} point(s) · ${label ? "foreground" : "background"}`, "ready");
        if (this.autoRunWidget?.value) void this.submitPoints();
    }

    renderPoints() {
        const context = this.pointsCanvas.getContext("2d");
        context.clearRect(0, 0, this.pointsCanvas.width, this.pointsCanvas.height);
        for (const [x, y, label] of this.clicks) {
            context.beginPath();
            context.arc(x, y, Math.max(7, Math.min(this.pointsCanvas.width, this.pointsCanvas.height) * 0.012), 0, Math.PI * 2);
            context.fillStyle = label ? "rgb(80 220 130 / 85%)" : "rgb(235 90 90 / 85%)";
            context.fill();
            context.lineWidth = 2;
            context.strokeStyle = "#ffffff";
            context.stroke();
        }
    }

    async undoPoint() {
        const snapshot = this.history.pop();
        if (!snapshot) return;
        this.clicks = JSON.parse(snapshot);
        this.pointsWidget.value = JSON.stringify(this.clicks);
        this.renderPoints();
        this.updateButtons();
        this.setStatus("Point restored", "ready");
    }

    clearPoints(recordHistory) {
        if (recordHistory && this.clicks.length) this.history.push(JSON.stringify(this.clicks));
        this.clicks = [];
        if (this.pointsWidget) this.pointsWidget.value = "[]";
        this.maskCanvas.getContext("2d").clearRect(0, 0, this.maskCanvas.width, this.maskCanvas.height);
        this.renderPoints();
        this.updateButtons();
        this.setStatus("Points cleared", "ready");
    }

    async submitPoints() {
        if (this.busy || !this.clicks.length) return;
        this.register();
        this.pointsWidget.value = JSON.stringify(this.clicks);
        this.promptId = null;
        this.setBusy(true);
        this.setStatus("Queued SAM segmentation...", "working");
        try {
            const accepted = await app.queuePrompt();
            this.promptId = accepted?.prompt_id || accepted?.promptId || this.promptId;
            if (accepted === false && !app.processingQueue) throw new Error("Queue validation failed");
        } catch (error) {
            this.setBusy(false);
            this.setStatus(error.message || "Queue failed", "error");
        }
    }

    markExecuting(detail) {
        this.executionSeen = true;
        this.promptId = detail?.prompt_id || detail?.promptId || this.promptId;
        this.setBusy(true);
        this.setStatus("Running SAM...", "working");
    }

    finishWithError(message) {
        this.executionSeen = false;
        this.promptId = null;
        this.setBusy(false);
        this.setStatus(message, "error");
    }

    async handleExecuted(output) {
        const values = output?.dvd_sam_mask || output?.output?.dvd_sam_mask;
        const sourceValues = output?.dvd_sam_source || output?.output?.dvd_sam_source;
        const result = Array.isArray(values) ? values[0] : values;
        const sourceResult = Array.isArray(sourceValues) ? sourceValues[0] : sourceValues;
        if (result) {
            try {
                const [image, sourceImage] = await Promise.all([
                    loadOutputImage(result),
                    sourceResult ? loadOutputImage(sourceResult) : Promise.resolve(null),
                ]);
                if (sourceImage) {
                    this.resizeCanvases(sourceImage.naturalWidth, sourceImage.naturalHeight);
                    this.baseCanvas.getContext("2d").drawImage(sourceImage, 0, 0);
                    this.renderPoints();
                    this.fitNodeToImage();
                }
                const context = this.maskCanvas.getContext("2d");
                const previewCanvas = document.createElement("canvas");
                previewCanvas.width = this.maskCanvas.width;
                previewCanvas.height = this.maskCanvas.height;
                const previewContext = previewCanvas.getContext("2d", { willReadFrequently: true });
                previewContext.drawImage(image, 0, 0, previewCanvas.width, previewCanvas.height);
                const pixels = previewContext.getImageData(0, 0, previewCanvas.width, previewCanvas.height);
                for (let index = 0; index < pixels.data.length; index += 4) {
                    const value = pixels.data[index];
                    pixels.data[index] = 255;
                    pixels.data[index + 1] = 255;
                    pixels.data[index + 2] = 255;
                    pixels.data[index + 3] = value;
                }
                context.clearRect(0, 0, this.maskCanvas.width, this.maskCanvas.height);
                context.putImageData(pixels, 0, 0);
                this.setStatus("SAM mask updated", "ready");
            } catch (error) {
                this.setStatus(error.message || "Mask preview failed", "error");
            }
        } else {
            this.setStatus("SAM finished without mask preview", "error");
        }
        this.executionSeen = false;
        this.promptId = null;
        this.setBusy(false);
    }

    resizeCanvases(width, height) {
        for (const canvas of [this.baseCanvas, this.maskCanvas, this.pointsCanvas]) {
            canvas.width = width;
            canvas.height = height;
        }
        this.stage.style.setProperty("--dvd-aspect", String(width / height));
        this.renderPoints();
    }

    fitNodeToImage() {
        if (this.layoutQueued) return;
        this.layoutQueued = true;
        requestAnimationFrame(() => {
            this.layoutQueued = false;
            const nodeWidth = Math.max(500, this.node.size?.[0] || 500);
            const availableWidth = Math.max(100, nodeWidth - 30);
            const aspect = (this.baseCanvas.width || 1) / (this.baseCanvas.height || 1);
            const width = Math.min(availableWidth, 520);
            const height = Math.max(120, width / aspect);
            this.stage.style.width = `${Math.round(width)}px`;
            this.stage.style.height = `${Math.round(height)}px`;
            this.widgetHeight = Math.ceil(height + 52);
            this.element.style.setProperty("--dvd-widget-height", `${this.widgetHeight}px`);
            const rows = (this.node.widgets || []).filter((widget) => widget !== this.domWidget && !widget.hidden && !String(widget.type || "").startsWith("converted-widget")).length;
            const titleHeight = globalThis.LiteGraph?.NODE_TITLE_HEIGHT ?? 30;
            const rowHeight = (globalThis.LiteGraph?.NODE_WIDGET_HEIGHT ?? 20) + 4;
            this.node.setSize([nodeWidth, titleHeight + rows * rowHeight + this.widgetHeight + 12]);
            this.node.graph?.setDirtyCanvas(true, true);
        });
    }

    async loadInput(filename) {
        if (!filename) {
            this.setStatus("Choose an image");
            return false;
        }
        const generation = ++this.loadGeneration;
        try {
            const image = await loadImage(inputImageUrl(filename));
            if (generation !== this.loadGeneration) return false;
            this.resizeCanvases(image.naturalWidth, image.naturalHeight);
            this.baseCanvas.getContext("2d").drawImage(image, 0, 0);
            this.setStatus(`${image.naturalWidth} × ${image.naturalHeight}`, "ready");
            this.fitNodeToImage();
            return true;
        } catch (error) {
            this.setStatus("Image load failed", "error");
            return false;
        }
    }

    setImageWidget(filename) {
        const values = this.imageWidget.options?.values;
        if (Array.isArray(values) && !values.includes(filename)) values.push(filename);
        this.imageWidget.value = filename;
    }

    droppedImageFile(event) {
        return [...(event.dataTransfer?.files || [])].find(isImageFile) || null;
    }

    async replaceImageFile(file) {
        if (!isImageFile(file) || this.busy) return false;
        this.setBusy(true);
        this.setStatus("Uploading image...", "working");
        try {
            const inputPath = await uploadImage(file, file.name, "dvd_iopaint");
            this.setImageWidget(inputPath);
            this.clearPoints(false);
            await this.loadInput(inputPath);
            this.setStatus("Image replaced", "ready");
            return true;
        } catch (error) {
            this.setStatus(error.message || "Image upload failed", "error");
            return false;
        } finally {
            this.setBusy(false);
        }
    }

    async restore() {
        this.register();
        await this.loadInput(this.imageWidget.value);
        try {
            const parsed = JSON.parse(this.pointsWidget.value || "[]");
            this.clicks = Array.isArray(parsed) ? parsed : [];
        } catch {
            this.clicks = [];
        }
        this.renderPoints();
        this.updateButtons();
    }
}


api.addEventListener("executed", ({ detail }) => {
    findEditor(detail)?.handleExecuted(detail?.output || detail);
});

api.addEventListener("executing", ({ detail }) => {
    const editor = findEditor(detail);
    if (!editor) return;
    editor.markExecuting(detail);
});

api.addEventListener("execution_error", ({ detail }) => {
    finishExecutionEditors(
        detail,
        detail?.exception_message || "Execution failed",
    );
});

api.addEventListener("execution_interrupted", ({ detail }) => {
    finishExecutionEditors(detail, "Execution interrupted");
});

api.addEventListener("execution_success", ({ detail }) => {
    setTimeout(() => {
        const target = findEditor(detail);
        const candidates = target ? [target] : [...uniqueEditors()];
        for (const editor of candidates) {
            if (editor.busy && editor.executionSeen && !editor.resultProcessing) {
                editor.finishWithError("Finished without result");
            }
        }
    }, 0);
});


app.registerExtension({
    name: "DvD.IOPaint.InteractiveEraser",
    init() {
        addStylesheet();
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            this.dvdIOPaintEditor = new DvDIOPaintEditor(this);
            this.setSize([Math.max(this.size[0], MIN_NODE_WIDTH), Math.max(this.size[1], 620)]);
            return result;
        };

        const originalResized = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function () {
            const result = originalResized?.apply(this, arguments);
            this.dvdIOPaintEditor?.fitNodeToImage();
            return result;
        };

        const originalAdded = nodeType.prototype.onAdded;
        nodeType.prototype.onAdded = function () {
            const result = originalAdded?.apply(this, arguments);
            this.dvdIOPaintEditor?.register();
            return result;
        };

        const originalDrawBackground = nodeType.prototype.onDrawBackground;
        nodeType.prototype.onDrawBackground = function () {
            this.imgs = [];
            this.images = [];
            return originalDrawBackground?.apply(this, arguments);
        };

        const originalDragOver = nodeType.prototype.onDragOver;
        nodeType.prototype.onDragOver = function (event) {
            if (this.dvdIOPaintEditor?.canReplaceWithDrop(event)) return true;
            return originalDragOver?.apply(this, arguments) ?? false;
        };

        const originalDragDrop = nodeType.prototype.onDragDrop;
        nodeType.prototype.onDragDrop = function (event) {
            const file = this.dvdIOPaintEditor?.droppedImageFile(event);
            if (file) {
                event.preventDefault();
                event.stopPropagation?.();
                void this.dvdIOPaintEditor.replaceImageFile(file);
                return true;
            }
            return originalDragDrop?.apply(this, arguments) ?? false;
        };

        const originalConfigured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalConfigured?.apply(this, arguments);
            setTimeout(() => this.dvdIOPaintEditor?.restore(), 0);
            return result;
        };

        const originalRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this.dvdIOPaintEditor?.remove();
            return originalRemoved?.apply(this, arguments);
        };
    },
});


app.registerExtension({
    name: "DvD.IOPaint.SAMInteractiveSegmentation",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "DvD_IOPaint_SAM_Interactive_Segmentation") return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            this.dvdSAMEditor = new DvDSAMEditor(this);
            this.setSize([Math.max(this.size[0], 500), Math.max(this.size[1], 500)]);
            return result;
        };

        const originalResized = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function () {
            const result = originalResized?.apply(this, arguments);
            this.dvdSAMEditor?.fitNodeToImage();
            return result;
        };

        const originalAdded = nodeType.prototype.onAdded;
        nodeType.prototype.onAdded = function () {
            const result = originalAdded?.apply(this, arguments);
            this.dvdSAMEditor?.register();
            return result;
        };

        const originalDrawBackground = nodeType.prototype.onDrawBackground;
        nodeType.prototype.onDrawBackground = function () {
            // The SAM source and mask previews already render inside the DOM
            // editor. Suppress ComfyUI's duplicate native image thumbnail.
            this.imgs = [];
            this.images = [];
            return originalDrawBackground?.apply(this, arguments);
        };

        const originalConfigured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalConfigured?.apply(this, arguments);
            setTimeout(() => this.dvdSAMEditor?.restore(), 0);
            return result;
        };

        const originalRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this.dvdSAMEditor?.remove();
            return originalRemoved?.apply(this, arguments);
        };

        const originalDragOver = nodeType.prototype.onDragOver;
        nodeType.prototype.onDragOver = function (event) {
            const file = this.dvdSAMEditor?.droppedImageFile?.(event);
            if (file) return true;
            return originalDragOver?.apply(this, arguments) ?? false;
        };

        const originalDragDrop = nodeType.prototype.onDragDrop;
        nodeType.prototype.onDragDrop = function (event) {
            const file = this.dvdSAMEditor?.droppedImageFile?.(event);
            if (file) {
                event.preventDefault();
                event.stopPropagation?.();
                void this.dvdSAMEditor.replaceImageFile(file);
                return true;
            }
            return originalDragDrop?.apply(this, arguments) ?? false;
        };
    },
});
