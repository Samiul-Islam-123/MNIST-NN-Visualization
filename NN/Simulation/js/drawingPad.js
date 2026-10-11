/**
 * Interactive Drawing Pad with real-time 28×28 downsampling and normalization
 */

export class DrawingPad {
    constructor(drawCanvas, previewCanvas, options = {}) {
        this.canvas = drawCanvas;
        this.ctx = drawCanvas.getContext("2d");
        this.previewCanvas = previewCanvas;
        this.previewCtx = previewCanvas.getContext("2d");

        this.options = {
            size: 280,
            brushRadius: 12,
            autoCenter: true,
            invert: false,
            onUpdate: null, // (floatArray784) => void
            ...options
        };

        this.isDrawing = false;
        this.isEraser = false;
        this.lastX = 0;
        this.lastY = 0;
        this.currentVector = new Float32Array(784);

        // Offscreen 28x28 canvas for crisp sampling
        this.off28 = document.createElement("canvas");
        this.off28.width = 28;
        this.off28.height = 28;
        this.off28Ctx = this.off28.getContext("2d", { willReadFrequently: true });

        this.init();
        this.bindEvents();
    }

    init() {
        this.canvas.width = this.options.size;
        this.canvas.height = this.options.size;
        this.clear();
    }

    clear() {
        // MNIST digits are white on black
        this.ctx.fillStyle = "#000000";
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.update28();
    }

    setBrushRadius(r) {
        this.options.brushRadius = Math.max(3, Math.min(32, r));
    }

    setEraser(isEraser) {
        this.isEraser = isEraser;
    }

    setInvert(invert) {
        this.options.invert = invert;
        this.update28();
    }

    setAutoCenter(autoCenter) {
        this.options.autoCenter = autoCenter;
        this.update28();
    }

    bindEvents() {
        const getPos = (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            const clientY = e.touches ? e.touches[0].clientY : e.clientY;
            return {
                x: (clientX - rect.left) * (this.canvas.width / rect.width),
                y: (clientY - rect.top) * (this.canvas.height / rect.height)
            };
        };

        const start = (e) => {
            e.preventDefault();
            this.isDrawing = true;
            const pos = getPos(e);
            this.lastX = pos.x;
            this.lastY = pos.y;
            this.drawDot(pos.x, pos.y);
            this.update28();
        };

        const move = (e) => {
            if (!this.isDrawing) return;
            e.preventDefault();
            const pos = getPos(e);
            this.drawLine(this.lastX, this.lastY, pos.x, pos.y);
            this.lastX = pos.x;
            this.lastY = pos.y;
            this.update28();
        };

        const end = (e) => {
            if (this.isDrawing) {
                this.isDrawing = false;
                this.update28(true); // final center pass
            }
        };

        this.canvas.addEventListener("mousedown", start);
        window.addEventListener("mousemove", move);
        window.addEventListener("mouseup", end);

        this.canvas.addEventListener("touchstart", start, { passive: false });
        window.addEventListener("touchmove", move, { passive: false });
        window.addEventListener("touchend", end);
    }

    drawDot(x, y) {
        this.ctx.save();
        this.ctx.fillStyle = this.isEraser ? "#000000" : "#ffffff";
        this.ctx.beginPath();
        this.ctx.arc(x, y, this.options.brushRadius, 0, Math.PI * 2);
        this.ctx.fill();
        this.ctx.restore();
    }

    drawLine(x1, y1, x2, y2) {
        this.ctx.save();
        this.ctx.strokeStyle = this.isEraser ? "#000000" : "#ffffff";
        this.ctx.lineWidth = this.options.brushRadius * 2;
        this.ctx.lineCap = "round";
        this.ctx.lineJoin = "round";
        this.ctx.beginPath();
        this.ctx.moveTo(x1, y1);
        this.ctx.lineTo(x2, y2);
        this.ctx.stroke();
        this.ctx.restore();
    }

    /**
     * Center of mass centering matching MNIST standard preprocessing
     */
    centerImage(imgData) {
        const w = 28, h = 28;
        const d = imgData.data;

        let totalMass = 0;
        let xSum = 0, ySum = 0;

        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const idx = (y * w + x) * 4;
                const v = d[idx]; // grayscale R channel
                if (v > 10) {
                    totalMass += v;
                    xSum += x * v;
                    ySum += y * v;
                }
            }
        }

        if (totalMass === 0) return imgData;

        const cX = xSum / totalMass;
        const cY = ySum / totalMass;
        const shiftX = Math.round(14 - cX);
        const shiftY = Math.round(14 - cY);

        if (shiftX === 0 && shiftY === 0) return imgData;

        // Shift onto new canvas
        const temp = document.createElement("canvas");
        temp.width = 28;
        temp.height = 28;
        const tempCtx = temp.getContext("2d");
        tempCtx.putImageData(imgData, 0, 0);

        this.off28Ctx.fillStyle = "#000000";
        this.off28Ctx.fillRect(0, 0, 28, 28);
        this.off28Ctx.drawImage(temp, shiftX, shiftY);

        return this.off28Ctx.getImageData(0, 0, 28, 28);
    }

    update28(finalPass = false) {
        // Step 1: Draw high-res canvas downscaled to 28x28
        this.off28Ctx.fillStyle = "#000000";
        this.off28Ctx.fillRect(0, 0, 28, 28);
        this.off28Ctx.drawImage(this.canvas, 0, 0, 28, 28);

        let imgData = this.off28Ctx.getImageData(0, 0, 28, 28);

        // Center on final pass if enabled
        if (finalPass && this.options.autoCenter) {
            imgData = this.centerImage(imgData);
        }

        // Step 2: Extract normalized 784 float values
        const pixels = imgData.data;
        const floats = new Float32Array(784);

        for (let i = 0; i < 784; i++) {
            let val = pixels[i * 4] / 255.0; // 0.0 to 1.0
            if (this.options.invert) val = 1.0 - val;
            floats[i] = val;
        }

        this.currentVector = floats;

        // Step 3: Render to visible 28x28 preview canvas
        this.renderPreview(floats);

        if (this.options.onUpdate) {
            this.options.onUpdate(floats);
        }
    }

    renderPreview(floats) {
        const pw = this.previewCanvas.width;
        const ph = this.previewCanvas.height;
        const cell = pw / 28;

        this.previewCtx.clearRect(0, 0, pw, ph);

        for (let r = 0; r < 28; r++) {
            for (let c = 0; c < 28; c++) {
                const idx = r * 28 + c;
                const v = floats[idx];
                const b = Math.floor(v * 255);
                this.previewCtx.fillStyle = `rgb(${b}, ${b}, ${b})`;
                this.previewCtx.fillRect(c * cell, r * cell, cell + 0.5, cell + 0.5);
            }
        }
    }

    loadSampleVector(vector784) {
        // Load external vector into drawing pad
        this.currentVector = Float32Array.from(vector784);

        // Render onto 28x28 offscreen then upscale to 280x280 pad
        const imgData = this.off28Ctx.createImageData(28, 28);
        for (let i = 0; i < 784; i++) {
            const v = Math.floor(this.currentVector[i] * 255);
            imgData.data[i * 4] = v;
            imgData.data[i * 4 + 1] = v;
            imgData.data[i * 4 + 2] = v;
            imgData.data[i * 4 + 3] = 255;
        }
        this.off28Ctx.putImageData(imgData, 0, 0);

        // Upscale onto drawing pad
        this.ctx.imageSmoothingEnabled = false;
        this.ctx.drawImage(this.off28, 0, 0, this.canvas.width, this.canvas.height);

        this.renderPreview(this.currentVector);

        if (this.options.onUpdate) {
            this.options.onUpdate(this.currentVector);
        }
    }

    getVector() {
        return this.currentVector;
    }
}
