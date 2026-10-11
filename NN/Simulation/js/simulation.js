/**
 * Main Controller for the MNIST Neural Network Simulation Playground
 */

import { Network } from "./network.js";
import { NetworkVisualizer } from "./visualizer.js";
import { DrawingPad } from "./drawingPad.js";

class SimulationApp {
    constructor() {
        this.net = null;
        this.visualizer = null;
        this.drawingPad = null;

        // Configuration
        this.config = {
            hiddenLayers: [16, 16],
            epochs: 3,
            learningRate: 0.1,
            stepSpeed: 50, // steps per second (or delay ms)
            stepAnimation: true,
            livePredictDrawing: true
        };

        // Training state
        this.trainState = {
            running: false,
            paused: false,
            stopped: false,
            singleStepRequested: false,
            currentEpoch: 0,
            currentStep: 0,
            totalSteps: 0,
            correctCount: 0,
            totalLoss: 0
        };

        // Data
        this.dataset = {
            train: [],
            test: []
        };

        this.currentSample = null;
        this.stepHistory = [];
        this.maxHistory = 100;

        this.dom = {};
    }

    init() {
        this.cacheDOM();
        this.initData();
        this.initNetwork();
        this.initVisualizer();
        this.initDrawingPad();
        this.bindEvents();
        this.updateUI();

        // Load a random sample initially so the visualizer is active
        this.loadRandomTestSample();
    }

    cacheDOM() {
        this.dom = {
            // Badges & Headers
            statusBadge: document.getElementById("status-badge"),
            statusText: document.getElementById("status-text"),
            archDisplay: document.getElementById("arch-display"),
            neuronsCount: document.getElementById("neurons-count"),
            synapsesCount: document.getElementById("synapses-count"),

            // Config inputs
            hiddenInput: document.getElementById("hidden-layers-input"),
            presetButtons: document.querySelectorAll(".preset-btn"),
            epochsInput: document.getElementById("epochs-input"),
            lrInput: document.getElementById("lr-input"),
            speedInput: document.getElementById("speed-input"),
            speedVal: document.getElementById("speed-val"),
            animToggle: document.getElementById("anim-toggle"),
            applyArchBtn: document.getElementById("btn-apply-arch"),

            // Training Control buttons
            btnTrain: document.getElementById("btn-train"),
            btnPause: document.getElementById("btn-pause"),
            btnStop: document.getElementById("btn-stop"),
            btnStep: document.getElementById("btn-step"),
            btnTest: document.getElementById("btn-test-10"),
            btnReset: document.getElementById("btn-reset-net"),
            btnLoadModel: document.getElementById("btn-load-model"),
            btnExportModel: document.getElementById("btn-export-model"),

            // Live metrics
            trainingMetrics: document.getElementById("training-metrics"),
            metricEpoch: document.getElementById("stat-epoch"),
            metricProgress: document.getElementById("stat-progress"),
            metricBar: document.getElementById("progress-fill"),
            metricLoss: document.getElementById("stat-loss"),
            metricAvgLoss: document.getElementById("stat-avgloss"),
            metricAcc: document.getElementById("stat-acc"),
            metricStatus: document.getElementById("stat-status"),

            // Current Sample Display
            samplePreview: document.getElementById("sample-preview-canvas"),
            sampleInfoName: document.getElementById("sample-name"),
            sampleInfoTarget: document.getElementById("sample-target"),
            sampleInfoPred: document.getElementById("sample-pred"),
            sampleInfoEval: document.getElementById("sample-eval"),

            // Step Log Table
            stepLogContainer: document.getElementById("step-log-rows"),
            btnClearLog: document.getElementById("btn-clear-log"),

            // Playground Canvas
            drawCanvas: document.getElementById("draw-canvas"),
            previewCanvas: document.getElementById("preview-canvas"),
            btnBrush: document.getElementById("btn-brush"),
            btnEraser: document.getElementById("btn-eraser"),
            btnClearPad: document.getElementById("btn-clear-pad"),
            brushSizeSlider: document.getElementById("brush-size"),
            brushSizeVal: document.getElementById("brush-size-val"),
            btnCenterPad: document.getElementById("btn-center-pad"),
            invertToggle: document.getElementById("invert-toggle"),
            btnPredictDrawing: document.getElementById("btn-predict-drawing"),
            customPredResult: document.getElementById("custom-pred-result"),
            customConfidence: document.getElementById("custom-pred-conf"),

            // Quick Samples Bar
            digitButtons: document.querySelectorAll(".digit-btn"),
            btnRandomSample: document.getElementById("btn-random-sample"),
            modelFileInput: document.getElementById("model-file-input"),
            imageUploadInput: document.getElementById("image-upload-input"),

            // Visualizer Options
            toggleWeights: document.getElementById("toggle-weights"),
            filterWeakToggle: document.getElementById("filter-weak-toggle")
        };
    }

    initData() {
        if (window.MNIST_DATA) {
            this.dataset.train = window.MNIST_DATA.train || [];
            this.dataset.test = window.MNIST_DATA.test || [];
            console.log(`Loaded dataset: ${this.dataset.train.length} train, ${this.dataset.test.length} test.`);
        } else {
            console.warn("window.MNIST_DATA not found. Will run with synthetic/custom drawings.");
        }
    }

    initNetwork() {
        const fullArch = [784, ...this.config.hiddenLayers, 10];
        this.net = new Network(fullArch);
        this.updateNetworkStats();
    }

    initVisualizer() {
        const canvas = document.getElementById("network-canvas");
        this.visualizer = new NetworkVisualizer(canvas, {
            showWeights: true,
            animatePulses: true
        });
        this.visualizer.setNetwork(this.net);
        this.visualizer.resize();

        window.addEventListener("resize", () => {
            this.visualizer.resize();
        });
    }

    initDrawingPad() {
        this.drawingPad = new DrawingPad(this.dom.drawCanvas, this.dom.previewCanvas, {
            brushRadius: parseInt(this.dom.brushSizeSlider.value) || 12,
            onUpdate: (floats784) => {
                if (this.config.livePredictDrawing && !this.trainState.running) {
                    this.predictDrawingVector(floats784);
                }
            }
        });
    }

    bindEvents() {
        // Architecture Configuration
        this.dom.applyArchBtn.addEventListener("click", () => {
            const raw = this.dom.hiddenInput.value.trim();
            try {
                const parts = raw.split(",").map(s => parseInt(s.trim())).filter(n => !isNaN(n) && n > 0);
                if (parts.length === 0) throw new Error("Specify at least 1 positive hidden layer size");
                this.config.hiddenLayers = parts;
                this.initNetwork();
                this.visualizer.setNetwork(this.net);
                this.updateUI();
                this.setStatus("READY", "emerald");
                this.logStepMessage(`Architecture updated to: ${this.net.layerSizes.join(" → ")}`);
            } catch (err) {
                alert("Invalid layer sizes format. Use comma separated numbers like: 16, 16");
            }
        });

        // Presets
        this.dom.presetButtons.forEach(btn => {
            btn.addEventListener("click", () => {
                const preset = btn.dataset.arch;
                this.dom.hiddenInput.value = preset;
                this.dom.applyArchBtn.click();
            });
        });

        // Hyperparameters
        this.dom.epochsInput.addEventListener("change", (e) => {
            this.config.epochs = Math.max(1, parseInt(e.target.value) || 1);
        });

        this.dom.lrInput.addEventListener("change", (e) => {
            this.config.learningRate = Math.max(0.0001, parseFloat(e.target.value) || 0.1);
        });

        this.dom.speedInput.addEventListener("input", (e) => {
            this.config.stepSpeed = parseInt(e.target.value);
            this.dom.speedVal.textContent = `${this.config.stepSpeed} steps/s`;
        });

        this.dom.animToggle.addEventListener("change", (e) => {
            this.config.stepAnimation = e.target.checked;
        });

        // Training Buttons
        this.dom.btnTrain.addEventListener("click", () => {
            if (!this.trainState.running) {
                this.startTraining();
            }
        });

        this.dom.btnPause.addEventListener("click", () => {
            if (this.trainState.running) {
                this.trainState.paused = !this.trainState.paused;
                this.dom.btnPause.textContent = this.trainState.paused ? "▶ RESUME" : "⏸ PAUSE";
                this.setStatus(this.trainState.paused ? "PAUSED" : "TRAINING", this.trainState.paused ? "amber" : "sky");
            }
        });

        this.dom.btnStop.addEventListener("click", () => {
            if (this.trainState.running) {
                this.trainState.stopped = true;
                this.setStatus("STOPPING...", "rose");
            }
        });

        this.dom.btnStep.addEventListener("click", () => {
            if (!this.trainState.running) {
                this.trainSingleRandomStep();
            } else if (this.trainState.paused) {
                this.trainState.singleStepRequested = true;
            }
        });

        this.dom.btnTest.addEventListener("click", () => {
            this.runTestEvaluation(10);
        });

        this.dom.btnReset.addEventListener("click", () => {
            if (this.trainState.running) {
                this.trainState.stopped = true;
            }
            this.net.init();
            this.visualizer.setNetwork(this.net);
            this.setStatus("RESET", "emerald");
            this.logStepMessage("Network weights reset to Xavier random distribution.");
        });

        // Load Default Pre-trained Model
        this.dom.btnLoadModel.addEventListener("click", () => {
            if (window.DEFAULT_MODEL) {
                try {
                    this.net = Network.fromJSON(window.DEFAULT_MODEL);
                    this.config.hiddenLayers = this.net.layerSizes.slice(1, -1);
                    this.dom.hiddenInput.value = this.config.hiddenLayers.join(", ");
                    this.visualizer.setNetwork(this.net);
                    this.updateUI();
                    this.setStatus("MODEL LOADED", "emerald");
                    this.logStepMessage(`Loaded pre-trained model: ${this.net.layerSizes.join(" → ")}`);
                } catch (e) {
                    alert("Failed to load model: " + e.message);
                }
            } else {
                this.dom.modelFileInput.click();
            }
        });

        // Model File Upload
        this.dom.modelFileInput.addEventListener("change", (e) => {
            const file = e.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = (evt) => {
                try {
                    const data = JSON.parse(evt.target.result);
                    this.net = Network.fromJSON(data);
                    this.config.hiddenLayers = this.net.layerSizes.slice(1, -1);
                    this.dom.hiddenInput.value = this.config.hiddenLayers.join(", ");
                    this.visualizer.setNetwork(this.net);
                    this.updateUI();
                    this.setStatus("CUSTOM MODEL", "emerald");
                    this.logStepMessage(`Loaded model from ${file.name}`);
                } catch (err) {
                    alert("Error loading model JSON: " + err.message);
                }
            };
            reader.readAsText(file);
        });

        // Export Model JSON
        this.dom.btnExportModel.addEventListener("click", () => {
            const data = this.net.toJSON();
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `mnist_model_${this.net.layerSizes.join("_")}.json`;
            a.click();
            URL.revokeObjectURL(url);
        });

        // Clear Log
        this.dom.btnClearLog.addEventListener("click", () => {
            this.dom.stepLogContainer.innerHTML = "";
            this.stepHistory = [];
        });

        // Drawing Pad Tools
        this.dom.btnBrush.addEventListener("click", () => {
            this.drawingPad.setEraser(false);
            this.dom.btnBrush.classList.add("active");
            this.dom.btnEraser.classList.remove("active");
        });

        this.dom.btnEraser.addEventListener("click", () => {
            this.drawingPad.setEraser(true);
            this.dom.btnEraser.classList.add("active");
            this.dom.btnBrush.classList.remove("active");
        });

        this.dom.btnClearPad.addEventListener("click", () => {
            this.drawingPad.clear();
            this.dom.customPredResult.textContent = "--";
            this.dom.customConfidence.textContent = "0.0%";
        });

        this.dom.brushSizeSlider.addEventListener("input", (e) => {
            const r = parseInt(e.target.value);
            this.drawingPad.setBrushRadius(r);
            this.dom.brushSizeVal.textContent = `${r}px`;
        });

        this.dom.btnCenterPad.addEventListener("click", () => {
            this.drawingPad.update28(true);
        });

        this.dom.invertToggle.addEventListener("change", (e) => {
            this.drawingPad.setInvert(e.target.checked);
        });

        this.dom.btnPredictDrawing.addEventListener("click", () => {
            this.predictDrawingVector(this.drawingPad.getVector());
        });

        // Quick digit testers
        this.dom.digitButtons.forEach(btn => {
            btn.addEventListener("click", () => {
                const d = parseInt(btn.dataset.digit);
                this.loadSampleByDigit(d);
            });
        });

        this.dom.btnRandomSample.addEventListener("click", () => {
            this.loadRandomTestSample();
        });

        // Custom image upload
        this.dom.imageUploadInput.addEventListener("change", (e) => {
            const file = e.target.files[0];
            if (!file) return;
            const img = new Image();
            img.onload = () => {
                const off = document.createElement("canvas");
                off.width = 28;
                off.height = 28;
                const oCtx = off.getContext("2d");
                oCtx.drawImage(img, 0, 0, 28, 28);
                const id = oCtx.getImageData(0, 0, 28, 28);
                const floats = new Float32Array(784);
                for (let i = 0; i < 784; i++) {
                    const r = id.data[i * 4];
                    const g = id.data[i * 4 + 1];
                    const b = id.data[i * 4 + 2];
                    // standard grayscale luminance
                    floats[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
                }
                this.drawingPad.loadSampleVector(floats);
            };
            img.src = URL.createObjectURL(file);
        });

        // Visualizer Controls
        this.dom.toggleWeights.addEventListener("change", (e) => {
            this.visualizer.options.showWeights = e.target.checked;
            this.visualizer.render();
        });

        this.dom.filterWeakToggle.addEventListener("change", (e) => {
            this.visualizer.options.filterWeak = e.target.checked;
            this.visualizer.render();
        });
    }

    updateNetworkStats() {
        const arch = this.net.layerSizes;
        this.dom.archDisplay.textContent = arch.join(" → ");

        const totalNeurons = arch.reduce((sum, n) => sum + n, 0);
        let totalWeights = 0;
        for (let l = 0; l < arch.length - 1; l++) {
            totalWeights += arch[l] * arch[l + 1] + arch[l + 1]; // weights + biases
        }

        this.dom.neuronsCount.textContent = totalNeurons.toLocaleString();
        this.dom.synapsesCount.textContent = totalWeights.toLocaleString();
    }

    updateUI() {
        this.updateNetworkStats();
        this.dom.hiddenInput.value = this.config.hiddenLayers.join(", ");
        this.dom.epochsInput.value = this.config.epochs;
        this.dom.lrInput.value = this.config.learningRate;
        this.dom.speedInput.value = this.config.stepSpeed;
        this.dom.speedVal.textContent = `${this.config.stepSpeed} steps/s`;
    }

    setStatus(text, color = "sky") {
        this.dom.statusText.textContent = text;
        this.dom.statusBadge.className = `status-badge ${color}`;
    }

    /**
     * Feeds the custom drawn vector to the network
     */
    predictDrawingVector(vector784) {
        const output = this.net.predict(vector784);
        const pred = this.net.getPrediction(output);
        const conf = output[pred];

        this.visualizer.setInput(vector784);
        this.visualizer.triggerSignalPulse();

        this.dom.customPredResult.textContent = `${pred}`;
        this.dom.customConfidence.textContent = `${(conf * 100).toFixed(1)}%`;
    }

    /**
     * Loads a sample digit from the dataset
     */
    loadSampleByDigit(digit) {
        const pool = this.dataset.test.filter(s => s.label === digit);
        if (!pool.length) return;
        const sample = pool[Math.floor(Math.random() * pool.length)];
        this.loadSample(sample);
    }

    loadRandomTestSample() {
        const pool = this.dataset.test.length ? this.dataset.test : this.dataset.train;
        if (!pool.length) return;
        const sample = pool[Math.floor(Math.random() * pool.length)];
        this.loadSample(sample);
    }

    loadSample(sample) {
        this.currentSample = sample;
        const floats = window.MNIST_DATA.decode(sample.b64);

        // Run prediction
        const output = this.net.predict(floats);
        const pred = this.net.getPrediction(output);
        const isCorrect = pred === sample.label;
        const conf = output[pred];

        // Update visualizer
        this.visualizer.setInput(floats);
        this.visualizer.triggerSignalPulse();

        // Update drawing pad preview
        this.drawingPad.loadSampleVector(floats);

        // Update current sample panel
        this.renderSampleThumbnail(floats);
        this.dom.sampleInfoName.textContent = sample.id;
        this.dom.sampleInfoTarget.textContent = `${sample.label}`;
        this.dom.sampleInfoPred.textContent = `${pred} (${(conf * 100).toFixed(1)}%)`;

        this.dom.sampleInfoEval.innerHTML = isCorrect
            ? `<span class="badge-eval correct">✓ CORRECT</span>`
            : `<span class="badge-eval mismatch">✗ MISMATCH</span>`;
    }

    renderSampleThumbnail(floats) {
        const canvas = this.dom.samplePreview;
        const ctx = canvas.getContext("2d");
        const w = canvas.width, h = canvas.height;
        const cell = w / 28;

        ctx.clearRect(0, 0, w, h);
        for (let r = 0; r < 28; r++) {
            for (let c = 0; c < 28; c++) {
                const v = floats[r * 28 + c];
                const b = Math.floor(v * 255);
                ctx.fillStyle = `rgb(${b}, ${b}, ${b})`;
                ctx.fillRect(c * cell, r * cell, cell + 0.5, cell + 0.5);
            }
        }
    }

    /**
     * Executes one random training sample step
     */
    async trainSingleRandomStep() {
        if (!this.dataset.train.length) return;
        const sample = this.dataset.train[Math.floor(Math.random() * this.dataset.train.length)];
        const floats = window.MNIST_DATA.decode(sample.b64);

        const target = new Array(10).fill(0);
        target[sample.label] = 1;

        // Forward
        const output = this.net.predict(floats);
        const pred = this.net.getPrediction(output);
        const isCorrect = pred === sample.label;

        // Loss
        let loss = 0;
        for (let j = 0; j < output.length; j++) {
            const diff = output[j] - target[j];
            loss += diff * diff;
        }
        loss /= output.length;

        // Backpropagation
        this.net.backpropagation(target, this.config.learningRate);

        // Update UI & visualizer
        this.visualizer.setInput(floats);
        this.visualizer.triggerSignalPulse();
        this.renderSampleThumbnail(floats);

        this.dom.sampleInfoName.textContent = sample.id;
        this.dom.sampleInfoTarget.textContent = `${sample.label}`;
        this.dom.sampleInfoPred.textContent = `${pred}`;
        this.dom.sampleInfoEval.innerHTML = isCorrect
            ? `<span class="badge-eval correct">✓ CORRECT</span>`
            : `<span class="badge-eval mismatch">✗ MISMATCH</span>`;

        // Log step with check mark
        this.appendStepLogRow({
            step: this.stepHistory.length + 1,
            sampleId: sample.id,
            target: sample.label,
            prediction: pred,
            confidence: output[pred],
            loss,
            isCorrect
        });
    }

    /**
     * Main Training Loop across epochs
     */
    async startTraining() {
        if (!this.dataset.train.length) {
            alert("No training samples loaded!");
            return;
        }

        this.trainState.running = true;
        this.trainState.paused = false;
        this.trainState.stopped = false;
        this.trainState.currentStep = 0;
        this.trainState.correctCount = 0;
        this.trainState.totalLoss = 0;

        this.setStatus("TRAINING", "sky");
        this.dom.btnTrain.disabled = true;
        this.dom.btnPause.disabled = false;
        this.dom.btnStop.disabled = false;
        this.dom.btnPause.textContent = "⏸ PAUSE";

        const epochs = this.config.epochs;
        const trainData = [...this.dataset.train];
        const totalSamples = trainData.length;

        for (let ep = 1; ep <= epochs && !this.trainState.stopped; ep++) {
            this.trainState.currentEpoch = ep;
            let epochCorrect = 0;
            let epochLoss = 0;

            // Shuffle data
            for (let i = trainData.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [trainData[i], trainData[j]] = [trainData[j], trainData[i]];
            }

            for (let i = 0; i < trainData.length && !this.trainState.stopped; i++) {
                // Pause gate
                while (this.trainState.paused && !this.trainState.stopped && !this.trainState.singleStepRequested) {
                    await new Promise(r => setTimeout(r, 40));
                }
                this.trainState.singleStepRequested = false;
                if (this.trainState.stopped) break;

                const sample = trainData[i];
                const floats = window.MNIST_DATA.decode(sample.b64);

                const target = new Array(10).fill(0);
                target[sample.label] = 1;

                // Forward pass
                const output = this.net.predict(floats);
                const pred = this.net.getPrediction(output);
                const isCorrect = pred === sample.label;

                if (isCorrect) {
                    epochCorrect++;
                    this.trainState.correctCount++;
                }

                // Loss
                let sampleLoss = 0;
                for (let j = 0; j < output.length; j++) {
                    const diff = output[j] - target[j];
                    sampleLoss += diff * diff;
                }
                sampleLoss /= output.length;
                epochLoss += sampleLoss;
                this.trainState.totalLoss += sampleLoss;
                this.trainState.currentStep++;

                // Backprop
                this.net.backpropagation(target, this.config.learningRate);

                // Live step display
                const progressPct = ((i + 1) / totalSamples) * 100;
                const accuracy = (epochCorrect / (i + 1)) * 100;
                const avgLoss = epochLoss / (i + 1);

                this.dom.metricEpoch.textContent = `${ep} / ${epochs}`;
                this.dom.metricProgress.textContent = `${progressPct.toFixed(1)}% (${i + 1}/${totalSamples})`;
                this.dom.metricBar.style.width = `${progressPct}%`;
                this.dom.metricLoss.textContent = sampleLoss.toFixed(5);
                this.dom.metricAvgLoss.textContent = avgLoss.toFixed(5);
                this.dom.metricAcc.textContent = `${accuracy.toFixed(1)}%`;

                // Render step if animation is enabled or periodic
                const shouldAnimate = this.config.stepAnimation || (i % 25 === 0) || (i === totalSamples - 1);

                if (shouldAnimate) {
                    this.visualizer.setInput(floats);
                    this.renderSampleThumbnail(floats);

                    this.dom.sampleInfoName.textContent = sample.id;
                    this.dom.sampleInfoTarget.textContent = `${sample.label}`;
                    this.dom.sampleInfoPred.textContent = `${pred}`;
                    this.dom.sampleInfoEval.innerHTML = isCorrect
                        ? `<span class="badge-eval correct">✓ CORRECT</span>`
                        : `<span class="badge-eval mismatch">✗ MISMATCH</span>`;

                    // Append to step log table
                    this.appendStepLogRow({
                        step: this.trainState.currentStep,
                        sampleId: sample.id,
                        target: sample.label,
                        prediction: pred,
                        confidence: output[pred],
                        loss: sampleLoss,
                        isCorrect
                    });

                    // Throttle delay based on speed setting
                    const delayMs = Math.max(0, 1000 / this.config.stepSpeed);
                    if (delayMs > 0) {
                        await new Promise(r => setTimeout(r, delayMs));
                    }
                }
            }

            if (!this.trainState.stopped) {
                this.logStepMessage(`Epoch ${ep}/${epochs} finished | Avg Loss: ${(epochLoss / totalSamples).toFixed(5)} | Accuracy: ${((epochCorrect / totalSamples) * 100).toFixed(1)}%`);
            }
        }

        this.trainState.running = false;
        this.trainState.paused = false;
        this.dom.btnTrain.disabled = false;
        this.dom.btnPause.disabled = true;
        this.dom.btnStop.disabled = true;

        if (this.trainState.stopped) {
            this.setStatus("STOPPED", "rose");
        } else {
            this.setStatus("COMPLETED", "emerald");
        }

        this.visualizer.render();
    }

    /**
     * Appends a step to the real-time visual step table with check mark
     */
    appendStepLogRow(row) {
        const container = this.dom.stepLogContainer;
        if (!container) return;

        const tr = document.createElement("tr");
        tr.className = row.isCorrect ? "step-row correct" : "step-row mismatch";

        const statusIcon = row.isCorrect
            ? `<span class="badge-eval correct">✓ CORRECT</span>`
            : `<span class="badge-eval mismatch">✗ MISMATCH</span>`;

        tr.innerHTML = `
            <td class="code-font">#${row.step}</td>
            <td class="code-font">${row.sampleId}</td>
            <td class="code-font font-bold">${row.target}</td>
            <td class="code-font font-bold">${row.prediction}</td>
            <td class="code-font">${(row.confidence * 100).toFixed(1)}%</td>
            <td class="code-font">${row.loss.toFixed(4)}</td>
            <td>${statusIcon}</td>
        `;

        container.insertBefore(tr, container.firstChild);

        // Cap history to keep DOM lean
        if (container.children.length > this.maxHistory) {
            container.removeChild(container.lastChild);
        }
    }

    logStepMessage(msg) {
        const container = this.dom.stepLogContainer;
        if (!container) return;
        const tr = document.createElement("tr");
        tr.className = "step-row info";
        tr.innerHTML = `<td colspan="7" class="step-info-cell">ℹ ${msg}</td>`;
        container.insertBefore(tr, container.firstChild);
    }

    /**
     * Runs 10 test evaluations (mirrors CLI test from src/index.js)
     */
    async runTestEvaluation(numSamples = 10) {
        const testPool = this.dataset.test.length ? this.dataset.test : this.dataset.train;
        if (!testPool.length) return;

        this.logStepMessage(`--- RUNNING EVALUATION ON ${numSamples} RANDOM TEST IMAGES ---`);
        const shuffled = [...testPool].sort(() => 0.5 - Math.random());
        const samples = shuffled.slice(0, Math.min(numSamples, shuffled.length));

        let correct = 0;
        for (let i = 0; i < samples.length; i++) {
            const sample = samples[i];
            const floats = window.MNIST_DATA.decode(sample.b64);
            const output = this.net.predict(floats);
            const pred = this.net.getPrediction(output);
            const isCorrect = pred === sample.label;
            if (isCorrect) correct++;

            this.visualizer.setInput(floats);
            this.renderSampleThumbnail(floats);

            this.appendStepLogRow({
                step: `T-${i + 1}`,
                sampleId: sample.id,
                target: sample.label,
                prediction: pred,
                confidence: output[pred],
                loss: 0,
                isCorrect
            });

            await new Promise(r => setTimeout(r, 80));
        }

        const acc = ((correct / samples.length) * 100).toFixed(1);
        this.logStepMessage(`Test Evaluation Result: ${correct}/${samples.length} Correct (${acc}% Accuracy)`);
        this.setStatus(`ACC: ${acc}%`, "emerald");
    }
}

// Boot application
window.addEventListener("DOMContentLoaded", () => {
    const app = new SimulationApp();
    window.simulationApp = app;
    app.init();
});
