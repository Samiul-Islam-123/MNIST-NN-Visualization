/**
 * High-performance Canvas Visualizer for Neural Network Simulation
 * Renders weights (color, thickness, brightness), biases, activations,
 * input 28x28 grid, pulses, and interactive inspection tooltips.
 */

export class NetworkVisualizer {
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");
        this.options = {
            showWeights: true,
            filterWeak: false,
            colorMode: "sign", // "sign" | "magnitude" | "activations"
            animatePulses: true,
            ...options
        };

        this.net = null;
        this.currentInput = null; // Float32Array(784)
        this.hoveredNeuron = null; // { layerIndex, neuronIndex, x, y, r }
        this.hoveredConnection = null; // { l, src, dst, weight }
        this.hoveredPixel = null; // { row, col, index }

        // Layout caches
        this.layout = {
            grid: { x: 40, y: 80, size: 200, cell: 200 / 28 },
            layers: [] // array of { x, neurons: [{ x, y, r, layer, index }] }
        };

        // Signal particles for animation
        this.particles = [];
        this.animFrameId = null;

        this.bindEvents();
    }

    setNetwork(network) {
        this.net = network;
        this.recalculateLayout();
        this.render();
    }

    setInput(inputArray) {
        if (!inputArray) {
            this.currentInput = null;
        } else {
            this.currentInput = Float32Array.from(inputArray);
        }
        this.render();
    }

    triggerSignalPulse() {
        if (!this.options.animatePulses || !this.layout.layers.length) return;
        this.particles = [];
        const layers = this.layout.layers;

        // Create pulses propagating from layer to layer
        for (let l = 0; l < layers.length - 1; l++) {
            const srcLayer = layers[l];
            const dstLayer = layers[l + 1];
            const weights = this.net ? this.net.NN[2 * l + 1] : null;

            srcLayer.neurons.forEach(src => {
                dstLayer.neurons.forEach(dst => {
                    let w = 1;
                    if (weights && weights[dst.index] && weights[dst.index][src.index] !== undefined) {
                        w = weights[dst.index][src.index];
                    }
                    if (Math.abs(w) > 0.05) {
                        this.particles.push({
                            x1: src.x,
                            y1: src.y,
                            x2: dst.x,
                            y2: dst.y,
                            progress: -l * 0.25, // staggered launch
                            speed: 0.05,
                            color: w >= 0 ? "rgba(56, 189, 248, 0.9)" : "rgba(244, 63, 94, 0.9)",
                            size: Math.min(4, Math.max(1.5, Math.abs(w) * 2.5))
                        });
                    }
                });
            });
        }

        this.startAnimationLoop();
    }

    startAnimationLoop() {
        if (this.animFrameId) cancelAnimationFrame(this.animFrameId);
        const loop = () => {
            let active = false;
            for (const p of this.particles) {
                p.progress += p.speed;
                if (p.progress < 1.0) active = true;
            }
            this.render();
            if (active) {
                this.animFrameId = requestAnimationFrame(loop);
            } else {
                this.particles = [];
                this.animFrameId = null;
                this.render();
            }
        };
        this.animFrameId = requestAnimationFrame(loop);
    }

    resize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = Math.max(760, rect.width) * dpr;
        this.canvas.height = Math.max(480, rect.height) * dpr;
        this.ctx.scale(dpr, dpr);
        this.displayW = Math.max(760, rect.width);
        this.displayH = Math.max(480, rect.height);
        this.recalculateLayout();
        this.render();
    }

    recalculateLayout() {
        if (!this.net) return;
        const w = this.displayW || this.canvas.width;
        const h = this.displayH || this.canvas.height;
        const arch = this.net.layerSizes;

        const padTop = 64;
        const padBottom = 40;
        const availH = Math.max(260, h - padTop - padBottom);

        // 28x28 input grid on far left
        const gridSize = Math.min(220, Math.max(150, availH * 0.65));
        const gridX = 40;
        const gridY = padTop + (availH - gridSize) / 2;
        this.layout.grid = {
            x: gridX,
            y: gridY,
            size: gridSize,
            cell: gridSize / 28
        };

        // Layer columns layout
        // Layer 0 is represented both by the 28x28 grid and an input node connector column
        const totalColumns = arch.length; // 0: Input, 1..N-2: Hidden, N-1: Output
        const startX = gridX + gridSize + 48;
        const endX = w - 160; // leave space on right for output labels and meters
        const colSpacing = (endX - startX) / Math.max(1, totalColumns - 1);

        this.layout.layers = [];

        for (let l = 0; l < arch.length; l++) {
            const count = arch[l];
            const colX = startX + l * colSpacing;
            const neurons = [];

            // For input layer (784 nodes), display a sampled representation column
            // so connection lines connect gracefully to hidden layers
            const isInput = l === 0;
            const isOutput = l === arch.length - 1;
            const displayCount = isInput ? Math.min(28, count) : count;

            // Neuron radius based on node density
            const r = isInput
                ? 3.5
                : count <= 4
                ? 18
                : count <= 10
                ? 14
                : count <= 18
                ? 11
                : count <= 32
                ? 8
                : 5;

            const spacingY = availH / (displayCount + 1);

            for (let i = 0; i < displayCount; i++) {
                neurons.push({
                    x: colX,
                    y: padTop + (i + 1) * spacingY,
                    r,
                    layer: l,
                    index: i,
                    actualIndex: isInput ? Math.floor(i * (count / displayCount)) : i
                });
            }

            this.layout.layers.push({
                x: colX,
                layerIndex: l,
                name: isInput ? "INPUT LAYER" : isOutput ? "OUTPUT LAYER" : `HIDDEN ${l}`,
                neurons,
                totalCount: count
            });
        }
    }

    render() {
        const ctx = this.ctx;
        const w = this.displayW || this.canvas.width;
        const h = this.displayH || this.canvas.height;

        ctx.clearRect(0, 0, w, h);

        if (!this.net) {
            this.renderEmptyState(ctx, w, h);
            return;
        }

        // 1. Column Guidelines & Backdrop
        this.renderGuides(ctx, h);

        // 2. Input 28x28 Grid
        this.renderInputGrid(ctx);

        // 3. Weight Connections (color, thickness, brightness)
        if (this.options.showWeights) {
            this.renderConnections(ctx);
        }

        // 4. Signal Particle Pulses
        this.renderParticles(ctx);

        // 5. Neurons (activations as brightness, biases as rings)
        this.renderNeurons(ctx);

        // 6. Layer Headers & Output Confidence Meters
        this.renderHeadersAndMeters(ctx);

        // 7. Interactive Hover Tooltip
        this.renderTooltip(ctx);
    }

    renderEmptyState(ctx, w, h) {
        ctx.fillStyle = "#64748b";
        ctx.font = "14px 'JetBrains Mono', monospace";
        ctx.textAlign = "center";
        ctx.fillText("Neural Network Not Initialized", w / 2, h / 2);
    }

    renderGuides(ctx, h) {
        ctx.save();
        ctx.strokeStyle = "rgba(30, 41, 59, 0.4)";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 6]);

        for (const layer of this.layout.layers) {
            ctx.beginPath();
            ctx.moveTo(layer.x, 50);
            ctx.lineTo(layer.x, h - 30);
            ctx.stroke();
        }
        ctx.restore();
    }

    renderInputGrid(ctx) {
        const grid = this.layout.grid;
        const input = this.currentInput;

        ctx.save();
        // Frame
        ctx.strokeStyle = "rgba(56, 189, 248, 0.3)";
        ctx.lineWidth = 1.5;
        ctx.fillStyle = "#090d16";
        ctx.strokeRect(grid.x - 3, grid.y - 3, grid.size + 6, grid.size + 6);
        ctx.fillRect(grid.x - 3, grid.y - 3, grid.size + 6, grid.size + 6);

        // 28x28 Pixels
        for (let r = 0; r < 28; r++) {
            for (let c = 0; c < 28; c++) {
                const idx = r * 28 + c;
                const v = input ? Math.min(1, Math.max(0, input[idx])) : 0;
                const px = grid.x + c * grid.cell;
                const py = grid.y + r * grid.cell;

                if (v > 0.01) {
                    // Brightness scales with activation
                    const bright = Math.floor(v * 255);
                    ctx.fillStyle = `rgb(${bright}, ${bright}, ${Math.floor(bright * 1.05)})`;
                } else {
                    ctx.fillStyle = "#0c101c";
                }
                ctx.fillRect(px, py, grid.cell + 0.5, grid.cell + 0.5);

                // Highlight hovered pixel
                if (this.hoveredPixel && this.hoveredPixel.index === idx) {
                    ctx.strokeStyle = "#fbbf24";
                    ctx.lineWidth = 1.5;
                    ctx.strokeRect(px, py, grid.cell, grid.cell);
                }
            }
        }

        // Subtitle
        ctx.fillStyle = "#94a3b8";
        ctx.font = "11px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("28×28 MNIST Input Vector (784)", grid.x + grid.size / 2, grid.y + grid.size + 20);
        ctx.restore();
    }

    renderConnections(ctx) {
        const layers = this.layout.layers;
        const weights = this.net.extractWeights();
        const hovered = this.hoveredNeuron;
        const isHovering = !!hovered;

        ctx.save();

        for (let l = 0; l < layers.length - 1; l++) {
            const srcLayer = layers[l];
            const dstLayer = layers[l + 1];
            const weightMatrix = weights[l]; // dstCount x srcCount
            if (!weightMatrix) continue;

            const isInputToHidden = l === 0;

            for (let j = 0; j < dstLayer.neurons.length; j++) {
                const dst = dstLayer.neurons[j];
                const dstIndex = dst.index;
                const wRow = weightMatrix[dstIndex];
                if (!wRow) continue;

                for (let i = 0; i < srcLayer.neurons.length; i++) {
                    const src = srcLayer.neurons[i];
                    const srcIndex = isInputToHidden ? src.actualIndex : src.index;
                    const w = wRow[srcIndex] || 0;

                    const absW = Math.abs(w);
                    if (this.options.filterWeak && absW < 0.15) continue;

                    // Hover logic: highlight connected, dim others
                    const isLinked = hovered && (
                        (hovered.layerIndex === l && hovered.neuronIndex === src.index) ||
                        (hovered.layerIndex === l + 1 && hovered.neuronIndex === dst.index)
                    );

                    let alpha, thickness;
                    if (isHovering) {
                        if (isLinked) {
                            alpha = Math.min(1.0, 0.4 + absW * 0.7);
                            thickness = Math.min(4.5, 1.2 + absW * 2.5);
                        } else {
                            alpha = 0.04;
                            thickness = 0.5;
                        }
                    } else {
                        // Normal mode: thickness and brightness based on weight magnitude
                        alpha = Math.min(0.85, Math.max(0.12, absW * 0.65));
                        thickness = isInputToHidden
                            ? Math.min(2.0, Math.max(0.4, absW * 1.2))
                            : Math.min(3.8, Math.max(0.7, absW * 2.2));
                    }

                    // Color: Cyan/Sky-Blue for positive (w >= 0), Coral/Rose for negative (w < 0)
                    if (w >= 0) {
                        ctx.strokeStyle = `rgba(56, 189, 248, ${alpha})`;
                    } else {
                        ctx.strokeStyle = `rgba(244, 63, 94, ${alpha})`;
                    }

                    ctx.lineWidth = thickness;
                    ctx.beginPath();
                    ctx.moveTo(src.x, src.y);
                    ctx.lineTo(dst.x, dst.y);
                    ctx.stroke();
                }
            }
        }

        ctx.restore();
    }

    renderParticles(ctx) {
        if (!this.particles.length) return;
        ctx.save();
        for (const p of this.particles) {
            if (p.progress >= 0 && p.progress <= 1) {
                const curX = p.x1 + (p.x2 - p.x1) * p.progress;
                const curY = p.y1 + (p.y2 - p.y1) * p.progress;

                ctx.fillStyle = p.color;
                ctx.shadowColor = p.color;
                ctx.shadowBlur = 8;
                ctx.beginPath();
                ctx.arc(curX, curY, p.size, 0, Math.PI * 2);
                ctx.fill();
            }
        }
        ctx.restore();
    }

    renderNeurons(ctx) {
        const layers = this.layout.layers;
        const activations = this.net.extractActivations();
        const biases = this.net.extractBiases();
        const lastLayerIdx = layers.length - 1;
        const outputActs = activations[lastLayerIdx] || [];
        const winnerIdx = this.net.getPrediction(outputActs);

        ctx.save();

        for (let l = 0; l < layers.length; l++) {
            const layer = layers[l];
            const isOutput = l === lastLayerIdx;
            const isInput = l === 0;
            const layerActs = activations[l] || [];
            const layerBiases = l > 0 ? biases[l - 1] : null;

            for (const n of layer.neurons) {
                const act = isInput
                    ? (this.currentInput ? this.currentInput[n.actualIndex] || 0 : 0)
                    : (layerActs[n.index] || 0);

                const bias = layerBiases ? (layerBiases[n.index] || 0) : 0;
                const isHovered = this.hoveredNeuron &&
                    this.hoveredNeuron.layerIndex === l &&
                    this.hoveredNeuron.neuronIndex === n.index;

                const isWinner = isOutput && n.index === winnerIdx && outputActs[winnerIdx] > 0;

                // 1. Halo Glow (scaled by activation)
                if (act > 0.1 || isWinner) {
                    const glowRadius = n.r * (isWinner ? 2.6 : 1.8);
                    const glowColor = isWinner
                        ? `rgba(251, 191, 36, ${Math.min(0.6, act * 0.7)})`
                        : `rgba(56, 189, 248, ${Math.min(0.5, act * 0.6)})`;

                    const grad = ctx.createRadialGradient(n.x, n.y, n.r * 0.5, n.x, n.y, glowRadius);
                    grad.addColorStop(0, glowColor);
                    grad.addColorStop(1, "rgba(0, 0, 0, 0)");
                    ctx.fillStyle = grad;
                    ctx.beginPath();
                    ctx.arc(n.x, n.y, glowRadius, 0, Math.PI * 2);
                    ctx.fill();
                }

                // 2. Neuron Body
                // Base background + activation brightness fill
                ctx.beginPath();
                ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);

                if (isWinner) {
                    ctx.fillStyle = "#fbbf24";
                } else if (isInput) {
                    const b = Math.floor(act * 255);
                    ctx.fillStyle = `rgb(${b}, ${b}, ${Math.floor(b * 1.1)})`;
                } else {
                    // Activation brightness
                    const rCol = Math.floor(20 + act * 36);
                    const gCol = Math.floor(30 + act * 159);
                    const bCol = Math.floor(50 + act * 205);
                    ctx.fillStyle = `rgb(${rCol}, ${gCol}, ${bCol})`;
                }
                ctx.fill();

                // 3. Bias Ring on Hidden & Output Neurons
                if (!isInput) {
                    ctx.lineWidth = isHovered ? 2.8 : 1.6;
                    if (isWinner) {
                        ctx.strokeStyle = "#ffffff";
                    } else if (bias >= 0) {
                        // Positive bias: soft emerald/cyan ring
                        ctx.strokeStyle = `rgba(52, 211, 153, ${Math.min(1.0, 0.4 + Math.abs(bias) * 0.5)})`;
                    } else {
                        // Negative bias: soft amber/orange ring
                        ctx.strokeStyle = `rgba(251, 146, 60, ${Math.min(1.0, 0.4 + Math.abs(bias) * 0.5)})`;
                    }
                    ctx.stroke();
                }

                // 4. Node Label
                if (n.r >= 10) {
                    ctx.fillStyle = isWinner ? "#090d16" : "#ffffff";
                    ctx.font = `${isWinner ? "bold 12px" : "10px"} Inter, sans-serif`;
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    const label = isOutput ? `${n.index}` : `${n.index}`;
                    ctx.fillText(label, n.x, n.y);
                }
            }
        }

        ctx.restore();
    }

    renderHeadersAndMeters(ctx) {
        ctx.save();

        // Layer titles
        for (const layer of this.layout.layers) {
            ctx.fillStyle = "#f8fafc";
            ctx.font = "bold 12px 'Outfit', Inter, sans-serif";
            ctx.textAlign = "center";
            ctx.fillText(layer.name, layer.x, 26);

            ctx.fillStyle = "#64748b";
            ctx.font = "10px 'JetBrains Mono', monospace";
            ctx.fillText(`${layer.totalCount} nodes`, layer.x, 42);
        }

        // Output Layer Probability Meters on the right
        const lastLayer = this.layout.layers[this.layout.layers.length - 1];
        if (lastLayer && lastLayer.layerIndex === this.net.layerSizes.length - 1) {
            const activations = this.net.extractActivations();
            const outputActs = activations[activations.length - 1] || [];
            const winnerIdx = this.net.getPrediction(outputActs);

            for (const n of lastLayer.neurons) {
                const prob = outputActs[n.index] || 0;
                const isWinner = n.index === winnerIdx && prob > 0;
                const barX = n.x + n.r + 14;
                const barY = n.y - 4;
                const barW = 75;
                const barH = 8;

                // Probability Bar Track
                ctx.fillStyle = "rgba(30, 41, 59, 0.8)";
                ctx.fillRect(barX, barY, barW, barH);

                // Probability Bar Fill
                ctx.fillStyle = isWinner ? "#fbbf24" : "rgba(56, 189, 248, 0.85)";
                ctx.fillRect(barX, barY, Math.max(2, barW * prob), barH);

                // Percentage Label
                ctx.fillStyle = isWinner ? "#fbbf24" : "#94a3b8";
                ctx.font = `${isWinner ? "bold" : "normal"} 10px 'JetBrains Mono', monospace`;
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                ctx.fillText(`${(prob * 100).toFixed(1)}%`, barX + barW + 8, n.y);
            }
        }

        ctx.restore();
    }

    renderTooltip(ctx) {
        if (!this.hoveredNeuron) return;
        const n = this.hoveredNeuron;
        const isInput = n.layerIndex === 0;
        const isOutput = n.layerIndex === this.layout.layers.length - 1;

        const acts = this.net.extractActivations();
        const biases = this.net.extractBiases();
        const act = isInput
            ? (this.currentInput ? this.currentInput[n.actualIndex] || 0 : 0)
            : (acts[n.layerIndex] ? acts[n.layerIndex][n.neuronIndex] || 0 : 0);

        const bias = (!isInput && biases[n.layerIndex - 1])
            ? biases[n.layerIndex - 1][n.neuronIndex] || 0
            : null;

        const lines = [
            isInput
                ? `Input Node #${n.actualIndex}`
                : isOutput
                ? `Output Digit [${n.neuronIndex}]`
                : `Hidden Node L${n.layerIndex} #${n.neuronIndex}`,
            `Activation: ${act.toFixed(4)}`
        ];

        if (bias !== null) {
            lines.push(`Bias: ${bias >= 0 ? "+" : ""}${bias.toFixed(4)}`);
        }
        if (isOutput) {
            lines.push(`Confidence: ${(act * 100).toFixed(1)}%`);
        }

        ctx.save();
        const padX = 10, padY = 8, lineH = 16;
        const boxW = 165;
        const boxH = lines.length * lineH + padY * 2;
        let boxX = n.x + n.r + 10;
        let boxY = n.y - boxH / 2;

        if (boxX + boxW > (this.displayW || this.canvas.width) - 10) {
            boxX = n.x - n.r - boxW - 10;
        }

        ctx.fillStyle = "rgba(15, 23, 42, 0.95)";
        ctx.strokeStyle = "rgba(56, 189, 248, 0.5)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(boxX, boxY, boxW, boxH, 6);
        ctx.fill();
        ctx.stroke();

        ctx.font = "11px Inter, sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "top";

        lines.forEach((line, i) => {
            ctx.fillStyle = i === 0 ? "#fbbf24" : "#e2e8f0";
            ctx.fillText(line, boxX + padX, boxY + padY + i * lineH);
        });

        ctx.restore();
    }

    bindEvents() {
        const getPos = (e) => {
            const rect = this.canvas.getBoundingClientRect();
            return {
                x: e.clientX - rect.left,
                y: e.clientY - rect.top
            };
        };

        this.canvas.addEventListener("mousemove", (e) => {
            const { x, y } = getPos(e);
            let foundNeuron = null;
            let foundPixel = null;

            // Check neurons
            for (const layer of this.layout.layers) {
                for (const n of layer.neurons) {
                    const dist = Math.hypot(x - n.x, y - n.y);
                    if (dist <= n.r + 4) {
                        foundNeuron = {
                            layerIndex: layer.layerIndex,
                            neuronIndex: n.index,
                            actualIndex: n.actualIndex,
                            x: n.x,
                            y: n.y,
                            r: n.r
                        };
                        break;
                    }
                }
                if (foundNeuron) break;
            }

            // Check input 28x28 grid
            const grid = this.layout.grid;
            if (x >= grid.x && x < grid.x + grid.size && y >= grid.y && y < grid.y + grid.size) {
                const col = Math.floor((x - grid.x) / grid.cell);
                const row = Math.floor((y - grid.y) / grid.cell);
                if (col >= 0 && col < 28 && row >= 0 && row < 28) {
                    foundPixel = { row, col, index: row * 28 + col };
                }
            }

            if (
                foundNeuron !== this.hoveredNeuron ||
                foundPixel !== this.hoveredPixel
            ) {
                this.hoveredNeuron = foundNeuron;
                this.hoveredPixel = foundPixel;
                this.render();
            }
        });

        this.canvas.addEventListener("mouseleave", () => {
            if (this.hoveredNeuron || this.hoveredPixel) {
                this.hoveredNeuron = null;
                this.hoveredPixel = null;
                this.render();
            }
        });
    }
}
