/**
 * ============================================================================
 * MNIST Neural Network Training Visualizer + Playground
 * p5.js + Chart.js + Socket.IO — runs the REAL NN/Network.js in the browser
 * ============================================================================
 */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let ARCHITECTURE = [784, 3, 3, 10];
const LAYER_NAMES = ["INPUT LAYER", "HIDDEN LAYER", "HIDDEN LAYER", "OUTPUT LAYER"];

let networkState = null;   // { architecture, weights, biases, activations, gradients }
let inputActs = null;      // Float32Array(784) | null — current sample pixels
let stats = null;          // latest training stats from server
let trainState = { running: false, paused: false };
let prepState = { ready: 0, total: 0, done: false };

let maxAbsW = [1, 1, 1];
let maxAbsG = [1, 1, 1];
let weightDeltas = null;   // per-layer nested deltas from last server update
let maxAbsDelta = 0;

let socket = null;
let clientNet = null;      // browser-side Network instance (real NN/Network.js)
let playSource = "server"; // server | local
let lastArchKey = "";

// Layout caches
let inputCells = [];
let inputGrid = null;
let layerX = [0];
let neuronPos = [];        // per-layer neuron positions, index 1..n-1

// Hover
let hoveredNeuron = null;
let hoveredCell = -1;
let hoveredConn = null;

// Color mode
let colorMode = "activation";
let showWeights = true;

// Charts
let lossChart = null, accChart = null, epochChart = null, predChart = null, playChart = null;
let pendingPoints = [];
let lossRaw = [];          // { x, y, avg, ema } for EMA/log rebuilds
let lossEMA = null;
let lastFlush = 0;
const MAX_POINTS = 800;

// Playground
const pad = {
  canvas: null, ctx: null,
  size: 280, cell: 10,
  brushR: 13,
  drawing: false, tool: "brush", invert: false,
  last: null
};
let previewCanvas = null, previewCtx = null;
let off28 = null, off28Ctx = null;
let currentVector = null;  // Float32Array(784)
let lastOut = null;        // last prediction outputs
let playGradients = null;
let playStepIdx = 0;
let playTarget = -1;

// DOM
let els = {};

// ============================================================================
// SOCKET.IO
// ============================================================================
function connectToServer() {
  socket = io({ transports: ["websocket", "polling"] });

  socket.on("connect", () => {
    setMode("READY", "#22c55e");
  });

  socket.on("disconnect", () => {
    setMode("DISCONNECTED", "#f43f5e");
  });

  socket.on("model:loaded", (data) => {
    networkState = data;
    if (data.architecture) ARCHITECTURE = data.architecture;
    maxAbsW = computeMaxAbs(networkState.weights);
    maxAbsG = networkState.gradients ? computeMaxAbs(networkState.gradients) : [1, 1, 1];
    weightDeltas = data.weightDeltas || null;
    maxAbsDelta = data.maxAbsDelta || 0;
    syncClientFromState(data);
    maybeRelayout();
    updateArchMetrics();

    if (data.reset || data.source === "upload") {
      inputActs = null;
      stats = null;
      playGradients = null;
      clearCharts();
      clearPlayChart();
      playStepIdx = 0;
      setMode("READY", "#22c55e");
      if (els.liveStats) els.liveStats.style.display = "none";
      if (els.progressWrap) els.progressWrap.style.display = "none";
    }
    if (data.source === "upload") {
      setPlaySource("local");
      playInfo(`uploaded model loaded — ${ARCHITECTURE.join("→")}`);
    } else if (data.reset) {
      setPlaySource("server");
      playInfo("server reset to fresh random weights");
    } else {
      setPlaySource("server");
    }
    applyStateToDOM();
  });

  socket.on("model:error", (e) => {
    playInfo(`<span class="bad">upload failed: ${escapeHtml(e.message)}</span>`);
  });

  socket.on("network:update", (s) => {
    networkState = s;
    if (s.input) inputActs = Float32Array.from(s.input);
    stats = s;
    maxAbsW = computeMaxAbs(s.weights);
    if (s.gradients) maxAbsG = computeMaxAbs(s.gradients);
    weightDeltas = s.weightDeltas || null;
    maxAbsDelta = s.maxAbsDelta || 0;
    setPlaySource("server");
    applyStateToDOM();

    pendingPoints.push({
      x: s.globalStep,
      y: s.loss,
      avg: s.avgLoss,
      acc: s.accuracy
    });
    const now = millis();
    if (now - lastFlush > 120) flushCharts();
  });

  socket.on("training:epoch", (e) => {
    flushCharts();
    epochChart.data.labels.push("E" + e.epoch);
    epochChart.data.datasets[0].data.push(e.loss);
    epochChart.data.datasets[1].data.push(e.accuracy);
    epochChart.update("none");
  });

  socket.on("training:state", (t) => {
    trainState = t;
    updateControlButtons();
    if (t.running && !t.paused) {
      setMode(t.turbo ? "TURBO" : "TRAINING", t.turbo ? "#f472b6" : "#38bdf8");
      setPlaySource("server");
    }
    else if (t.running && t.paused) setMode("PAUSED", "#fbbf24");
  });

  socket.on("training:done", (d) => {
    trainState = { running: false, paused: false };
    updateControlButtons();
    setMode(stats ? "COMPLETE" : "READY", "#22c55e");
    flushCharts();
    if (d && d.rawWeights) {
      syncClientFromState({ architecture: ARCHITECTURE, weights: d.rawWeights, biases: d.rawBiases });
    }
  });

  socket.on("data:prep", (p) => {
    prepState = p;
    if (!p.done && !trainState.running) {
      setMode(`PREP ${p.ready}/${p.total}`, "#fbbf24");
    }
  });
}

function syncClientFromState(s) {
  if (!window.Network) return;
  const w = s.rawWeights || s.weights;
  const b = s.rawBiases || s.biases;
  if (!w || !b || !s.architecture) return;
  try {
    clientNet = window.Network.fromJSON({ architecture: s.architecture, weights: w, biases: b });
    lastArchKey = s.architecture.join(",");
  } catch (e) {
    console.warn("clientNet sync failed:", e.message);
  }
}

function computeMaxAbs(matrices) {
  // handles both nested weight matrices (array of rows) and flat gradient arrays
  return matrices.map((m) => {
    let mx = 1e-6;
    for (const row of m) {
      if (typeof row === "number") {
        const a = Math.abs(row);
        if (a > mx) mx = a;
        continue;
      }
      for (const v of row) { const a = Math.abs(v); if (a > mx) mx = a; }
    }
    return mx;
  });
}

// ============================================================================
// LAYOUT (dynamic architecture, input always a 28x28 grid)
// ============================================================================
function currentArch() {
  if (networkState && networkState.architecture) return networkState.architecture;
  if (clientNet && clientNet.layerSizes) return clientNet.layerSizes;
  return ARCHITECTURE;
}

function maybeRelayout() {
  const key = currentArch().join(",");
  if (key !== lastArchKey) {
    lastArchKey = key;
    if (typeof width === "number") calculateLayout(width, height);
  }
}

function calculateLayout(w, h) {
  const arch = currentArch();
  const marginT = 54;
  const marginB = 28;
  const availH = Math.max(220, h - marginT - marginB);

  // --- input grid on the left ---
  const gridMax = Math.min(availH, 320, w * 0.32);
  const cell = gridMax / 28;
  const gridW = cell * 28;
  const gridX = 52;
  const gridY = marginT + (availH - gridW) / 2;
  inputGrid = { x: gridX, y: gridY, size: gridW, cell };

  inputCells = [];
  for (let i = 0; i < 784; i++) {
    const r = (i / 28) | 0, c = i % 28;
    inputCells.push({
      x: gridX + c * cell,
      y: gridY + r * cell,
      w: cell,
      layer: 0,
      index: i
    });
  }

  // --- layer columns: arch.length - 1 of them ---
  const layerCount = arch.length - 1; // hidden + output columns
  const rightPad = Math.max(150, w * 0.15);
  const gridRight = gridX + gridW;
  const usable = Math.max(220, w - gridRight - rightPad);
  layerX = [0];
  for (let li = 1; li <= layerCount; li++) {
    layerX.push(gridRight + usable * (li / (layerCount + 0.15)));
  }

  // --- neuron columns ---
  neuronPos = [[]];
  for (let li = 1; li < arch.length; li++) {
    const count = arch[li];
    const r = count <= 4
      ? Math.min(24, Math.max(15, availH * 0.03))
      : count <= 16
        ? Math.min(17, Math.max(11, availH * 0.02))
        : 9;
    const spacing = availH / (count + 1);
    const col = [];
    for (let i = 0; i < count; i++) {
      col.push({
        x: layerX[li],
        y: marginT + (i + 1) * spacing,
        r,
        layer: li,
        index: i
      });
    }
    neuronPos.push(col);
  }
  lastArchKey = arch.join(",");
}

// ============================================================================
// P5 SETUP / DRAW
// ============================================================================
function setup() {
  const container = document.getElementById("canvas-container");
  const initW = Math.max(container.clientWidth || 900, 700);
  const initH = Math.max(container.clientHeight || 620, 500);
  const canvas = createCanvas(initW, initH);
  canvas.parent("canvas-container");
  canvas.elt.style.touchAction = "none";

  calculateLayout(width, height);
  initUI();
  initCharts();
  initPlayground();
  connectToServer();

  if (window.Network) initClientNet();
  else window.addEventListener("netlib-ready", initClientNet, { once: true });

  frameRate(60);
}

function initClientNet() {
  if (!networkState) {
    clientNet = new window.Network(ARCHITECTURE);
    lastArchKey = ARCHITECTURE.join(",");
  } else {
    syncClientFromState(networkState);
  }
  updateArchMetrics();
  setPlayInfoReady();
}

function windowResized() {
  const container = document.getElementById("canvas-container");
  if (!container) return;
  const newW = Math.max(container.clientWidth, 700);
  const newH = Math.max(container.clientHeight, 500);
  resizeCanvas(newW, newH);
  calculateLayout(newW, newH);
}

function draw() {
  background(8, 10, 15);

  handleHover();

  drawColumnGuides();
  drawInputGrid();
  drawConnections();
  drawNeurons();
  drawLayerLabels();
}

// ---------------------------------------------------------------------------
// Background guides
// ---------------------------------------------------------------------------
function drawColumnGuides() {
  push();
  stroke(24, 32, 48, 80);
  strokeWeight(1);
  drawingContext.setLineDash([4, 6]);
  for (let l = 1; l < layerX.length; l++) {
    line(layerX[l], 46, layerX[l], height - 20);
  }
  drawingContext.setLineDash([]);
  if (inputGrid) {
    noFill();
    stroke(30, 41, 59, 160);
    strokeWeight(1);
    rect(inputGrid.x - 4, inputGrid.y - 4, inputGrid.size + 8, inputGrid.size + 8, 3);
  }
  pop();
}

// ---------------------------------------------------------------------------
// Input grid (28x28) — brightness = activation
// ---------------------------------------------------------------------------
function drawInputGrid() {
  if (!inputCells.length) return;
  push();
  noStroke();
  for (let i = 0; i < 784; i++) {
    const c = inputCells[i];
    const v = inputActs ? Math.min(1, Math.max(0, inputActs[i])) : 0;
    const g = lerp(10, 235, v);
    fill(g * 0.92, g * 0.96, g);
    rect(c.x, c.y, c.w + 0.5, c.w + 0.5);
  }
  if (hoveredCell >= 0) {
    const c = inputCells[hoveredCell];
    noFill();
    stroke(251, 191, 36);
    strokeWeight(1.5);
    rect(c.x - 0.5, c.y - 0.5, c.w + 1, c.w + 1);
  }
  pop();
}

// ---------------------------------------------------------------------------
// Connections — color & thickness from weight value
// ---------------------------------------------------------------------------
function connectionStyle(weight, norm, isHover, linkedToHover, layerIdx, delta) {
  if (isHover) return { stroke: [255, 255, 255, 240], sw: 3 };

  // Weight-change mode: brightness = |Δw|, color = Δw sign
  if (colorMode === "change" && delta !== undefined && delta !== null) {
    const mag = maxAbsDelta > 0 ? Math.min(Math.abs(delta) / maxAbsDelta, 1) : 0;
    const grew = delta >= 0;
    const rgb = grew ? [34, 197, 94] : [244, 63, 94]; // green up / red down
    const baseAlpha = layerIdx === 0
      ? map(mag, 0, 1, 4, 70)
      : layerIdx === 1
        ? map(mag, 0, 1, 20, 220)
        : map(mag, 0, 1, 15, 200);
    const baseSw = layerIdx === 0
      ? map(mag, 0, 1, 0.3, 1.2)
      : map(mag, 0, 1, 0.8, 4);
    let alpha = baseAlpha;
    let sw = baseSw;
    if (linkedToHover) {
      alpha = Math.max(baseAlpha, 200);
      sw = Math.max(baseSw, 2);
    } else if (hoveredNeuron || hoveredCell >= 0) {
      alpha = 6;
      sw = 0.35;
    }
    return { stroke: [rgb[0], rgb[1], rgb[2], alpha], sw };
  }

  const strength = Math.min(Math.abs(weight) / norm, 1);
  const pos = weight >= 0;

  const base = layerIdx === 0
    ? [map(strength, 0, 1, 5, 90), map(strength, 0, 1, 0.4, 1.3)]
    : layerIdx === 1
      ? [map(strength, 0, 1, 30, 210), map(strength, 0, 1, 1, 4)]
      : [map(strength, 0, 1, 25, 200), map(strength, 0, 1, 0.8, 3.2)];

  let alpha, sw;
  if (linkedToHover) {
    alpha = Math.max(base[0], 200);
    sw = Math.max(base[1], 2);
  } else if (hoveredNeuron || hoveredCell >= 0) {
    alpha = 8;
    sw = 0.4;
  } else {
    alpha = base[0];
    sw = base[1];
  }

  const rgb = colorMode === "gradient"
    ? (pos ? [168, 85, 247] : [244, 63, 94])
    : pos ? [56, 189, 248] : [244, 63, 94];

  return { stroke: [rgb[0], rgb[1], rgb[2], alpha], sw };
}

function drawConnections() {
  if (!showWeights || !networkState || !networkState.weights) return;

  const w0 = networkState.weights[0]; // fanOut x 784
  const d0 = weightDeltas && weightDeltas[0];
  const linkHover0 = hoveredCell >= 0 || (hoveredNeuron && hoveredNeuron.layer === 1);

  push();
  for (let j = 0; j < w0.length; j++) {
    const dest = neuronPos[1] && neuronPos[1][j];
    if (!dest) continue;
    const row = w0[j];
    const dRow = d0 && d0[j];
    for (let i = 0; i < 784; i++) {
      const cell = inputCells[i];
      const isHover = hoveredCell === i || (hoveredNeuron && hoveredNeuron.layer === 1 && hoveredNeuron.index === j && hoveredCell >= 0);
      const linked = (hoveredCell === i) ||
        (hoveredNeuron && ((hoveredNeuron.layer === 0 && hoveredNeuron.index === i) ||
          (hoveredNeuron.layer === 1 && hoveredNeuron.index === j)));
      const st = connectionStyle(row[i], maxAbsW[0] || 1, isHover, linkHover0 && linked, 0, dRow ? dRow[i] : 0);
      stroke(st.stroke[0], st.stroke[1], st.stroke[2], st.stroke[3]);
      strokeWeight(st.sw);
      line(cell.x + cell.w / 2, cell.y + cell.w / 2, dest.x, dest.y);
    }
  }
  pop();

  push();
  for (let l = 1; l < networkState.weights.length; l++) {
    const matrix = networkState.weights[l];
    const deltaM = weightDeltas && weightDeltas[l];
    const srcArr = neuronPos[l];
    const dstArr = neuronPos[l + 1];
    if (!srcArr || !dstArr) continue;
    const linkedSet = l === 1 ? linkHover0 : (hoveredNeuron && hoveredNeuron.layer >= 1 && hoveredNeuron.layer <= 2);
    for (let j = 0; j < matrix.length; j++) {
      const pDst = dstArr[j];
      if (!pDst) continue;
      for (let i = 0; i < matrix[j].length; i++) {
        const pSrc = srcArr[i];
        if (!pSrc) continue;
        const isHover = hoveredConn && hoveredConn.layer === l && hoveredConn.src === i && hoveredConn.dest === j;
        const linked = hoveredNeuron && (
          (hoveredNeuron.layer === l && hoveredNeuron.index === i) ||
          (hoveredNeuron.layer === l + 1 && hoveredNeuron.index === j)
        );
        const dVal = deltaM && deltaM[j] ? deltaM[j][i] : 0;
        const st = connectionStyle(matrix[j][i], (maxAbsW[l] || 1), isHover, linkedSet && linked, l, dVal);
        stroke(st.stroke[0], st.stroke[1], st.stroke[2], st.stroke[3]);
        strokeWeight(st.sw);
        line(pSrc.x, pSrc.y, pDst.x, pDst.y);
      }
    }
  }
  pop();
}

// ---------------------------------------------------------------------------
// Neurons — brightness = activation (or gradient)
// ---------------------------------------------------------------------------
function argmax(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

function drawNeurons() {
  if (!networkState || !networkState.activations) return;
  const acts = networkState.activations;
  const grads = networkState.gradients;
  const out = acts[acts.length - 1];
  const lastLayer = neuronPos.length - 1;
  const winIdx = out ? argmax(out) : 0;
  const winVal = out ? out[winIdx] : 0;

  for (let l = 1; l < neuronPos.length; l++) {
    const arr = neuronPos[l];
    const layerActs = acts[l];
    if (!layerActs) continue;
    const layerGrads = grads ? grads[l - 1] : null;
    const isOutput = l === lastLayer;

    for (let i = 0; i < arr.length; i++) {
      const p = arr[i];
      const act = Math.min(1, Math.max(0, layerActs[i]));
      const grad = layerGrads ? Math.min(1, Math.abs(layerGrads[i]) / (maxAbsG[l - 1] || 1)) : 0;
      const isHovered = hoveredNeuron && hoveredNeuron.layer === l && hoveredNeuron.index === i;

      const bright = colorMode === "gradient" ? grad : act;

      if (bright > 0.25) {
        noStroke();
        const haloRGB = colorMode === "gradient" ? [168, 85, 247] : [56, 189, 248];
        fill(haloRGB[0], haloRGB[1], haloRGB[2], bright * 45);
        circle(p.x, p.y, p.r * 2.9);
      }

      const isWinner = isOutput && i === winIdx && winVal > 0;
      if (isWinner) {
        noStroke();
        fill(251, 191, 36, 40 + act * 70);
        circle(p.x, p.y, p.r * 3);
      }

      let base, hot;
      if (isWinner) { base = [35, 30, 20]; hot = [251, 191, 36]; }
      else if (colorMode === "gradient") { base = [16, 20, 32]; hot = [168, 85, 247]; }
      else { base = [16, 22, 34]; hot = [56, 189, 248]; }
      const fillCol = lerpColor(color(base[0], base[1], base[2]), color(hot[0], hot[1], hot[2]), bright);
      fill(fillCol);

      if (isHovered) { stroke(255); strokeWeight(2.5); }
      else if (isWinner) { stroke(251, 191, 36); strokeWeight(2); }
      else { stroke(148, 163, 184, 150); strokeWeight(1.3); }

      circle(p.x, p.y, p.r * 2);

      noStroke();
      if (!isOutput) {
        fill(255, 255, 255, 220);
        textAlign(CENTER, CENTER);
        textSize(10);
        textStyle(BOLD);
        text("#" + i, p.x, p.y - 1);
        fill(148, 163, 184);
        textSize(9);
        textStyle(NORMAL);
        text(layerActs[i].toFixed(2), p.x, p.y + p.r + 11);
      } else {
        const digitLabels = arr.length === 10;
        fill(isWinner ? 20 : 255);
        textAlign(CENTER, CENTER);
        textSize(11);
        textStyle(BOLD);
        text(digitLabels ? i : "#" + i, p.x, p.y);
        drawOutputMeter(p.x + p.r + 10, p.y, i, layerActs[i], isWinner, digitLabels);
      }
    }
  }
}

function drawOutputMeter(x, y, digit, act, isWinner, digitLabels) {
  push();
  textAlign(LEFT, CENTER);
  textSize(10);
  textStyle(isWinner ? BOLD : NORMAL);
  fill(isWinner ? color(251, 191, 36) : color(226, 232, 240));
  text((digitLabels ? "D" : "#") + digit, x, y - 5);
  fill(isWinner ? color(251, 191, 36) : color(148, 163, 184));
  textSize(8);
  text(act.toFixed(3), x, y + 7);

  const barX = x + 34, barY = y - 3, barW = 52, barH = 5;
  fill(24, 32, 48);
  noStroke();
  rect(barX, barY, barW, barH, 2);
  if (isWinner) fill(251, 191, 36, 255); else fill(56, 189, 248, 180);
  rect(barX, barY, Math.max(2, barW * act), barH, 2);
  pop();
}

function drawLayerLabels() {
  push();
  textAlign(CENTER, CENTER);
  const arch = currentArch();
  for (let l = 1; l < layerX.length; l++) {
    const count = arch[l] !== undefined ? arch[l] : "?";
    const isInputCol = false;
    const name = l === arch.length - 1 ? "OUTPUT LAYER" : "HIDDEN LAYER " + l;
    fill(241, 245, 249);
    textSize(11);
    textStyle(BOLD);
    text(name, layerX[l], 18);
    fill(100, 116, 139);
    textSize(9);
    textStyle(NORMAL);
    text(count + " neurons" + (l === arch.length - 1 ? " · digits 0–9" : ""), layerX[l], 33);
  }
  if (inputGrid) {
    fill(241, 245, 249);
    textSize(11);
    textStyle(BOLD);
    text("INPUT LAYER", inputGrid.x + inputGrid.size / 2, 18);
    fill(100, 116, 139);
    textSize(9);
    textStyle(NORMAL);
    text("784 neurons · 28×28 grid", inputGrid.x + inputGrid.size / 2, 33);
  }
  pop();
}

// ============================================================================
// HOVER & TOOLTIP
// ============================================================================
function handleHover() {
  hoveredNeuron = null;
  hoveredCell = -1;
  hoveredConn = null;

  if (mouseX < 0 || mouseX > width || mouseY < 0 || mouseY > height) { hideTooltip(); return; }

  if (inputGrid &&
      mouseX >= inputGrid.x && mouseX < inputGrid.x + inputGrid.size &&
      mouseY >= inputGrid.y && mouseY < inputGrid.y + inputGrid.size) {
    const c = Math.floor((mouseX - inputGrid.x) / inputGrid.cell);
    const r = Math.floor((mouseY - inputGrid.y) / inputGrid.cell);
    hoveredCell = r * 28 + c;
    showCellTooltip(hoveredCell);
    cursor("pointer");
    return;
  }

  for (let l = 1; l < neuronPos.length; l++) {
    for (const p of neuronPos[l]) {
      if (dist(mouseX, mouseY, p.x, p.y) <= p.r + 5) {
        hoveredNeuron = p;
        showNeuronTooltip(p);
        cursor("pointer");
        return;
      }
    }
  }

  if (networkState) {
    for (let l = 1; l <= 2 && l < networkState.weights.length; l++) {
      const matrix = networkState.weights[l];
      const srcArr = neuronPos[l], dstArr = neuronPos[l + 1];
      if (!srcArr || !dstArr) continue;
      for (let j = 0; j < matrix.length; j++) {
        for (let i = 0; i < matrix[j].length; i++) {
          if (!srcArr[i] || !dstArr[j]) continue;
          const d = distToSegment(mouseX, mouseY, srcArr[i].x, srcArr[i].y, dstArr[j].x, dstArr[j].y);
          if (d <= 5.5) {
            hoveredConn = { layer: l, src: i, dest: j, weight: matrix[j][i] };
            showConnTooltip(hoveredConn);
            cursor("pointer");
            return;
          }
        }
      }
    }
  }

  cursor("default");
  hideTooltip();
}

function distToSegment(px, py, x1, y1, x2, y2) {
  const l2 = (x2 - x1) ** 2 + (y2 - y1) ** 2;
  if (l2 === 0) return dist(px, py, x1, y1);
  let t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2;
  t = Math.max(0, Math.min(1, t));
  return dist(px, py, x1 + t * (x2 - x1), y1 + t * (y2 - y1));
}

function showCellTooltip(i) {
  const v = inputActs ? inputActs[i] : 0;
  const r = (i / 28) | 0, c = i % 28;
  tooltipShow(
    "Input Neuron #" + i,
    row("Pixel", "row " + r + ", col " + c) +
    row("Activation", v.toFixed(4)) +
    row("Outgoing", (currentArch()[1] || 0) + " weights → H1")
  );
}

function showNeuronTooltip(p) {
  const acts = networkState.activations[p.layer];
  const grads = networkState.gradients;
  const act = acts[p.index];
  let html = row("Layer", p.layer === currentArch().length - 1 ? "OUTPUT LAYER" : "HIDDEN LAYER " + p.layer) +
    row("Activation", act.toFixed(4));
  if (networkState.biases && networkState.biases[p.layer - 1]) {
    html += row("Bias", fmtSigned(networkState.biases[p.layer - 1][p.index]));
  }
  if (grads && grads[p.layer - 1]) {
    html += row("Gradient (δ)", fmtSigned(grads[p.layer - 1][p.index]));
  }
  if (p.layer === currentArch().length - 1 && acts.length === 10) {
    html += row("Digit class", "[" + p.index + "]");
    html += row("Probability", (act * 100).toFixed(1) + "%");
  }
  tooltipShow("Neuron #" + p.index, html);
}

function showConnTooltip(cn) {
  const pos = cn.weight >= 0;
  tooltipShow(
    "Connection: layer " + cn.layer + " → " + (cn.layer + 1),
    row("Source", "neuron #" + cn.src) +
    row("Destination", "neuron #" + cn.dest) +
    row("Weight",
      `<span class="${pos ? "tooltip-badge-pos" : "tooltip-badge-neg"}">` +
      `${pos ? "+" : ""}${cn.weight.toFixed(5)}</span>`) +
    row("Sign", pos ? "positive (blue)" : "negative (red)")
  );
}

function fmtSigned(v) {
  return `<span class="${v >= 0 ? "tooltip-badge-pos" : "tooltip-badge-neg"}">${v >= 0 ? "+" : ""}${v.toFixed(4)}</span>`;
}

function row(label, value) {
  return `<div class="tooltip-row"><span class="label">${label}</span><span class="value">${value}</span></div>`;
}

function tooltipShow(title, html) {
  els.ttTitle.textContent = title;
  els.ttBody.innerHTML = html;
  positionTooltip();
  els.tooltip.classList.remove("hidden");
}

function positionTooltip() {
  let tx = mouseX + 15, ty = mouseY + 15;
  if (tx + 270 > window.innerWidth) tx = mouseX - 280;
  if (ty + 150 > window.innerHeight) ty = mouseY - 160;
  els.tooltip.style.left = Math.max(8, tx) + "px";
  els.tooltip.style.top = Math.max(8, ty) + "px";
}

function hideTooltip() {
  if (els.tooltip) els.tooltip.classList.add("hidden");
}

// ============================================================================
// CHARTS
// ============================================================================
function initCharts() {
  Chart.defaults.color = "#94a3b8";
  Chart.defaults.borderColor = "#1e2638";
  Chart.defaults.font.family = "monospace";
  Chart.defaults.font.size = 10;

  // --- live prediction (horizontal bars, digits 0-9) ---
  predChart = new Chart(document.getElementById("pred-chart"), {
    type: "bar",
    data: {
      labels: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
      datasets: [{
        label: "output",
        data: new Array(10).fill(0),
        backgroundColor: new Array(10).fill("rgba(56,189,248,0.55)"),
        borderColor: new Array(10).fill("#38bdf8"),
        borderWidth: 1,
        borderRadius: 3
      }]
    },
    options: {
      indexAxis: "y",
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { min: 0, max: 1, grid: { color: "#161d2c" } },
        y: { grid: { color: "#161d2c" } }
      },
      plugins: { legend: { display: false } }
    }
  });

  // --- training loss (raw + avg + EMA, optional log) ---
  lossChart = new Chart(document.getElementById("loss-chart"), {
    type: "line",
    data: {
      datasets: [
        {
          label: "sample loss",
          data: [],
          borderColor: "rgba(56,189,248,0.55)",
          borderWidth: 1,
          pointRadius: 0,
          tension: 0.25
        },
        {
          label: "avg loss",
          data: [],
          borderColor: "#fbbf24",
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.25
        },
        {
          label: "EMA loss",
          data: [],
          borderColor: "#22c55e",
          borderWidth: 1.8,
          borderDash: [5, 3],
          pointRadius: 0,
          tension: 0.25
        }
      ]
    },
    options: chartOpts("sample", "loss", true)
  });

  // --- accuracy ---
  accChart = new Chart(document.getElementById("acc-chart"), {
    type: "line",
    data: {
      datasets: [
        {
          label: "accuracy %",
          data: [],
          borderColor: "#22c55e",
          backgroundColor: "rgba(34,197,94,0.08)",
          borderWidth: 1.2,
          pointRadius: 0,
          tension: 0.25,
          fill: true
        }
      ]
    },
    options: (() => {
      const o = chartOpts("sample", "accuracy %", false);
      o.scales.y.min = 0;
      o.scales.y.max = 100;
      return o;
    })()
  });

  // --- epoch summary ---
  epochChart = new Chart(document.getElementById("epoch-chart"), {
    data: {
      labels: [],
      datasets: [
        {
          type: "bar",
          label: "epoch avg loss",
          data: [],
          backgroundColor: "rgba(56,189,248,0.65)",
          yAxisID: "y",
          borderRadius: 3
        },
        {
          type: "line",
          label: "epoch accuracy %",
          data: [],
          borderColor: "#fbbf24",
          backgroundColor: "#fbbf24",
          borderWidth: 2,
          pointRadius: 3,
          yAxisID: "y1",
          tension: 0.3
        }
      ]
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { color: "#161d2c" } },
        y: { position: "left", beginAtZero: true, title: { display: true, text: "loss" }, grid: { color: "#161d2c" } },
        y1: { position: "right", min: 0, max: 100, title: { display: true, text: "acc %" }, grid: { drawOnChartArea: false } }
      },
      plugins: { legend: { labels: { boxWidth: 10, boxHeight: 10 } } }
    }
  });

  // --- playground backprop steps ---
  playChart = new Chart(document.getElementById("play-chart"), {
    type: "line",
    data: {
      datasets: [
        {
          label: "step loss",
          data: [],
          borderColor: "#38bdf8",
          backgroundColor: "rgba(56,189,248,0.08)",
          borderWidth: 1.5,
          pointRadius: 2,
          tension: 0.2,
          yAxisID: "y",
          fill: true
        },
        {
          label: "target conf %",
          data: [],
          borderColor: "#fbbf24",
          borderWidth: 1.5,
          pointRadius: 2,
          tension: 0.2,
          yAxisID: "y1"
        }
      ]
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      scales: {
        x: { type: "linear", title: { display: true, text: "backprop step" }, grid: { color: "#161d2c" } },
        y: { position: "left", beginAtZero: true, title: { display: true, text: "loss" }, grid: { color: "#161d2c" } },
        y1: { position: "right", min: 0, max: 100, title: { display: true, text: "conf %" }, grid: { drawOnChartArea: false } }
      },
      plugins: { legend: { labels: { boxWidth: 10, boxHeight: 10 } } }
    }
  });

  // loss toggles
  document.getElementById("loss-log").addEventListener("change", (e) => {
    const log = e.target.checked;
    lossChart.options.scales.y.type = log ? "logarithmic" : "linear";
    lossChart.options.scales.y.min = log ? 1e-5 : 0;
    rebuildLossDatasets();
  });
  document.getElementById("loss-ema").addEventListener("change", (e) => {
    lossChart.data.datasets[2].hidden = !e.target.checked;
    lossChart.update("none");
  });
}

function chartOpts(xTitle, yTitle, fill) {
  return {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    parsing: false,
    scales: {
      x: { type: "linear", title: { display: true, text: xTitle }, grid: { color: "#161d2c" } },
      y: { beginAtZero: true, title: { display: true, text: yTitle }, grid: { color: "#161d2c" } }
    },
    plugins: { legend: { labels: { boxWidth: 10, boxHeight: 10 } } }
  };
}

function flushCharts() {
  lastFlush = millis();
  if (!pendingPoints.length) return;

  const alpha = 0.08;
  for (const p of pendingPoints) {
    lossEMA = lossEMA === null ? p.y : lossEMA + alpha * (p.y - lossEMA);
    lossRaw.push({ x: p.x, y: p.y, avg: p.avg, ema: lossEMA });
    accChart.data.datasets[0].data.push({ x: p.x, y: p.acc });
  }
  pendingPoints = [];

  if (lossRaw.length > MAX_POINTS) {
    const cut = lossRaw.length - MAX_POINTS;
    lossRaw.splice(0, cut);
    accChart.data.datasets[0].data.splice(0, cut);
  }
  rebuildLossDatasets();
  accChart.update("none");
}

function rebuildLossDatasets() {
  const logOn = document.getElementById("loss-log").checked;
  const emaOn = document.getElementById("loss-ema").checked;
  const clamp = (v) => (logOn ? Math.max(v, 1e-5) : v);
  const raw = lossChart.data.datasets[0].data;
  const avg = lossChart.data.datasets[1].data;
  const ema = lossChart.data.datasets[2].data;
  raw.length = 0; avg.length = 0; ema.length = 0;
  for (const p of lossRaw) {
    raw.push({ x: p.x, y: clamp(p.y) });
    avg.push({ x: p.x, y: clamp(p.avg) });
    ema.push({ x: p.x, y: clamp(p.ema) });
  }
  lossChart.data.datasets[2].hidden = !emaOn;
  lossChart.update("none");
}

function clearCharts() {
  pendingPoints = [];
  lossRaw = [];
  lossEMA = null;
  lossChart.data.datasets.forEach((d) => (d.data = []));
  accChart.data.datasets.forEach((d) => (d.data = []));
  epochChart.data.labels = [];
  epochChart.data.datasets.forEach((d) => (d.data = []));
  lossChart.update("none");
  accChart.update("none");
  epochChart.update("none");
}

function clearPlayChart() {
  playChart.data.datasets.forEach((d) => (d.data = []));
  playChart.update("none");
}

function updatePredChart(out) {
  if (!out || out.length < 10) return;
  const ds = predChart.data.datasets[0];
  const win = argmax(out);
  for (let i = 0; i < 10; i++) {
    ds.data[i] = Math.max(0, Math.min(1, out[i]));
    ds.backgroundColor[i] = i === win ? "rgba(251,191,36,0.75)" : "rgba(56,189,248,0.5)";
    ds.borderColor[i] = i === win ? "#fbbf24" : "#38bdf8";
  }
  predChart.update("none");
}

// ============================================================================
// DOM / HUD
// ============================================================================
function initUI() {
  els = {
    tooltip: document.getElementById("inspector-tooltip"),
    ttTitle: document.getElementById("tooltip-title"),
    ttBody: document.getElementById("tooltip-content"),
    modeBadge: document.getElementById("mode-badge"),
    epoch: document.getElementById("stat-epoch"),
    sample: document.getElementById("stat-sample"),
    loss: document.getElementById("stat-loss"),
    avgloss: document.getElementById("stat-avgloss"),
    acc: document.getElementById("stat-acc"),
    target: document.getElementById("stat-target"),
    pred: document.getElementById("stat-pred"),
    liveStats: document.getElementById("live-stats"),
    progressWrap: document.getElementById("progress-container"),
    progressBar: document.getElementById("progress-bar"),
    progressText: document.getElementById("progress-text"),
    btnTrain: document.getElementById("btn-train"),
    turboMode: document.getElementById("turbo-mode"),
    btnPause: document.getElementById("btn-pause"),
    btnStop: document.getElementById("btn-stop"),
    btnStep: document.getElementById("btn-step"),
    btnReset: document.getElementById("btn-reset"),
    epochs: document.getElementById("epochs-input"),
    lr: document.getElementById("lr-input"),
    samples: document.getElementById("samples-input"),
    speed: document.getElementById("speed-input"),
    playSource: document.getElementById("play-source"),
    playInfo: document.getElementById("play-info"),
    playTarget: document.getElementById("play-target"),
    playLr: document.getElementById("play-lr"),
    showWeights: document.getElementById("show-weights"),
    showActs: document.getElementById("show-activations"),
    showGrads: document.getElementById("show-gradients"),
    colorMode: document.getElementById("color-mode")
  };

  // target digit options
  for (let d = 0; d < 10; d++) {
    const opt = document.createElement("option");
    opt.value = d;
    opt.textContent = "digit " + d;
    els.playTarget.appendChild(opt);
  }

  els.colorMode = document.getElementById("color-mode");
  colorMode = els.colorMode.value;

  els.colorMode.addEventListener("change", (e) => {
    colorMode = e.target.value;
    syncGradientToggle();
    updateLegend();
  });
  els.showGrads.addEventListener("change", (e) => {
    if (e.target.checked) { colorMode = "gradient"; els.colorMode.value = "gradient"; }
    else { colorMode = "activation"; els.colorMode.value = "activation"; }
  });
  els.showWeights.addEventListener("change", (e) => { showWeights = e.target.checked; });

  els.btnTrain.addEventListener("click", () => {
    if (trainState.running && trainState.paused) { socket.emit("resume"); return; }
    if (trainState.running) return;
    setPlaySource("server");
    socket.emit("train", {
      epochs: clampNum(els.epochs.value, 1, 100, 3),
      lr: clampNum(els.lr.value, 0.0001, 1, 0.1),
      samples: clampNum(els.samples.value, 10, 60000, 3000),
      speed: clampNum(els.speed.value, 1, 500, 60),
      turbo: !!(els.turboMode && els.turboMode.checked)
    });
  });

  els.btnPause.addEventListener("click", () => {
    if (!trainState.running) return;
    socket.emit(trainState.paused ? "resume" : "pause");
  });

  els.btnStop.addEventListener("click", () => socket.emit("stop"));
  els.btnStep.addEventListener("click", () => socket.emit("step"));
  els.btnReset.addEventListener("click", () => {
    socket.emit("reset");
    setPlaySource("server");
  });

  // layer legend
  updateLegend();
  updateControlButtons();
}

function updateLegend() {
  const legend = document.getElementById("layer-legend");
  if (!legend) return;
  if (colorMode === "change") {
    legend.innerHTML =
      `<span><span class="color-box" style="background:#e2e8f0"></span>input brightness = pixel</span>` +
      `<span><span class="color-box" style="background:#22c55e"></span>weight increased (Δw &gt; 0)</span>` +
      `<span><span class="color-box" style="background:#f43f5e"></span>weight decreased (Δw &lt; 0)</span>` +
      `<span><span class="color-box" style="background:linear-gradient(90deg,#10161f,#22c55e)"></span>brightness = |Δw| since last step</span>` +
      `<span><span class="color-box" style="background:#fbbf24"></span>predicted digit</span>`;
  } else {
    legend.innerHTML =
      `<span><span class="color-box" style="background:#e2e8f0"></span>input brightness = pixel</span>` +
      `<span><span class="color-box" style="background:#38bdf8"></span>positive weight</span>` +
      `<span><span class="color-box" style="background:#f43f5e"></span>negative weight</span>` +
      `<span><span class="color-box" style="background:linear-gradient(90deg,#10161f,#38bdf8)"></span>neuron brightness = activation</span>` +
      `<span><span class="color-box" style="background:#fbbf24"></span>predicted digit</span>`;
  }
}

function syncGradientToggle() {
  if (els.showGrads) els.showGrads.checked = colorMode === "gradient";
}

function clampNum(v, min, max, def) {
  const n = parseFloat(v);
  if (isNaN(n)) return def;
  return Math.min(max, Math.max(min, n));
}

function updateControlButtons() {
  const running = trainState.running;
  const paused = trainState.paused;

  els.btnTrain.disabled = running && !paused;
  els.btnPause.disabled = !running;
  els.btnStop.disabled = !running;
  els.btnPause.innerHTML = paused
    ? `<span class="btn-icon">▶</span> RESUME`
    : `<span class="btn-icon">⏸</span> PAUSE`;
}

function setMode(text, color) {
  const badge = document.getElementById("mode-badge");
  if (!badge) return;
  const label = document.getElementById("mode-text");
  if (label) label.textContent = text;
  badge.style.color = color;
  badge.style.borderColor = color + "55";
  badge.style.background = color + "15";
  const dot = badge.querySelector(".badge-dot");
  if (dot) dot.style.backgroundColor = color;
}

function setPlaySource(src) {
  playSource = src;
  if (!els.playSource) return;
  if (src === "local") {
    els.playSource.textContent = "browser (local)";
    els.playSource.classList.add("local");
  } else {
    els.playSource.textContent = trainState.running ? "server live" : "server model";
    els.playSource.classList.remove("local");
  }
}

function playInfo(html) {
  if (els.playInfo) els.playInfo.innerHTML = html;
}

function setPlayInfoReady() {
  playInfo(clientNet
    ? `net <b>${clientNet.layerSizes.join("→")}</b> ready — draw a digit, click a test sample, or load an image…`
    : "waiting for Network.js library…");
}

function updateArchMetrics() {
  const arch = currentArch();
  const neurons = arch.reduce((s, n) => s + n, 0);
  let conns = 0;
  for (let i = 0; i < arch.length - 1; i++) conns += arch[i] * arch[i + 1];
  const a = document.getElementById("arch-display");
  const n = document.getElementById("metric-neurons");
  const c = document.getElementById("metric-connections");
  if (a) a.textContent = arch.join(" → ");
  if (n) n.textContent = neurons.toLocaleString();
  if (c) c.textContent = conns.toLocaleString();
}

function applyStateToDOM() {
  if (!networkState) return;

  els.liveStats.style.display = "flex";
  els.progressWrap.style.display = "block";

  if (stats) {
    els.epoch.textContent = `${stats.epoch}/${stats.totalEpochs}`;
    els.sample.textContent = `${stats.step}/${stats.totalSteps}`;
    els.loss.textContent = stats.loss.toFixed(5);
    els.avgloss.textContent = stats.avgLoss.toFixed(5);
    els.acc.textContent = stats.accuracy.toFixed(1) + "%";
    els.target.textContent = stats.label;
    els.pred.textContent = stats.prediction + (stats.isCorrect ? " ✓" : " ✗");
    els.pred.style.color = stats.isCorrect ? "#22c55e" : "#f43f5e";

    const pct = (stats.step / Math.max(1, stats.totalSteps)) * 100;
    els.progressBar.style.width = pct + "%";
    els.progressText.textContent = `${Math.round(pct)}% · epoch ${stats.epoch}`;

    if (playSource === "server") {
      playInfo(`training sample <b>${escapeHtml(stats.filename)}</b> — label ${stats.label}, pred <b>${stats.prediction}</b>, loss ${stats.loss.toFixed(5)}`);
    }
  }

  const out = networkState.activations[networkState.activations.length - 1];
  if (out) updatePredChart(out);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[ch]));
}

// ============================================================================
// PLAYGROUND — draw pad, live predict, backprop, samples, model I/O
// ============================================================================
function initPlayground() {
  pad.canvas = document.getElementById("draw-pad");
  pad.ctx = pad.canvas.getContext("2d");
  previewCanvas = document.getElementById("preview-28");
  previewCtx = previewCanvas.getContext("2d");
  off28 = document.createElement("canvas");
  off28.width = 28; off28.height = 28;
  off28Ctx = off28.getContext("2d", { willReadFrequently: true });

  clearPad(false);

  // --- pointer drawing ---
  const posOf = (e) => {
    const r = pad.canvas.getBoundingClientRect();
    return {
      x: (e.clientX - r.left) * (pad.size / r.width),
      y: (e.clientY - r.top) * (pad.size / r.height)
    };
  };
  const strokeSeg = (a, b) => {
    const ctx = pad.ctx;
    ctx.strokeStyle = pad.tool === "erase" ? padBg() : padFg();
    ctx.lineWidth = pad.brushR * 2;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  };
  const strokeDot = (p) => {
    const ctx = pad.ctx;
    ctx.fillStyle = pad.tool === "erase" ? padBg() : padFg();
    ctx.beginPath();
    ctx.arc(p.x, p.y, pad.brushR, 0, Math.PI * 2);
    ctx.fill();
  };

  pad.canvas.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    pad.drawing = true;
    try { pad.canvas.setPointerCapture(e.pointerId); } catch (_) {}
    pad.last = posOf(e);
    strokeDot(pad.last);
    flushPad();
  });
  pad.canvas.addEventListener("pointermove", (e) => {
    if (!pad.drawing) return;
    e.preventDefault();
    const p = posOf(e);
    strokeSeg(pad.last, p);
    pad.last = p;
    flushPad();
  });
  const endDraw = () => { pad.drawing = false; pad.last = null; };
  pad.canvas.addEventListener("pointerup", endDraw);
  pad.canvas.addEventListener("pointercancel", endDraw);
  pad.canvas.addEventListener("pointerleave", () => { if (pad.drawing) endDraw(); });

  // --- tools ---
  document.getElementById("btn-brush").addEventListener("click", () => setTool("brush"));
  document.getElementById("btn-erase").addEventListener("click", () => setTool("erase"));
  document.getElementById("btn-clear-pad").addEventListener("click", () => clearPad(true));
  document.getElementById("brush-size").addEventListener("input", (e) => {
    pad.brushR = +e.target.value / 2;
  });
  document.getElementById("pad-invert").addEventListener("change", (e) => {
    pad.invert = e.target.checked;
    clearPad(true);
  });

  // --- backprop / local reset ---
  document.getElementById("btn-backprop").addEventListener("click", playBackpropStep);
  document.getElementById("btn-reset-local").addEventListener("click", playResetLocal);
  els.playTarget.addEventListener("change", (e) => { playTarget = +e.target.value; });

  // --- samples / images / model files ---
  document.getElementById("btn-random-sample").addEventListener("click", () => loadSamples(true));
  document.getElementById("img-upload").addEventListener("change", handleImageUpload);
  document.getElementById("model-upload").addEventListener("change", handleModelUpload);
  document.getElementById("btn-download-model").addEventListener("click", downloadModel);

  loadSamples(false);
}

function padBg() { return pad.invert ? "#ffffff" : "#000000"; }
function padFg() { return pad.invert ? "#000000" : "#ffffff"; }

function setTool(tool) {
  pad.tool = tool;
  document.getElementById("btn-brush").classList.toggle("active-tool", tool === "brush");
  document.getElementById("btn-erase").classList.toggle("active-tool", tool === "erase");
}

function clearPad(predictEmpty) {
  pad.ctx.fillStyle = padBg();
  pad.ctx.fillRect(0, 0, pad.size, pad.size);
  if (predictEmpty !== false) {
    currentVector = new Float32Array(784);
    updatePreviewBlank();
    if (clientNet) applyPlayState(currentVector);
  }
}

function updatePreviewBlank() {
  previewCtx.fillStyle = padBg();
  previewCtx.fillRect(0, 0, 28, 28);
}

/** Downsample pad → 28×28 vector → live predict */
function flushPad() {
  off28Ctx.clearRect(0, 0, 28, 28);
  off28Ctx.drawImage(pad.canvas, 0, 0, 28, 28);
  previewCtx.clearRect(0, 0, 28, 28);
  previewCtx.drawImage(pad.canvas, 0, 0, 28, 28);

  const d = off28Ctx.getImageData(0, 0, 28, 28).data;
  const vec = new Float32Array(784);
  for (let i = 0; i < 784; i++) {
    const g = d[i * 4]; // r==g==b for our fills
    vec[i] = pad.invert ? 1 - g / 255 : g / 255;
  }
  currentVector = vec;
  if (clientNet) applyPlayState(vec);
}

/** Push a vector (e.g. from a test image) onto the pad + predictor */
function setPadFromVector(vec, labelHtml) {
  currentVector = Float32Array.from(vec);
  // draw scaled onto pad
  const img = pad.ctx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    // vector is "ink" space (1 = stroke); pad shows fg ink on bg
    const ink = Math.round(Math.max(0, Math.min(1, currentVector[i])) * 255);
    if (pad.invert) {
      // bg white, ink black
      img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = 255 - ink;
    } else {
      img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = ink;
    }
    img.data[i * 4 + 3] = 255;
  }
  // write via temp 28 canvas then scale up (nearest, crisp)
  off28Ctx.putImageData(img, 0, 0);
  pad.ctx.imageSmoothingEnabled = false;
  pad.ctx.fillStyle = padBg();
  pad.ctx.fillRect(0, 0, pad.size, pad.size);
  pad.ctx.drawImage(off28, 0, 0, pad.size, pad.size);
  pad.ctx.imageSmoothingEnabled = true;
  previewCtx.clearRect(0, 0, 28, 28);
  previewCtx.drawImage(off28, 0, 0);

  if (clientNet) applyPlayState(currentVector);
  if (labelHtml) playInfo(labelHtml);
}

/**
 * Local predict using the real Network.js in the browser.
 * Builds the same networkState structure the renderer already understands.
 */
function applyPlayState(vector) {
  if (!clientNet || !vector) return null;

  const out = clientNet.predict(vector);
  lastOut = Array.from(out);

  const acts = clientNet.NN
    .filter((_, i) => i % 2 === 0)
    .map((a) => Array.from(a));

  networkState = {
    architecture: clientNet.layerSizes,
    weights: clientNet.NN.filter((_, i) => i % 2 === 1),
    biases: clientNet.biases,
    activations: acts,
    gradients: playGradients,
    weightDeltas,
    maxAbsDelta
  };
  inputActs = vector;
  maxAbsW = computeMaxAbs(networkState.weights);
  if (playGradients) maxAbsG = computeMaxAbs(playGradients);
  maybeRelayout();
  updateArchMetrics();

  const pred = argmax(lastOut);
  updatePredChart(lastOut);

  // header reflect play state
  els.liveStats.style.display = "flex";
  els.pred.textContent = pred;
  els.pred.style.color = "#38bdf8";
  if (playTarget >= 0) {
    els.target.textContent = playTarget;
  }
  setPlaySource("local");

  return { out: lastOut, pred };
}

function playBackpropStep() {
  if (!clientNet) { playInfo(`<span class="bad">Network.js not loaded yet</span>`); return; }
  if (!currentVector || currentVector.every((v) => v === 0)) {
    playInfo(`<span class="bad">nothing to train — draw or load a digit first</span>`);
    return;
  }
  const out = clientNet.predict(currentVector);
  const target = playTarget >= 0 ? playTarget : argmax(out);
  const t = new Array(10).fill(0);
  t[target] = 1;

  let loss = 0;
  for (let j = 0; j < out.length; j++) {
    const d = out[j] - t[j];
    loss += d * d;
  }
  loss /= out.length;

  // Snapshot weights before step so we can show Δw
  const prevW = clientNet.extractWeights
    ? clientNet.extractWeights()
    : clientNet.NN.filter((_, i) => i % 2 === 1).map((m) => m.map((r) => [...r]));

  playGradients = clientNet.backpropagation(t, clampNum(els.playLr.value, 0.0001, 1, 0.1));
  playStepIdx++;

  const curW = clientNet.NN.filter((_, i) => i % 2 === 1).map((m) => m.map((r) => [...r]));
  weightDeltas = curW.map((m, l) =>
    m.map((row, j) => Array.from(row, (v, i) => v - (prevW[l] && prevW[l][j] ? prevW[l][j][i] : 0)))
  );
  maxAbsDelta = 0;
  for (const m of weightDeltas) {
    for (const row of m) {
      for (const d of row) {
        const a = Math.abs(d);
        if (a > maxAbsDelta) maxAbsDelta = a;
      }
    }
  }
  maxAbsW = computeMaxAbs(curW);

  const out2 = clientNet.predict(currentVector);
  playChart.data.datasets[0].data.push({ x: playStepIdx, y: loss });
  playChart.data.datasets[1].data.push({ x: playStepIdx, y: out[target] * 100 });
  if (playChart.data.datasets[0].data.length > MAX_POINTS) {
    playChart.data.datasets[0].data.shift();
    playChart.data.datasets[1].data.shift();
  }
  playChart.update("none");

  applyPlayState(currentVector);
  setPlaySource("local");

  const confBefore = (out[target] * 100).toFixed(1);
  const confAfter = (out2[target] * 100).toFixed(1);
  playInfo(
    `backprop #${playStepIdx} → target <b>${target}</b> · loss ${loss.toFixed(5)} · ` +
    `conf ${confBefore}% → <b>${confAfter}%</b> · max|Δw| ${maxAbsDelta.toFixed(4)} (browser only)`
  );
}

function playResetLocal() {
  if (!window.Network) return;
  const arch = currentArch();
  clientNet = new window.Network(arch);
  playGradients = null;
  playStepIdx = 0;
  weightDeltas = null;
  maxAbsDelta = 0;
  clearPlayChart();
  if (currentVector) applyPlayState(currentVector);
  else {
    // show zero-input activations of the fresh net
    applyPlayState(new Float32Array(784));
  }
  setPlaySource("local");
  playInfo(`fresh random net <b>${arch.join("→")}</b> in browser — server model unchanged`);
}

// ---------------------------------------------------------------------------
// Test samples from the real MNIST test set
// ---------------------------------------------------------------------------
async function loadSamples(pickFirst) {
  const strip = document.getElementById("sample-strip");
  try {
    const res = await fetch("/api/random-samples?n=12");
    const list = await res.json();
    strip.innerHTML = "";
    for (const s of list) {
      const btn = document.createElement("button");
      btn.className = "sample-thumb";
      btn.title = `digit ${s.label} — ${s.filename}`;
      const c = document.createElement("canvas");
      c.width = 28; c.height = 28;
      btn.appendChild(c);
      strip.appendChild(btn);

      const img = new Image();
      img.src = s.url;
      img.onload = () => {
        c.getContext("2d").drawImage(img, 0, 0, 28, 28);
      };

      btn.addEventListener("click", () => {
        strip.querySelectorAll(".sample-thumb").forEach((el) => el.classList.remove("active"));
        btn.classList.add("active");
        const g = c.getContext("2d").getImageData(0, 0, 28, 28);
        vectorFromInkImage(g, (vec) => {
          setPadFromVector(vec,
            `test image <b>${escapeHtml(s.filename)}</b> — true label <b>${s.label}</b>`);
        });
      });

      if (pickFirst && s === list[0]) {
        // auto-load first fresh sample
        const img2 = new Image();
        img2.src = s.url;
        img2.onload = () => {
          const tmp = document.createElement("canvas");
          tmp.width = 28; tmp.height = 28;
          tmp.getContext("2d").drawImage(img2, 0, 0, 28, 28);
          const g = tmp.getContext("2d").getImageData(0, 0, 28, 28);
          vectorFromInkImage(g, (vec) => {
            setPadFromVector(vec,
              `test image <b>${escapeHtml(s.filename)}</b> — true label <b>${s.label}</b>`);
          });
        };
      }
    }
  } catch (e) {
    playInfo(`<span class="bad">failed to load samples: ${escapeHtml(e.message)}</span>`);
  }
}

/** ImageData (white ink on black) → Float32Array(784) */
function vectorFromInkImage(imageData, cb) {
  const vec = new Float32Array(784);
  for (let i = 0; i < 784; i++) {
    vec[i] = imageData.data[i * 4] / 255;
  }
  cb(vec);
}

// ---------------------------------------------------------------------------
// User image upload → center-crop square → 28×28
// ---------------------------------------------------------------------------
function handleImageUpload(e) {
  const file = e.target.files && e.target.files[0];
  e.target.value = "";
  if (!file) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    URL.revokeObjectURL(url);
    const s = Math.min(img.width, img.height);
    const sx = (img.width - s) / 2, sy = (img.height - s) / 2;
    off28Ctx.fillStyle = "#000";
    off28Ctx.fillRect(0, 0, 28, 28);
    off28Ctx.drawImage(img, sx, sy, s, s, 0, 0, 28, 28);
    const g = off28Ctx.getImageData(0, 0, 28, 28);
    // luminance → ink (bright strokes). If the image is dark-on-light, user should tick INVERT.
    vectorFromInkImage(g, (vec) => {
      setPadFromVector(vec, `loaded image <b>${escapeHtml(file.name)}</b> (${img.width}×${img.height} → 28×28)`);
    });
  };
  img.onerror = () => {
    URL.revokeObjectURL(url);
    playInfo(`<span class="bad">could not read image</span>`);
  };
  img.src = url;
}

// ---------------------------------------------------------------------------
// Model upload / download — the real Network.js format
// ---------------------------------------------------------------------------
function handleModelUpload(e) {
  const file = e.target.files && e.target.files[0];
  e.target.value = "";
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    let model;
    try {
      model = JSON.parse(reader.result);
    } catch (err) {
      playInfo(`<span class="bad">invalid JSON: ${escapeHtml(err.message)}</span>`);
      return;
    }
    const sizes = model.architecture || model.layerSizes;
    if (!Array.isArray(sizes) || sizes[0] !== 784 || sizes[sizes.length - 1] !== 10) {
      playInfo(`<span class="bad">model must be 784 → … → 10</span>`);
      return;
    }
    try {
      clientNet = window.Network.fromJSON(model);
    } catch (err) {
      playInfo(`<span class="bad">bad model: ${escapeHtml(err.message)}</span>`);
      return;
    }
    // apply locally right away, then hand it to the server too
    playGradients = null;
    playStepIdx = 0;
    clearPlayChart();
    ARCHITECTURE = sizes;
    lastArchKey = sizes.join(",");
    calculateLayout(width, height);
    if (currentVector) applyPlayState(currentVector);
    else applyPlayState(new Float32Array(784));
    setPlaySource("local");
    playInfo(`uploaded <b>${escapeHtml(file.name)}</b> (${sizes.join("→")}) — applying to browser + server…`);
    socket.emit("model:upload", model);
  };
  reader.readAsText(file);
}

function downloadModel() {
  if (!clientNet) return;
  const data = clientNet.toJSON();
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "model-playground.json";
  a.click();
  URL.revokeObjectURL(a.href);
  playInfo(`downloaded <b>model-playground.json</b> (${data.architecture.join("→")})`);
}

// ============================================================================
// Test hooks (used by test/e2e.mjs)
// ============================================================================
window.__play = {
  hasLib: () => !!window.Network,
  getClient: () => clientNet,
  setVector(v) {
    currentVector = Float32Array.from(v);
    if (clientNet) return applyPlayState(currentVector);
    return null;
  },
  predict(v) {
    if (!clientNet) return null;
    return Array.from(clientNet.predict(Float32Array.from(v)));
  },
  backprop: () => playBackpropStep(),
  // synthetic "1" stroke for deterministic tests
  drawOne() {
    const v = new Float32Array(784);
    for (let r = 2; r < 26; r++) {
      for (let c = 12; c <= 15; c++) v[r * 28 + c] = 1;
    }
    return this.setVector(v);
  }
};
