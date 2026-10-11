import express from "express";
import { createServer } from "http";
import { Server } from "socket.io";
import { readFileSync, readdirSync, existsSync, writeFileSync } from "fs";
import { readFile } from "fs/promises";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { availableParallelism } from "os";
import sharp from "sharp";
import { Network } from "../NN/Network.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const httpServer = createServer(app);
const io = new Server(httpServer, { cors: { origin: "*" } });
app.use(express.static(join(__dirname, "../visualizer")));
app.use("/NN", express.static(join(__dirname, "../NN")));

const MODEL_FILE = join(__dirname, "../model.json");
const TRAIN_DIR = join(__dirname, "../dataset/train");
const TEST_DIR = join(__dirname, "../dataset/test");
const ARCH = [784, 3, 3, 10];

let testFiles = [];

function loadTestFiles() {
  testFiles = readdirSync(TEST_DIR).filter((f) => f.endsWith(".png"));
  console.log(`Test set indexed: ${testFiles.length} images`);
}

app.get("/api/random-samples", (req, res) => {
  const n = Math.min(24, Math.max(1, +req.query.n || 12));
  const picks = [];
  for (let i = 0; i < n; i++) {
    const f = testFiles[Math.floor(Math.random() * testFiles.length)];
    picks.push({ filename: f, label: parseInt(f.split("_")[0], 10), url: `/dataset/test/${f}` });
  }
  res.json(picks);
});

let net = null;
let dataset = [];      // { filename, label }
let inputCache = [];   // Float32Array(784) | null
let readyFlags = [];
let readyCount = 0;

const ctl = { running: false, paused: false, stopped: false, step: false, turbo: false };
let lastGradients = null;
let lrCurrent = 0.1;
let turboWorkers = 1;
let prevWeights = null; // deep copy of weight matrices for delta viz

function copyWeights() {
  return net.NN
    .filter((_, i) => i % 2 === 1)
    .map((m) => m.map((row) => [...row]));
}

function weightDeltasFrom(prev) {
  if (!prev) return null;
  const cur = copyWeights();
  return cur.map((m, l) =>
    m.map((row, j) => Array.from(row, (v, i) => round(v - prev[l][j][i], 5)))
  );
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const round = (v, d) => Math.round(v * Math.pow(10, d)) / Math.pow(10, d);

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// ---------------------------------------------------------------------------
// Network lifecycle
// ---------------------------------------------------------------------------
function initNetwork(fromModel) {
  if (fromModel && existsSync(MODEL_FILE)) {
    net = Network.load(MODEL_FILE);
    console.log("Loaded trained model from model.json");
  } else {
    net = new Network(ARCH);
    console.log("Initialized fresh random network");
  }
  lastGradients = null;
  prevWeights = copyWeights();
}

function rawState() {
  return {
    rawWeights: net.NN
      .filter((_, i) => i % 2 === 1)
      .map((m) => m.map((row) => [...row])),
    rawBiases: net.biases.map((b) => [...b])
  };
}

function snapshot(extra = {}) {
  const weights = [];
  const activations = [];
  for (let i = 0; i < net.NN.length; i++) {
    if (i % 2 === 1) {
      weights.push(net.NN[i].map((row) => Array.from(row, (v) => round(v, 5))));
    } else {
      activations.push(Array.from(net.NN[i], (v) => round(v, 4)));
    }
  }
  const weightDeltas = weightDeltasFrom(prevWeights);
  let maxAbsDelta = 0;
  if (weightDeltas) {
    for (const m of weightDeltas) {
      for (const row of m) {
        for (const d of row) {
          const a = Math.abs(d);
          if (a > maxAbsDelta) maxAbsDelta = a;
        }
      }
    }
  }
  return {
    architecture: [...net.layerSizes],
    weights,
    biases: net.biases.map((b) => Array.from(b, (v) => round(v, 5))),
    activations,
    gradients: lastGradients
      ? lastGradients.map((g) => Array.from(g, (v) => round(v, 4)))
      : null,
    weightDeltas,
    maxAbsDelta: round(maxAbsDelta, 5),
    ...extra
  };
}

function commitPrevWeights() {
  prevWeights = copyWeights();
}

// ---------------------------------------------------------------------------
// Dataset loading + background preprocessing (lazy, ahead of training)
// ---------------------------------------------------------------------------
const CACHE_BIN = join(__dirname, "../dataset/cache.bin");
const CACHE_META = join(__dirname, "../dataset/cache.json");

function loadDataset() {
  const files = readdirSync(TRAIN_DIR).filter((f) => f.endsWith(".png"));

  // fast path: binary cache from a previous run
  if (existsSync(CACHE_BIN) && existsSync(CACHE_META)) {
    try {
      const meta = JSON.parse(readFileSync(CACHE_META, "utf-8"));
      if (meta.files && meta.files.length === files.length) {
        const buf = readFileSync(CACHE_BIN);
        if (buf.byteLength === files.length * 784 * 4) {
          dataset = meta.files.map((f) => ({
            filename: f,
            label: parseInt(f.split("_")[0], 10)
          }));
          const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
          const all = new Float32Array(ab);
          inputCache = [];
          readyFlags = [];
          for (let i = 0; i < dataset.length; i++) {
            inputCache.push(all.subarray(i * 784, (i + 1) * 784));
            readyFlags.push(true);
          }
          readyCount = dataset.length;
          console.log(`Dataset cache loaded: ${dataset.length} images`);
          io.emit("data:prep", { ready: readyCount, total: dataset.length, done: true });
          return;
        }
      }
      console.warn("Cache invalid, reprocessing dataset");
    } catch (e) {
      console.warn("Cache load failed, reprocessing:", e.message);
    }
  }

  dataset = shuffle(
    files.map((f) => ({ filename: f, label: parseInt(f.split("_")[0], 10) }))
  );
  inputCache = new Array(dataset.length).fill(null);
  readyFlags = new Array(dataset.length).fill(false);
  readyCount = 0;
  console.log(`Dataset indexed: ${dataset.length} training images (shuffled)`);
  runPreprocessor();
}

async function runPreprocessor() {
  const CONCURRENCY = 16;
  // shuffled order so any random training sample becomes ready uniformly quickly
  const order = shuffle([...Array(dataset.length).keys()]);
  let pos = 0;
  const worker = async () => {
    while (pos < order.length) {
      const myIdx = order[pos++];
      const { filename } = dataset[myIdx];
      try {
        const buf = await readFile(join(TRAIN_DIR, filename));
        const { data } = await sharp(buf)
          .resize(28, 28)
          .grayscale()
          .raw()
          .toBuffer({ resolveWithObject: true });
        const arr = new Float32Array(784);
        for (let i = 0; i < 784; i++) arr[i] = data[i] / 255;
        inputCache[myIdx] = arr;
        readyFlags[myIdx] = true;
        readyCount++;
        if (readyCount % 2000 === 0) {
          io.emit("data:prep", { ready: readyCount, total: dataset.length });
        }
      } catch (e) {
        console.error("prep failed:", filename, e.message);
        readyFlags[myIdx] = true; // unblock training, skip bad file
      }
    }
  };
  await Promise.all(Array.from({ length: CONCURRENCY }, worker));
  io.emit("data:prep", { ready: readyCount, total: dataset.length, done: true });
  console.log(`Preprocessing complete: ${readyCount}/${dataset.length}`);

  // persist binary cache for instant restarts
  try {
    const all = new Float32Array(dataset.length * 784);
    for (let i = 0; i < dataset.length; i++) {
      if (inputCache[i]) all.set(inputCache[i], i * 784);
    }
    writeFileSync(CACHE_BIN, Buffer.from(all.buffer));
    writeFileSync(CACHE_META, JSON.stringify({ files: dataset.map((d) => d.filename) }));
    console.log("Dataset cache saved");
  } catch (e) {
    console.warn("cache save failed:", e.message);
  }
}

function waitReady(i) {
  if (readyFlags[i]) return Promise.resolve(inputCache[i]);
  return new Promise((resolve) => {
    const t = setInterval(() => {
      if (readyFlags[i]) {
        clearInterval(t);
        resolve(inputCache[i]);
      }
    }, 15);
  });
}

// ---------------------------------------------------------------------------
// Training loop (mirrors Network.train from src/index.js, with stream + controls)
// ---------------------------------------------------------------------------
async function trainSample(idx, epoch, epochs, orderLen, acc) {
  const input = await waitReady(idx);
  if (!input) return false;
  const { filename, label } = dataset[idx];

  // forward
  const target = new Array(10).fill(0);
  target[label] = 1;
  const output = net.predict(input);

  // loss (same as Network.train: mean squared error)
  let sampleLoss = 0;
  for (let j = 0; j < output.length; j++) {
    const d = output[j] - target[j];
    sampleLoss += d * d;
  }
  sampleLoss /= output.length;
  acc.totalLoss += sampleLoss;

  const prediction = net.getPrediction(output);
  const isCorrect = prediction === label;
  if (isCorrect) acc.correct++;

  // backward
  lastGradients = net.backpropagation(target, lrCurrent);

  acc.count++;
  const avgLoss = acc.totalLoss / acc.count;
  const accuracy = (acc.correct / acc.count) * 100;

  io.emit(
    "network:update",
    snapshot({
      epoch,
      totalEpochs: epochs,
      step: acc.count,
      totalSteps: orderLen,
      globalStep: (epoch - 1) * orderLen + acc.count,
      filename,
      label,
      prediction,
      isCorrect,
      loss: round(sampleLoss, 6),
      avgLoss: round(avgLoss, 6),
      accuracy: round(accuracy, 2),
      input: Array.from(input, (v) => round(v, 3))
    })
  );
  commitPrevWeights();
  return true;
}

async function trainLoop({ epochs, lr, samples, speed }) {
  lrCurrent = lr;
  const n = Math.min(samples, dataset.length);
  ctl.running = true;
  ctl.stopped = false;
  ctl.step = false;
  io.emit("training:state", { running: true, paused: ctl.paused });
  const delay = 1000 / speed;

  for (let epoch = 1; epoch <= epochs && !ctl.stopped; epoch++) {
    const order = shuffle([...Array(dataset.length).keys()]).slice(0, n);
    const acc = { totalLoss: 0, correct: 0, count: 0 };

    for (let i = 0; i < order.length; i++) {
      // pause / step / stop gates
      while (ctl.paused && !ctl.stopped && !ctl.step) await sleep(40);
      if (ctl.stopped) break;
      ctl.step = false;

      await trainSample(order[i], epoch, epochs, order.length, acc);

      if (delay > 0) await sleep(delay);
    }

    if (ctl.stopped) break;

    io.emit("training:epoch", {
      epoch,
      loss: round(acc.totalLoss / Math.max(1, acc.count), 6),
      accuracy: round((acc.correct / Math.max(1, acc.count)) * 100, 2)
    });
  }

  ctl.running = false;
  ctl.paused = false;
  ctl.turbo = false;
  io.emit("training:state", { running: false, paused: false, turbo: false });
  io.emit("training:done", { stopped: ctl.stopped, ...rawState() });
}

// ---------------------------------------------------------------------------
// Turbo training: multi-core mini-batch SGD via Network.trainTurbo
// Master network stays on this thread; workers only compute gradients.
// ---------------------------------------------------------------------------
async function trainTurboLoop({ epochs, lr, samples, batchSize }) {
  lrCurrent = lr;
  const n = Math.min(samples, dataset.length);
  ctl.running = true;
  ctl.stopped = false;
  ctl.step = false;
  ctl.turbo = true;
  turboWorkers = availableParallelism?.() ?? 4;
  io.emit("training:state", { running: true, paused: ctl.paused, turbo: true, workers: turboWorkers });
  console.log(`Turbo training: ${turboWorkers} workers, batch=${batchSize}, samples/epoch=${n}`);

  // Snapshot preprocessed inputs once for workers (structured-clone friendly)
  const order = shuffle([...Array(dataset.length).keys()]).slice(0, n);
  const trainingData = [];
  for (const idx of order) {
    const input = await waitReady(idx);
    if (!input) continue;
    trainingData.push({
      input: Array.from(input),
      label: dataset[idx].label,
      filename: dataset[idx].filename
    });
  }

  await net.trainTurbo(epochs, trainingData, null, lr, (info) => {
    lastGradients = info.deltas;

    io.emit(
      "network:update",
      snapshot({
        epoch: info.epoch,
        totalEpochs: info.totalEpochs,
        step: info.step,
        totalSteps: trainingData.length * info.totalEpochs,
        globalStep: info.batch,
        filename: `turbo-batch-${info.batch}`,
        label: info.label,
        prediction: info.prediction,
        isCorrect: info.prediction === info.label,
        loss: round(info.batchLoss, 6),
        avgLoss: round(info.averageLoss, 6),
        accuracy: round(info.accuracy, 2),
        turbo: true,
        workers: info.workers,
        batchSize: info.batchSize,
        input: info.input
      })
    );
    commitPrevWeights();
  }, {
    workers: turboWorkers,
    batchSize: batchSize || turboWorkers * 8,
    shouldStop: () => ctl.stopped,
    onBatchStart: async () => {
      while (ctl.paused && !ctl.stopped && !ctl.step) await sleep(40);
      if (ctl.stopped) return false;
      ctl.step = false;
      return true;
    },
    onEpoch: (e) => {
      io.emit("training:epoch", {
        epoch: e.epoch,
        loss: round(e.loss, 6),
        accuracy: round(e.accuracy, 2),
        turbo: true
      });
    }
  });

  ctl.running = false;
  ctl.paused = false;
  ctl.turbo = false;
  io.emit("training:state", { running: false, paused: false, turbo: false });
  io.emit("training:done", { stopped: ctl.stopped, turbo: true, ...rawState() });
  console.log("Turbo training finished");
}

// ---------------------------------------------------------------------------
// Socket.IO
// ---------------------------------------------------------------------------
app.use("/dataset/test", express.static(TEST_DIR));

io.on("connection", (socket) => {
  console.log("Client connected:", socket.id);

  socket.emit("model:loaded", snapshot({ reset: false, ...rawState() }));
  socket.emit("data:prep", {
    ready: readyCount,
    total: dataset.length,
    done: readyCount >= dataset.length
  });
  socket.emit("training:state", { running: ctl.running, paused: ctl.paused, turbo: ctl.turbo, workers: turboWorkers });

  socket.on("train", (cfg) => {
    if (ctl.running) return;
    console.log("train:", cfg);
    ctl.paused = false;
    const common = {
      epochs: Math.max(1, cfg.epochs | 0),
      lr: Math.min(1, Math.max(0.0001, +cfg.lr || 0.1)),
      samples: Math.min(dataset.length, Math.max(10, cfg.samples | 0))
    };
    if (cfg.turbo) {
      trainTurboLoop({
        ...common,
        batchSize: Math.max(1, cfg.batchSize | 0) || undefined
      });
    } else {
      trainLoop({
        ...common,
        speed: Math.min(500, Math.max(1, +cfg.speed || 60))
      });
    }
  });

  socket.on("pause", () => {
    if (!ctl.running) return;
    ctl.paused = true;
    io.emit("training:state", { running: true, paused: true, turbo: ctl.turbo });
  });

  socket.on("resume", () => {
    if (!ctl.running) return;
    ctl.paused = false;
    io.emit("training:state", { running: true, paused: false, turbo: ctl.turbo });
  });

  socket.on("stop", () => {
    if (!ctl.running) return;
    ctl.stopped = true;
    ctl.paused = false;
    ctl.step = false;
  });

  socket.on("step", () => {
    if (ctl.running) {
      ctl.step = true;
      return;
    }
    // idle: process exactly one random sample (no loop)
    (async () => {
      const idx = Math.floor(Math.random() * dataset.length);
      const acc = { totalLoss: 0, correct: 0, count: 0 };
      await trainSample(idx, 1, 1, 1, acc);
    })();
  });

  socket.on("reset", () => {
    if (ctl.running) {
      ctl.stopped = true;
      ctl.paused = false;
      ctl.step = false;
    }
    initNetwork(false); // fresh random weights
    io.emit("model:loaded", snapshot({ reset: true, ...rawState() }));
  });

  socket.on("model:upload", (model) => {
    try {
      const sizes = model.architecture || model.layerSizes;
      if (!Array.isArray(sizes) || sizes[0] !== 784 || sizes[sizes.length - 1] !== 10) {
        throw new Error("Model must be 784 -> ... -> 10 (MNIST input/output)");
      }
      if (ctl.running) {
        ctl.stopped = true;
        ctl.paused = false;
        ctl.step = false;
      }
      net = Network.fromJSON(model); // validates architecture/weights/biases
      lastGradients = null;
      prevWeights = copyWeights();
      console.log("Model uploaded:", net.layerSizes.join(" -> "));
      io.emit("model:loaded", snapshot({ reset: false, source: "upload", ...rawState() }));
    } catch (e) {
      socket.emit("model:error", { message: e.message });
    }
  });
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
initNetwork(true);
loadDataset();
loadTestFiles();
httpServer.listen(3000, () => {
  console.log("Training server: http://localhost:3000");
});