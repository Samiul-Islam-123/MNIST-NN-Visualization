import express from "express";
import http from "http";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const server = http.createServer(app);

const SIMULATION_DIR = __dirname;
const ROOT_DIR = path.join(__dirname, "../..");
const DATASET_DIR = path.join(ROOT_DIR, "dataset");
const MODEL_FILE = path.join(ROOT_DIR, "model.json");

// Serve Simulation static files
app.use(express.static(SIMULATION_DIR));

// Serve parent NN folder for direct module imports if needed
app.use("/NN", express.static(path.join(ROOT_DIR, "NN")));

// Serve full dataset for deep browsing
app.use("/dataset", express.static(DATASET_DIR));

// Serve root model.json
app.get("/model.json", (req, res) => {
  if (fs.existsSync(MODEL_FILE)) {
    res.sendFile(MODEL_FILE);
  } else {
    res.status(404).json({ error: "model.json not found" });
  }
});

// API for random test image
app.get("/api/random-test", (req, res) => {
  const testDir = path.join(DATASET_DIR, "test");
  if (!fs.existsSync(testDir)) {
    return res.status(404).json({ error: "Test dataset folder not found" });
  }
  const files = fs.readdirSync(testDir).filter(f => f.endsWith(".png"));
  if (!files.length) {
    return res.status(404).json({ error: "No PNG images found" });
  }
  const randomFile = files[Math.floor(Math.random() * files.length)];
  const label = parseInt(randomFile.split("_")[0], 10);
  res.json({
    filename: randomFile,
    label,
    url: `/dataset/test/${randomFile}`
  });
});

let currentPort = parseInt(process.env.PORT, 10) || 3000;

function listenOnPort(port) {
  server.listen(port, () => {
    console.log(`\n========================================================`);
    console.log(`  MNIST NEURAL NETWORK SIMULATION PLAYGROUND RUNNING`);
    console.log(`  Local URL: http://localhost:${port}`);
    console.log(`========================================================\n`);
  });
}

server.on("error", (err) => {
  if (err.code === "EADDRINUSE") {
    const prevPort = currentPort;
    currentPort++;
    console.warn(`[!] Port ${prevPort} is in use. Retrying on port ${currentPort}...`);
    setTimeout(() => {
      listenOnPort(currentPort);
    }, 100);
  } else {
    console.error("Server error:", err);
  }
});

listenOnPort(currentPort);
