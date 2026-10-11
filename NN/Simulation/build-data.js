import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import sharp from "sharp";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const TRAIN_DIR = path.join(__dirname, "../../dataset/train");
const TEST_DIR = path.join(__dirname, "../../dataset/test");
const OUT_FILE = path.join(__dirname, "data/samples.js");

async function extractSamples() {
  console.log("Extracting MNIST samples for browser simulation...");
  const trainFiles = fs.readdirSync(TRAIN_DIR);
  const testFiles = fs.readdirSync(TEST_DIR);

  const trainSamples = [];
  const testSamples = [];

  // 60 samples per digit for training = 600 training samples
  // 15 samples per digit for testing = 150 test samples
  for (let d = 0; d <= 9; d++) {
    const dTrain = trainFiles.filter(f => f.startsWith(d + "_")).slice(0, 60);
    for (const f of dTrain) {
      const buf = await sharp(path.join(TRAIN_DIR, f))
        .grayscale()
        .resize(28, 28)
        .raw()
        .toBuffer();
      trainSamples.push({
        id: f,
        label: d,
        b64: buf.toString("base64")
      });
    }

    const dTest = testFiles.filter(f => f.startsWith(d + "_")).slice(0, 15);
    for (const f of dTest) {
      const buf = await sharp(path.join(TEST_DIR, f))
        .grayscale()
        .resize(28, 28)
        .raw()
        .toBuffer();
      testSamples.push({
        id: f,
        label: d,
        b64: buf.toString("base64")
      });
    }
  }

  // Shuffle training samples
  for (let i = trainSamples.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [trainSamples[i], trainSamples[j]] = [trainSamples[j], trainSamples[i]];
  }

  const jsContent = `// Auto-generated bundled MNIST samples for browser simulation
// Train: ${trainSamples.length} samples, Test: ${testSamples.length} samples
window.MNIST_DATA = {
  train: ${JSON.stringify(trainSamples)},
  test: ${JSON.stringify(testSamples)},
  decode: function(b64) {
    const bin = atob(b64);
    const len = bin.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
    const floats = new Float32Array(len);
    for (let i = 0; i < len; i++) floats[i] = bytes[i] / 255.0;
    return floats;
  }
};
`;

  fs.mkdirSync(path.dirname(OUT_FILE), { recursive: true });
  fs.writeFileSync(OUT_FILE, jsContent, "utf-8");
  console.log(`Saved ${trainSamples.length} train and ${testSamples.length} test samples to ${OUT_FILE}`);
  const stats = fs.statSync(OUT_FILE);
  console.log(`File size: ${(stats.size / 1024).toFixed(1)} KB`);
}

extractSamples().catch(console.error);
