import puppeteer from "puppeteer-core";

const EDGE = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const URL = "http://localhost:3000";
const errors = [];
const consoleErrs = [];

const browser = await puppeteer.launch({
  executablePath: EDGE,
  headless: true,
  args: ["--no-sandbox", "--disable-gpu"]
});
const page = await browser.newPage();
await page.setViewport({ width: 1720, height: 1000, deviceScaleFactor: 1 });

page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
page.on("console", (m) => { if (m.type() === "error") consoleErrs.push("console: " + m.text()); });

await page.goto(URL, { waitUntil: "domcontentloaded", timeout: 20000 });
await page.waitForSelector("#canvas-container canvas", { timeout: 15000 });
await new Promise((r) => setTimeout(r, 4000));

// --- initial render checks ---
const initial = await page.evaluate(() => {
  const c = document.querySelector("#canvas-container canvas");
  if (!c) return { canvas: false };
  const ctx = c.getContext("2d");
  const { data } = ctx.getImageData(0, 0, c.width, c.height);
  let lit = 0;
  for (let i = 0; i < data.length; i += 40) {
    if (data[i] > 45 || data[i + 1] > 45) lit++;
  }
  return {
    canvas: true,
    litSamples: lit,
    socketConnected: typeof io !== "undefined",
    modeText: document.getElementById("mode-text")?.textContent,
    netLib: window.__play.hasLib(),
    predChart: !!Chart.getChart("pred-chart"),
    playChart: !!Chart.getChart("play-chart"),
    padExists: !!document.getElementById("draw-pad"),
    playSource: document.getElementById("play-source")?.textContent
  };
});
console.log("INITIAL:", JSON.stringify(initial));
await page.screenshot({ path: "test/shot-initial.png" });

// --- hover over input grid for tooltip ---
const cellPos = await page.evaluate(() => {
  const c = document.querySelector("#canvas-container canvas");
  const r = c.getBoundingClientRect();
  const g = inputGrid;
  return { x: r.left + g.x + g.cell * 5, y: r.top + g.y + g.cell * 14 };
});
await page.mouse.move(cellPos.x, cellPos.y);
await new Promise((r) => setTimeout(r, 400));
const tooltip = await page.evaluate(() => {
  const t = document.getElementById("inspector-tooltip");
  return { hidden: t.classList.contains("hidden"), title: document.getElementById("tooltip-title").textContent };
});
console.log("TOOLTIP on grid:", JSON.stringify(tooltip));
await page.screenshot({ path: "test/shot-tooltip.png" });

// --- playground: synthetic vector predict via real Network.js in browser ---
const playPredict = await page.evaluate(() => {
  const res = window.__play.drawOne();
  const acts = networkState.activations.map((a) => a.length);
  const out = networkState.activations[networkState.activations.length - 1];
  const predChart = Chart.getChart("pred-chart");
  return {
    pred: res && res.pred,
    acts,
    outNonzero: out.some((v) => v > 0.01),
    predChartData: predChart.data.datasets[0].data.filter((v) => v > 0.01).length,
    source: document.getElementById("play-source").textContent,
    inputNonzero: inputActs && inputActs.some((v) => v > 0.5)
  };
});
console.log("PLAY PREDICT (synthetic 1):", JSON.stringify(playPredict));

// --- playground: draw with the mouse on the pad ---
const padBox = await page.evaluate(() => {
  const r = document.getElementById("draw-pad").getBoundingClientRect();
  return { x: r.x, y: r.y, w: r.width, h: r.height };
});
await page.mouse.move(padBox.x + padBox.w * 0.45, padBox.y + padBox.h * 0.2);
await page.mouse.down();
for (let i = 0; i <= 10; i++) {
  await page.mouse.move(padBox.x + padBox.w * 0.45, padBox.y + padBox.h * (0.2 + i * 0.05));
}
await page.mouse.up();
const drawn = await page.evaluate(() => ({
  vectorSum: currentVector ? Array.from(currentVector).reduce((s, v) => s + v, 0) : 0,
  outChanged: lastOut && lastOut.some((v) => v > 0.01)
}));
console.log("DRAWN on pad:", JSON.stringify(drawn));

// --- backprop step in browser ---
const bp = await page.evaluate(() => {
  document.getElementById("play-target").value = "1";
  window.__play.backprop();
  const pc = Chart.getChart("play-chart");
  return {
    stepPoints: pc.data.datasets[0].data.length,
    confPoints: pc.data.datasets[1].data.length,
    gradients: networkState.gradients ? networkState.gradients.map((g) => g.length) : null,
    info: document.getElementById("play-info").textContent.slice(0, 80)
  };
});
console.log("BACKPROP:", JSON.stringify(bp));
await page.screenshot({ path: "test/shot-playground.png" });

// --- random test sample loads into pad ---
await page.click("#btn-random-sample");
await new Promise((r) => setTimeout(r, 1200));
const sampleLoad = await page.evaluate(() => ({
  thumbs: document.getElementById("sample-strip").children.length,
  info: document.getElementById("play-info").textContent.slice(0, 90)
}));
console.log("SAMPLE STRIP:", JSON.stringify(sampleLoad));

// --- model upload: invalid file rejected ---
await page.evaluate(() => {
  const dt = new DataTransfer();
  dt.items.add(new File(['{"architecture":[10,10]}'], "bad.json", { type: "application/json" }));
  const inp = document.getElementById("model-upload");
  inp.files = dt.files;
  inp.dispatchEvent(new Event("change"));
});
await new Promise((r) => setTimeout(r, 400));
const badUpload = await page.evaluate(() => document.getElementById("play-info").textContent);
console.log("BAD UPLOAD:", JSON.stringify(badUpload));

// --- model upload: valid model.json ---
const modelJson = await page.evaluate(async () => {
  const res = await fetch("/socket.io/socket.io.js"); // any same-origin ok; model comes from server state
  // pull current model from a fresh connect snapshot instead:
  return null;
});
const modelText = await (await import("fs")).promises.readFile("model.json", "utf-8");
await page.evaluate(async (text) => {
  const dt = new DataTransfer();
  dt.items.add(new File([text], "model.json", { type: "application/json" }));
  const inp = document.getElementById("model-upload");
  inp.files = dt.files;
  inp.dispatchEvent(new Event("change"));
}, modelText);
await new Promise((r) => setTimeout(r, 1000));
const goodUpload = await page.evaluate(() => ({
  info: document.getElementById("play-info").textContent.slice(0, 90),
  arch: document.getElementById("arch-display").textContent,
  source: document.getElementById("play-source").textContent
}));
console.log("GOOD UPLOAD:", JSON.stringify(goodUpload));

// --- start training (small run so the epoch completes within the wait) ---
await page.evaluate(() => {
  document.getElementById("epochs-input").value = "1";
  document.getElementById("samples-input").value = "150";
  document.getElementById("speed-input").value = "100";
});
await page.click("#btn-train");
await new Promise((r) => setTimeout(r, 6000));

const during = await page.evaluate(() => {
  const loss = Chart.getChart("loss-chart");
  const acc = Chart.getChart("acc-chart");
  const ep = Chart.getChart("epoch-chart");
  return {
    mode: document.getElementById("mode-text").textContent,
    epochStat: document.getElementById("stat-epoch").textContent,
    lossStat: document.getElementById("stat-loss").textContent,
    lossPoints: loss.data.datasets[0].data.length,
    emaPoints: loss.data.datasets[2].data.length,
    accPoints: acc.data.datasets[0].data.length,
    epochBars: ep.data.datasets[0].data.length,
    progress: document.getElementById("progress-text").textContent,
    predChartMax: Math.max(...Chart.getChart("pred-chart").data.datasets[0].data)
  };
});
console.log("TRAINING:", JSON.stringify(during));
await page.screenshot({ path: "test/shot-training.png" });

const epochDone = during.epochBars === 1 && during.mode === "COMPLETE" && during.lossPoints > 10;
console.log("EPOCH CHARTED + COMPLETE:", epochDone);

// --- log-scale toggle ---
const logToggle = await page.evaluate(() => {
  const cb = document.getElementById("loss-log");
  cb.checked = true;
  cb.dispatchEvent(new Event("change"));
  const t = Chart.getChart("loss-chart").options.scales.y.type;
  cb.checked = false;
  cb.dispatchEvent(new Event("change"));
  return t;
});
console.log("LOG TOGGLE type:", logToggle);

// --- pause/resume + stop on a second run ---
await page.evaluate(() => {
  document.getElementById("epochs-input").value = "3";
  document.getElementById("samples-input").value = "2000";
  document.getElementById("speed-input").value = "60";
});
await page.click("#btn-train");
await new Promise((r) => setTimeout(r, 1500));
await page.click("#btn-pause");
await new Promise((r) => setTimeout(r, 500));
const paused = await page.evaluate(() => ({
  mode: document.getElementById("mode-text").textContent,
  pauseLabel: document.getElementById("btn-pause").textContent.trim()
}));
console.log("PAUSED:", JSON.stringify(paused));

await page.click("#btn-stop");
await new Promise((r) => setTimeout(r, 1200));
const stopped = await page.evaluate(() => ({
  mode: document.getElementById("mode-text").textContent,
  trainDisabled: document.getElementById("btn-train").disabled
}));
console.log("STOPPED:", JSON.stringify(stopped));

// --- reset clears charts ---
await page.click("#btn-reset");
await new Promise((r) => setTimeout(r, 800));
const reset = await page.evaluate(() => ({
  lossPoints: Chart.getChart("loss-chart").data.datasets[0].data.length,
  epochBars: Chart.getChart("epoch-chart").data.datasets[0].data.length,
  mode: document.getElementById("mode-text").textContent
}));
console.log("RESET:", JSON.stringify(reset));
await page.screenshot({ path: "test/shot-stopped.png" });

await browser.close();

const pass =
  initial.canvas && initial.litSamples > 500 &&
  initial.socketConnected && initial.netLib && initial.predChart && initial.playChart && initial.padExists &&
  tooltip.hidden === false &&
  typeof playPredict.pred === "number" && playPredict.outNonzero && playPredict.predChartData > 0 && playPredict.inputNonzero &&
  drawn.vectorSum > 5 && drawn.outChanged &&
  bp.stepPoints === 1 && bp.confPoints === 1 && bp.gradients && bp.gradients.join(",") === "3,3,10" &&
  sampleLoad.thumbs === 12 && sampleLoad.info.includes("test image") &&
  badUpload.includes("784") &&
  goodUpload.source.includes("browser") && goodUpload.arch === "784 → 3 → 3 → 10" &&
  epochDone && logToggle === "logarithmic" &&
  paused.pauseLabel.includes("RESUME") &&
  stopped.trainDisabled === false &&
  reset.lossPoints === 0 && reset.epochBars === 0 && reset.mode === "READY" &&
  errors.length === 0 && consoleErrs.length === 0;

console.log("PAGE ERRORS:", errors.length ? errors : "none");
console.log("CONSOLE ERRORS:", consoleErrs.length ? consoleErrs.slice(0, 5) : "none");
console.log(pass ? "E2E PASS" : "E2E FAIL");
process.exit(pass ? 0 : 1);
