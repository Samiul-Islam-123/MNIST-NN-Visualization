import { io } from "socket.io-client";

const s = io("http://localhost:3000", { transports: ["websocket"] });
const log = [];
let updates = 0;
let resetSeen = false;
let baselineW = null;

s.on("connect", () => {
  s.emit("train", { epochs: 5, lr: 0.1, samples: 500, speed: 40 });
  setTimeout(async () => {
    // 1. pause
    s.emit("pause");
    await wait(600);
    const afterPause = updates;
    await wait(800);
    const stillPaused = updates;
    log.push(`pause holds: ${stillPaused === afterPause} (${afterPause}->${stillPaused})`);

    // 2. step while paused
    s.emit("step");
    await wait(900);
    log.push(`step advances: ${updates > afterPause} (+${updates - afterPause})`);

    // 3. resume
    s.emit("resume");
    await wait(1200);
    log.push(`resume flows: ${updates > stillPaused + 1}`);

    // 4. stop
    s.emit("stop");
    await wait(1000);
    const stopA = updates;
    await wait(800);
    log.push(`stop halts: ${updates === stopA}`);

    // 5. capture weights, reset
    s.once("model:loaded", (d) => {
      resetSeen = true;
      baselineW = d.weights[0][0].slice(0, 5);
      log.push(`reset model: arch=${d.architecture} reset=${d.reset} acts0=${d.activations[0].every(v => v === 0) ? "zeros" : "nonzero"}`);
    });
    s.emit("reset");
    await wait(700);

    // 6. idle step after reset (fresh random net, one sample)
    const beforeStep = updates;
    s.emit("step");
    await wait(1200);
    log.push(`idle step after reset: ${updates > beforeStep} (+${updates - beforeStep})`);

    const joined = log.join("\n");
    const pass =
      joined.includes("pause holds: true") && joined.includes("step advances: true") &&
      joined.includes("resume flows: true") && joined.includes("stop halts: true") &&
      joined.includes("idle step after reset: true") &&
      resetSeen && baselineW;
    console.log(log.join("\n"));
    console.log("updates total:", updates);
    console.log(pass ? "CONTROLS PASS" : "CONTROLS FAIL");
    process.exit(pass ? 0 : 1);
  }, 2500);
});

s.on("network:update", () => updates++);
s.on("training:state", (t) => log.push("state: " + JSON.stringify(t)));
s.on("training:done", (d) => log.push("done: " + JSON.stringify(d)));

function wait(ms) { return new Promise((r) => setTimeout(r, ms)); }
