import { io } from "socket.io-client";

const s = io("http://localhost:3000", { transports: ["websocket"] });
const seen = { model: 0, update: 0, epoch: 0, done: 0, state: 0, prep: 0 };
let last = null;

s.on("connect", () => {
  s.emit("train", { epochs: 2, lr: 0.1, samples: 30, speed: 200 });
});

s.on("model:loaded", (d) => {
  seen.model++;
  console.log("model:loaded arch=", d.architecture, "w0dims=", d.weights[0].length + "x" + d.weights[0][0].length, "acts=", d.activations.map(a => a.length).join(","));
});

s.on("network:update", (d) => {
  seen.update++;
  last = d;
  if (seen.update === 1) {
    console.log("update#1 loss=", d.loss, "pred=", d.prediction, "label=", d.label, "inputLen=", d.input.length, "grads=", d.gradients ? d.gradients.map(g => g.length).join(",") : null);
  }
});

s.on("training:epoch", (e) => { seen.epoch++; console.log("epoch:", JSON.stringify(e)); });
s.on("training:state", (t) => { seen.state++; console.log("state:", JSON.stringify(t)); });
s.on("training:done", (d) => { seen.done++; console.log("done:", JSON.stringify(d)); });
s.on("data:prep", (p) => { seen.prep++; if (seen.prep === 1) console.log("prep:", JSON.stringify(p)); });

setTimeout(() => {
  console.log("RESULT:", JSON.stringify(seen));
  if (last) console.log("LAST: step=", last.step, "loss=", last.loss, "avgLoss=", last.avgLoss, "acc=", last.accuracy, "epoch=", last.epoch);
  const ok = seen.model >= 1 && seen.update >= 50 && seen.epoch === 2 && seen.done === 1 && last && last.input.length === 784;
  console.log(ok ? "SMOKE PASS" : "SMOKE FAIL");
  process.exit(ok ? 0 : 1);
}, 90000);
