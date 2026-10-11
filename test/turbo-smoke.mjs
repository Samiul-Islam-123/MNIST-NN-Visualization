import { Network } from "../NN/Network.js";

// tiny synthetic regression-style task: learn identity-ish mapping on 784->3->3->10
function makeData(n) {
  const data = [];
  for (let i = 0; i < n; i++) {
    const label = i % 10;
    const input = new Array(784).fill(0);
    // sparse pattern per digit
    input[label * 7 + 3] = 1;
    input[label * 7 + 11] = 0.8;
    input[100 + label] = 1;
    data.push({ input, label });
  }
  return data;
}

function runSequential(net, data, lr, epochs) {
  let totalLoss = 0, count = 0;
  for (let e = 0; e < epochs; e++) {
    for (const s of data) {
      const target = Array(10).fill(0);
      target[s.label] = 1;
      const output = net.predict(s.input);
      let loss = 0;
      for (let j = 0; j < output.length; j++) {
        const d = output[j] - target[j];
        loss += d * d;
      }
      loss /= output.length;
      net.backpropagation(target, lr);
      totalLoss += loss;
      count++;
    }
  }
  return totalLoss / count;
}

function accuracy(net, data) {
  let correct = 0;
  for (const s of data) {
    const out = net.predict(s.input);
    if (net.getPrediction(out) === s.label) correct++;
  }
  return (correct / data.length) * 100;
}

const data = makeData(500);
const epochs = 4;

// sequential baseline
const netSeq = new Network([784, 3, 3, 10]);
const t0 = Date.now();
const seqLoss = runSequential(netSeq, data, 0.5, epochs);
const seqMs = Date.now() - t0;
const seqAcc = accuracy(netSeq, data);

// turbo
const netTurbo = new Network([784, 3, 3, 10]);
// share weights so we compare optimization path fairly? No — turbo averages batch grads so it's different SGD.
// Just check turbo runs, reduces loss, and uses workers.
const t1 = Date.now();
let batches = 0;
await netTurbo.trainTurbo(epochs, data, null, 0.5, () => { batches++; }, {
  workers: 4,
  batchSize: 32
});
const turboMs = Date.now() - t1;
const turboAcc = accuracy(netTurbo, data);

// loss before turbo on same init
const netBefore = Network.fromJSON(netTurbo.toJSON()); // after
// measure loss reduction: train 0 epochs vs after
const net0 = new Network([784, 3, 3, 10]);
// clone net0 weights into another by training 0
// Instead: compare average loss early vs late via onUpdate
const losses = [];
const netLoss = new Network([784, 3, 3, 10]);
await netLoss.trainTurbo(1, data, null, 0.5, (info) => {
  losses.push(info.batchLoss);
}, { workers: 4, batchSize: 32 });

const first = losses.slice(0, 3).reduce((a, b) => a + b, 0) / 3;
const last = losses.slice(-3).reduce((a, b) => a + b, 0) / 3;

console.log(JSON.stringify({
  seqMs, turboMs, batches,
  seqLoss: +seqLoss.toFixed(5),
  seqAcc: +seqAcc.toFixed(1),
  turboAcc: +turboAcc.toFixed(1),
  firstBatchLoss: +first.toFixed(5),
  lastBatchLoss: +last.toFixed(5),
  lossDropped: last < first,
  accOk: turboAcc > 20
}, null, 2));

if (!(last < first)) {
  console.error("FAIL: turbo loss did not drop");
  process.exit(1);
}
if (batches < 10) {
  console.error("FAIL: too few batches", batches);
  process.exit(1);
}
console.log("PASS turbo smoke");
