import { parentPort, workerData } from "worker_threads";

if (!parentPort) {
    throw new Error("turboWorker.js must be run as a worker_thread");
}

const { layerSizes, weightCount, inputLen } = workerData;
const layerCount = layerSizes.length - 1;

// Flat layout: [W0 (fanOut*fanIn), b0 (fanOut), W1, b1, ...]
const wOffsets = [];
const bOffsets = [];
{
    let o = 0;
    for (let l = 0; l < layerCount; l++) {
        const n = layerSizes[l] * layerSizes[l + 1];
        wOffsets.push(o);
        o += n;
        bOffsets.push(o);
        o += layerSizes[l + 1];
    }
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

/**
 * Flat forward+backward on summed gradients. Weights stay in Float64Array views.
 */
function accumulateSample(weights, sampleInput, label, gradSum) {
    // Forward
    const acts = [sampleInput];
    for (let l = 0; l < layerCount; l++) {
        const fanIn = layerSizes[l];
        const fanOut = layerSizes[l + 1];
        const wBase = wOffsets[l];
        const bBase = bOffsets[l];
        const prev = acts[l];
        const cur = new Float64Array(fanOut);
        for (let j = 0; j < fanOut; j++) {
            let sum = weights[bBase + j];
            const rowOff = wBase + j * fanIn;
            for (let i = 0; i < fanIn; i++) {
                sum += prev[i] * weights[rowOff + i];
            }
            cur[j] = sigmoid(sum);
        }
        acts.push(cur);
    }

    const out = acts[layerCount];
    const outSize = layerSizes[layerCount];
    let loss = 0;
    let pred = 0;
    let best = -Infinity;
    for (let j = 0; j < outSize; j++) {
        const t = j === label ? 1 : 0;
        const d = out[j] - t;
        loss += d * d;
        if (out[j] > best) {
            best = out[j];
            pred = j;
        }
    }
    loss /= outSize;

    // Output deltas
    const deltas = new Array(layerCount);
    deltas[layerCount - 1] = new Float64Array(outSize);
    for (let j = 0; j < outSize; j++) {
        const t = j === label ? 1 : 0;
        const act = out[j];
        deltas[layerCount - 1][j] = (act - t) * act * (1 - act);
    }

    // Hidden deltas
    for (let l = layerCount - 2; l >= 0; l--) {
        const curSize = layerSizes[l + 1];   // activations at layer l+1
        const nextSize = layerSizes[l + 2];  // activations at layer l+2
        const wBase = wOffsets[l + 1];       // W for layer l+1: (nextSize x curSize)
        const nextD = deltas[l + 1];
        const hidden = acts[l + 1];
        const d = new Float64Array(curSize);
        for (let j = 0; j < curSize; j++) {
            let error = 0;
            for (let k = 0; k < nextSize; k++) {
                error += weights[wBase + k * curSize + j] * nextD[k];
            }
            const act = hidden[j];
            d[j] = error * act * (1 - act);
        }
        deltas[l] = d;
    }

    // Accumulate grads
    for (let l = 0; l < layerCount; l++) {
        const fanIn = layerSizes[l];
        const fanOut = layerSizes[l + 1];
        const wBase = wOffsets[l];
        const bBase = bOffsets[l];
        const prev = acts[l];
        const d = deltas[l];
        const gw = gradSum.w[l];
        const gb = gradSum.b[l];
        for (let j = 0; j < fanOut; j++) {
            const dj = d[j];
            const rowOff = wBase + j * fanIn;
            const grow = gw[j];
            for (let i = 0; i < fanIn; i++) {
                grow[i] += dj * prev[i];
            }
            gb[j] += dj;
        }
    }

    return { loss, pred };
}

parentPort.on("message", (msg) => {
    const { id, weightsSab, inputsSab, indices, labels, count } = msg;

    try {
        const weights = new Float64Array(weightsSab);
        const inputs = new Float32Array(inputsSab);

        // Alloc grad buffers once per message
        const gradSum = { w: [], b: [] };
        for (let l = 0; l < layerCount; l++) {
            const fanIn = layerSizes[l];
            const fanOut = layerSizes[l + 1];
            const m = [];
            for (let j = 0; j < fanOut; j++) m.push(new Float64Array(fanIn));
            gradSum.w.push(m);
            gradSum.b.push(new Float64Array(fanOut));
        }

        let batchLoss = 0;
        let correct = 0;

        for (let s = 0; s < count; s++) {
            const row = indices[s];
            const input = inputs.subarray(row * inputLen, (row + 1) * inputLen);
            const label = labels[s];

            const { loss, pred } = accumulateSample(weights, input, label, gradSum);
            batchLoss += loss;
            if (pred === label) correct++;
        }

        // Pack summed grads
        let gn = 0;
        for (let l = 0; l < layerCount; l++) {
            gn += layerSizes[l] * layerSizes[l + 1] + layerSizes[l + 1];
        }
        const flatGrads = new Float64Array(gn);
        let o = 0;
        for (let l = 0; l < layerCount; l++) {
            for (let j = 0; j < gradSum.w[l].length; j++) {
                const row = gradSum.w[l][j];
                for (let i = 0; i < row.length; i++) flatGrads[o++] = row[i];
            }
            for (let j = 0; j < gradSum.b[l].length; j++) flatGrads[o++] = gradSum.b[l][j];
        }

        parentPort.postMessage({
            id,
            flatGrads,
            batchLoss,
            correct,
            count,
            lastDeltas: null
        }, [flatGrads.buffer]);
    } catch (err) {
        parentPort.postMessage({
            id,
            error: err?.message || String(err),
            flatGrads: null,
            batchLoss: 0,
            correct: 0,
            count: 0,
            lastDeltas: null
        });
    }
});
