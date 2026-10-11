import fs from "fs";

export class Network {
    constructor(layerSizes) {
        if (
            !Array.isArray(layerSizes) ||
            layerSizes.length < 2 ||
            !layerSizes.every(
                n => Number.isInteger(n) && n > 0
            )
        ) {
            throw new Error(
                "Provide at least two positive integer layer sizes."
            );
        }

        this.layerSizes = [...layerSizes];

        // Your exact alternating network representation.
        this.NN = [];

        this.init();
    }


    init() {
        this.NN = [];
        this.biases = [];

        // Input layer
        this.NN.push(
            Array(this.layerSizes[0]).fill(0)
        );

        for (let l = 0; l < this.layerSizes.length - 1; l++) {
            const fanIn = this.layerSizes[l];
            const fanOut = this.layerSizes[l + 1];

            // Initialize weights
            const limit = Math.sqrt(6 / (fanIn + fanOut));

            const weights = Array.from(
                { length: fanOut },
                () => Array.from(
                    { length: fanIn },
                    () => (Math.random() * 2 - 1) * limit
                )
            );

            // Initialize neurons
            const neurons = Array(fanOut).fill(0);

            // One bias for each destination neuron
            const bias = Array(fanOut).fill(0);

            this.NN.push(weights);
            this.NN.push(neurons);

            this.biases.push(bias);
        }

        this.input = this.NN[0];
        this.output = this.NN[this.NN.length - 1];

        return this;
    }

    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }


    predict(input) {

        // Save the input
        this.NN[0] = input;

        // Go through each layer
        for (let l = 0; l < this.layerSizes.length - 1; l++) {

            let weights = this.NN[2 * l + 1];
            let previousLayer = this.NN[2 * l];
            let currentLayer = [];

            // Calculate each neuron in the next layer
            for (let j = 0; j < weights.length; j++) {

                let sum = this.biases[l][j];

                // Multiply each input by its weight
                for (let i = 0; i < previousLayer.length; i++) {
                    sum += previousLayer[i] * weights[j][i];
                }

                // Sigmoid activation
                let activation = this.sigmoid(sum);

                currentLayer.push(activation);
            }

            // Save the calculated neurons
            this.NN[2 * l + 2] = currentLayer;
        }

        // Return final output
        return this.NN[this.NN.length - 1];
    }


    backpropagation(target, learningRate = 0.1) {

        let layerCount = this.layerSizes.length - 1;

        // Store the error for each layer
        let deltas = Array(layerCount).fill(null);

        // STEP 1: Calculate output layer error

        let output = this.NN[this.NN.length - 1];

        for (let j = 0; j < output.length; j++) {

            let activation = output[j];

            let error = activation - target[j];

            let derivative = activation * (1 - activation);

            let delta = error * derivative;

            if (deltas[layerCount - 1] === null) {
                deltas[layerCount - 1] = [];
            }

            deltas[layerCount - 1].push(delta);
        }


        // STEP 2: Calculate hidden layer errors

        for (let l = layerCount - 2; l >= 0; l--) {

            let hiddenActivations = this.NN[2 * l + 2];

            let nextWeights = this.NN[2 * (l + 1) + 1];

            let nextDeltas = deltas[l + 1];

            deltas[l] = [];

            for (let j = 0; j < hiddenActivations.length; j++) {

                let error = 0;

                // Collect error coming from the next layer
                for (let k = 0; k < nextWeights.length; k++) {

                    error += nextWeights[k][j] * nextDeltas[k];

                }

                let activation = hiddenActivations[j];

                let derivative = activation * (1 - activation);

                let delta = error * derivative;

                deltas[l].push(delta);
            }
        }

        // STEP 3: Update weights and biases

        for (let l = 0; l < layerCount; l++) {

            let weights = this.NN[2 * l + 1];

            let previousLayer = this.NN[2 * l];

            for (let j = 0; j < weights.length; j++) {

                for (let i = 0; i < previousLayer.length; i++) {

                    let change =
                        learningRate *
                        deltas[l][j] *
                        previousLayer[i];

                    weights[j][i] -= change;

                }

                // Update the bias of this neuron
                this.biases[l][j] -= learningRate * deltas[l][j];

            }
        }

        return deltas;
    }




    extractWeights() {
        return this.NN
            .filter((_, index) => index % 2 === 1)
            .map(matrix => matrix.map(row => [...row]));
    }

    extractBiases() {
        return this.biases.map(layer => [...layer]);
    }

    /**
     * Forward pass + gradient computation without mutating weights.
     * @returns {{ weightGrads: number[][][], biasGrads: number[][], deltas: number[][], loss: number, prediction: number }}
     */
    computeGradients(input, target) {
        const output = this.predict(input);
        const layerCount = this.layerSizes.length - 1;
        const deltas = Array(layerCount);

        let loss = 0;
        for (let j = 0; j < output.length; j++) {
            const difference = output[j] - target[j];
            loss += difference * difference;
        }
        loss /= output.length;

        deltas[layerCount - 1] = output.map((activation, j) =>
            (activation - target[j]) * activation * (1 - activation)
        );

        for (let l = layerCount - 2; l >= 0; l--) {
            const hiddenActivations = this.NN[2 * l + 2];
            const nextWeights = this.NN[2 * (l + 1) + 1];
            const nextDeltas = deltas[l + 1];

            deltas[l] = hiddenActivations.map((activation, j) => {
                let error = 0;
                for (let k = 0; k < nextWeights.length; k++) {
                    error += nextWeights[k][j] * nextDeltas[k];
                }
                return error * activation * (1 - activation);
            });
        }

        const weightGrads = [];
        const biasGrads = [];
        for (let l = 0; l < layerCount; l++) {
            const previousLayer = this.NN[2 * l];
            weightGrads.push(
                deltas[l].map(delta => previousLayer.map(activation => delta * activation))
            );
            biasGrads.push([...deltas[l]]);
        }

        return {
            weightGrads,
            biasGrads,
            deltas,
            loss,
            prediction: this.getPrediction(output)
        };
    }

    /**
     * Applies averaged gradients: weights -= lr * grad (single master update).
     */
    applyGradients(weightGrads, biasGrads, learningRate) {
        for (let l = 0; l < weightGrads.length; l++) {
            const weights = this.NN[2 * l + 1];
            for (let j = 0; j < weights.length; j++) {
                for (let i = 0; i < weights[j].length; i++) {
                    weights[j][i] -= learningRate * weightGrads[l][j][i];
                }
                this.biases[l][j] -= learningRate * biasGrads[l][j];
            }
        }
    }

    /**
     * Multi-core mini-batch SGD. One master network lives on this thread;
     * workers only compute gradients from a SharedArrayBuffer snapshot and return them.
     * The main thread averages gradients and applies a single synchronized update.
     *
     * @param {number} epochs
     * @param {Array<{ input: ArrayLike<number>, label: number }>} trainingData
     * @param {null} _inputLoader unused (inputs must be pre-loaded)
     * @param {number} learningRate
     * @param {(info: object) => void} onUpdate per-batch callback
     * @param {{ workers?: number, batchSize?: number, shouldStop?: () => boolean, onBatchStart?: () => Promise<boolean>, onEpoch?: (e: object) => void }} options
     */
    async trainTurbo(
        epochs,
        trainingData,
        _inputLoader = null,
        learningRate = 0.1,
        onUpdate = null,
        options = {}
    ) {
        if (!Array.isArray(trainingData) || trainingData.length === 0) {
            throw new Error("trainTurbo requires a non-empty trainingData array with { input, label }.");
        }

        const { Worker } = await import("worker_threads");
        const os = await import("os");
        const path = await import("path");
        const { fileURLToPath } = await import("url");

        const workerCount = Math.max(
            1,
            options.workers ?? (os.availableParallelism?.() ?? os.cpus().length)
        );
        const batchSize = Math.max(
            workerCount,
            options.batchSize ?? workerCount * 16
        );
        const shouldStop = options.shouldStop ?? (() => false);
        const onBatchStart = options.onBatchStart ?? (async () => true);

        const workerPath = path.join(
            path.dirname(fileURLToPath(import.meta.url)),
            "turboWorker.js"
        );

        const layerSizes = [...this.layerSizes];
        const inputLen = layerSizes[0];
        let weightCount = 0;
        for (let l = 0; l < layerSizes.length - 1; l++) {
            weightCount += layerSizes[l] * layerSizes[l + 1] + layerSizes[l + 1];
        }

        // Pack the full dataset once into a SharedArrayBuffer (zero-copy to workers)
        const nAll = trainingData.length;
        const inputsSab = new SharedArrayBuffer(nAll * inputLen * 4);
        const inputsView = new Float32Array(inputsSab);
        for (let i = 0; i < nAll; i++) {
            inputsView.set(trainingData[i].input, i * inputLen);
        }

        // Shared weights buffer — master writes, workers read each batch
        const weightsSab = new SharedArrayBuffer(weightCount * 8);
        const weightsView = new Float64Array(weightsSab);

        const writeWeights = () => {
            let o = 0;
            for (let l = 0; l < layerSizes.length - 1; l++) {
                const w = this.NN[2 * l + 1];
                for (let j = 0; j < w.length; j++) {
                    for (let i = 0; i < w[j].length; i++) weightsView[o++] = w[j][i];
                }
                for (let j = 0; j < this.biases[l].length; j++) weightsView[o++] = this.biases[l][j];
            }
        };

        const workers = Array.from(
            { length: workerCount },
            () => new Worker(workerPath, {
                workerData: { layerSizes, weightCount, inputLen, maxSamples: batchSize }
            })
        );

        let msgId = 0;
        const pending = new Map();

        const onWorkerMessage = (msg) => {
            const entry = pending.get(msg.id);
            if (!entry) return;
            pending.delete(msg.id);
            entry.resolve(msg);
        };

        for (const worker of workers) {
            worker.on("message", onWorkerMessage);
            worker.on("error", (err) => {
                for (const [, entry] of pending) entry.reject(err);
                pending.clear();
            });
        }

        const runOnWorker = (worker, payload) =>
            new Promise((resolve, reject) => {
                const id = ++msgId;
                pending.set(id, { resolve, reject });
                worker.postMessage({ ...payload, id });
            });

        let globalBatch = 0;
        let stopped = false;

        try {
            for (let epoch = 0; epoch < epochs && !stopped; epoch++) {
                const order = trainingData.map((_, i) => i);
                for (let i = order.length - 1; i > 0; i--) {
                    const j = Math.floor(Math.random() * (i + 1));
                    [order[i], order[j]] = [order[j], order[i]];
                }

                let epochLoss = 0;
                let epochCorrect = 0;
                let epochCount = 0;

                for (let b = 0; b < order.length; b += batchSize) {
                    if (shouldStop()) {
                        stopped = true;
                        break;
                    }

                    const keepGoing = await onBatchStart();
                    if (!keepGoing || shouldStop()) {
                        stopped = true;
                        break;
                    }

                    const slice = order.slice(b, b + batchSize);
                    writeWeights();

                    // Split batch indices across workers; pack labels as Int32Array
                    const perWorker = workers.map((worker, wi) => {
                        const idxs = [];
                        for (let i = wi; i < slice.length; i += workerCount) {
                            idxs.push(slice[i]);
                        }
                        if (idxs.length === 0) {
                            return Promise.resolve({
                                flatGrads: null,
                                batchLoss: 0,
                                correct: 0,
                                count: 0,
                                lastDeltas: null
                            });
                        }
                        const labels = new Int32Array(idxs.map(i => trainingData[i].label));
                        return runOnWorker(worker, {
                            weightsSab,
                            inputsSab,
                            indices: Int32Array.from(idxs),
                            labels,
                            count: idxs.length
                        });
                    });

                    const results = await Promise.all(perWorker);

                    // Average gradients across all workers → one update on master
                    const n = results.reduce((s, r) => s + r.count, 0) || 1;
                    const avgGrads = new Float64Array(weightCount);
                    let batchLoss = 0;
                    let correct = 0;
                    let lastDeltas = null;
                    let lastLabel = null;

                    for (let wi = 0; wi < results.length; wi++) {
                        const r = results[wi];
                        if (!r.flatGrads) continue;
                        batchLoss += r.batchLoss;
                        correct += r.correct;
                        if (r.lastDeltas) {
                            lastDeltas = r.lastDeltas;
                            // approximate last label from worker chunk
                        }
                        for (let i = 0; i < weightCount; i++) avgGrads[i] += r.flatGrads[i];
                    }

                    for (let i = 0; i < weightCount; i++) avgGrads[i] /= n;

                    // Apply averaged grads to master nested arrays
                    // Worker packs interleaved: W0, b0, W1, b1, ...
                    {
                        const weightGrads = [];
                        const biasGrads = [];
                        let o = 0;
                        for (let l = 0; l < layerSizes.length - 1; l++) {
                            const fanIn = layerSizes[l];
                            const fanOut = layerSizes[l + 1];
                            const m = [];
                            for (let j = 0; j < fanOut; j++) {
                                const row = new Array(fanIn);
                                for (let i = 0; i < fanIn; i++) row[i] = avgGrads[o++];
                                m.push(row);
                            }
                            weightGrads.push(m);
                            const b = new Array(fanOut);
                            for (let j = 0; j < fanOut; j++) b[j] = avgGrads[o++];
                            biasGrads.push(b);
                        }
                        this.applyGradients(weightGrads, biasGrads, learningRate);
                    }

                    epochLoss += batchLoss;
                    epochCorrect += correct;
                    epochCount += n;
                    globalBatch++;

                    if (onUpdate) {
                        const lastIdx = slice[slice.length - 1];
                        const lastSample = trainingData[lastIdx];
                        const output = this.predict(lastSample.input);
                        onUpdate({
                            mode: "turbo",
                            epoch: epoch + 1,
                            totalEpochs: epochs,
                            batch: globalBatch,
                            batchSize: n,
                            step: epochCount,
                            totalSteps: order.length,
                            workers: workerCount,
                            batchLoss: batchLoss / n,
                            averageLoss: epochLoss / epochCount,
                            accuracy: (epochCorrect / epochCount) * 100,
                            prediction: this.getPrediction(output),
                            label: lastSample.label,
                            deltas: lastDeltas,
                            input: Array.from(lastSample.input),
                            activations: this.NN
                                .filter((_, i) => i % 2 === 0)
                                .map(layer => [...layer])
                        });
                    }
                }

                if (options.onEpoch && !stopped) {
                    options.onEpoch({
                        epoch: epoch + 1,
                        totalEpochs: epochs,
                        loss: epochLoss / Math.max(1, epochCount),
                        accuracy: (epochCorrect / Math.max(1, epochCount)) * 100,
                        samples: epochCount
                    });
                }
            }
        } finally {
            await Promise.all(workers.map(w => w.terminate()));
        }

        return { stopped };
    }


    async train(
        epochs,
        trainingData,
        inputLoader,
        learningRate = 0.1,
        onUpdate = null
    ) {
        const spinnerFrames = ["|", "/", "-", "\\"];
        let spinnerIndex = 0;

        const renderDashboard = (lines, firstRender = false) => {
            if (!firstRender) {
                // Move back to the beginning of the previous 3-line display
                process.stdout.write("\x1B[3A");
            }

            for (const line of lines) {
                // Clear the current terminal line, then write new content
                process.stdout.write("\x1B[2K" + line + "\n");
            }
        };

        // Draw the dashboard once so later updates overwrite it.
        renderDashboard([
            "MNIST NEURAL NETWORK TRAINING",
            "Preparing training...",
            "Please wait..."
        ], true);

        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalLoss = 0;
            let correctPredictions = 0;

            // Shuffle the training data
            let shuffledData = [...trainingData];

            for (let i = shuffledData.length - 1; i > 0; i--) {
                let j = Math.floor(Math.random() * (i + 1));

                let temp = shuffledData[i];
                shuffledData[i] = shuffledData[j];
                shuffledData[j] = temp;
            }

            for (let i = 0; i < shuffledData.length; i++) {
                let sample = shuffledData[i];

                let imageName = sample.filePath.split(/[\\/]/).pop();

                let input = await inputLoader(sample);

                let target = Array(
                    this.layerSizes[this.layerSizes.length - 1]
                ).fill(0);

                target[sample.target] = 1;

                // Forward pass
                let output = this.predict(input);

                // Calculate loss
                let sampleLoss = 0;

                for (let j = 0; j < output.length; j++) {
                    let difference = output[j] - target[j];
                    sampleLoss += difference * difference;
                }

                sampleLoss /= output.length;
                totalLoss += sampleLoss;

                // Check prediction before updating weights
                let prediction = this.getPrediction(output);
                let isCorrect = prediction === sample.target;

                if (isCorrect) {
                    correctPredictions++;
                }

                let progress = ((i + 1) / shuffledData.length) * 100;
                let averageLoss = totalLoss / (i + 1);
                let accuracy =
                    (correctPredictions / (i + 1)) * 100;

                // Build the progress bar
                const barWidth = 24;
                const filled = Math.floor(
                    (progress / 100) * barWidth
                );

                const progressBar =
                    "[" +
                    "#".repeat(filled) +
                    "-".repeat(barWidth - filled) +
                    "]";

                const spinner =
                    spinnerFrames[spinnerIndex % spinnerFrames.length];

                spinnerIndex++;

                // Update the same three terminal lines
                renderDashboard([
                    ` ${spinner} MNIST TRAINING | Epoch ${epoch + 1}/${epochs}`,
                    ` ${progressBar} ${progress.toFixed(2)}% | Image ${i + 1}/${shuffledData.length}`,
                    ` File: ${imageName} | Target: ${sample.target} | Predicted: ${prediction} | ${isCorrect ? "CORRECT" : "WRONG"} | Loss: ${sampleLoss.toFixed(5)} | Avg Loss: ${averageLoss.toFixed(5)} | Accuracy: ${accuracy.toFixed(2)}%`
                ]);

                // Update weights and biases
                this.backpropagation(target, learningRate);

                // Send live state to the WebSocket server / browser
                if (onUpdate) {
                    onUpdate({
                        epoch: epoch + 1,
                        totalEpochs: epochs,
                        step: i + 1,
                        totalSteps: shuffledData.length,
                        imageName,
                        progress,
                        loss: sampleLoss,
                        averageLoss,
                        accuracy,
                        prediction,
                        target: sample.target,

                        weights: this.NN
                            .filter((item, index) => index % 2 === 1)
                            .map(matrix =>
                                matrix.map(row => [...row])
                            ),

                        biases: this.biases.map(layer => [...layer]),

                        activations: this.NN
                            .filter((item, index) => index % 2 === 0)
                            .map(layer => [...layer])
                    });
                }
            }

            // Print a permanent summary after each epoch
            const epochLoss = totalLoss / shuffledData.length;
            const epochAccuracy =
                (correctPredictions / shuffledData.length) * 100;

            console.log("");
            console.log(
                `Epoch ${epoch + 1}/${epochs} completed | ` +
                `Average Loss: ${epochLoss.toFixed(6)} | ` +
                `Accuracy: ${epochAccuracy.toFixed(2)}%`
            );
        }

        console.log("\nTRAINING COMPLETED");
    }


    getPrediction(output) {
        let highestValue = output[0];
        let predictedDigit = 0;

        for (let i = 1; i < output.length; i++) {
            if (output[i] > highestValue) {
                highestValue = output[i];
                predictedDigit = i;
            }
        }

        return predictedDigit;
    }


    display() {
        console.log("\n========== NEURAL NETWORK ==========\n");

        for (let i = 0; i < this.NN.length; i++) {
            const item = this.NN[i];

            // Detect whether this item is a weight matrix.
            const isWeights = Array.isArray(item[0]);

            console.log(`--- ${this.getName(i)} ---`);

            if (isWeights) {
                console.table(
                    item.map((row, index) => ({
                        Neuron: index,
                        ...Object.fromEntries(
                            row.map((weight, j) => [`w${j}`, weight])
                        )
                    }))
                );
            } else {
                console.table(
                    item.map((value, index) => ({
                        Neuron: index,
                        Activation: value
                    }))
                );
            }
        }

        console.log("====================================\n");
    }

    getName(index) {
        if (index === 0) return "INPUT";

        if (index === this.NN.length - 1) return "OUTPUT";

        return index % 2 === 1
            ? `WEIGHTS W${Math.floor(index / 2)}`
            : `HIDDEN LAYER H${index / 2 - 1}`;
    }

    /**
     * Serializes the neural network architecture, weights, and biases into a JSON-compatible object.
     */
    toJSON() {
        return {
            architecture: [...this.layerSizes],
            layerSizes: [...this.layerSizes],
            weights: this.NN
                .filter((item, index) => index % 2 === 1)
                .map(matrix => matrix.map(row => [...row])),
            biases: this.biases.map(layer => [...layer])
        };
    }

    /**
     * Saves the entire model architecture, weights, and biases to a JSON file.
     * @param {string} filePath - Absolute or relative path to save the JSON file.
     */
    save(filePath) {
        const data = this.toJSON();
        fs.writeFileSync(filePath, JSON.stringify(data, null, 2), "utf-8");
        return filePath;
    }

    /**
     * Loads a model from a saved JSON file and returns a new Network instance.
     * @param {string} filePath - Path to the saved model JSON file.
     */
    static load(filePath) {
        if (!fs.existsSync(filePath)) {
            throw new Error(`Model file not found at: ${filePath}`);
        }
        const raw = fs.readFileSync(filePath, "utf-8");
        const data = JSON.parse(raw);
        return Network.fromJSON(data);
    }

    /**
     * Reconstructs a Network instance from a parsed JSON model object.
     * @param {Object} data - Model object containing architecture/layerSizes, weights, and biases.
     */
    static fromJSON(data) {
        const sizes = data.architecture || data.layerSizes;
        if (!sizes || !data.weights || !data.biases) {
            throw new Error("Invalid model file: missing architecture/layerSizes, weights, or biases.");
        }
        const network = new Network(sizes);
        network.loadState(data);
        return network;
    }

    /**
     * Loads weights and biases into this network instance.
     * @param {Object} data - Model data object.
     */
    loadState(data) {
        const sizes = data.architecture || data.layerSizes;
        this.layerSizes = [...sizes];
        this.biases = data.biases.map(layer => [...layer]);

        this.NN = [];
        // Input activation placeholders
        this.NN.push(Array(this.layerSizes[0]).fill(0));

        for (let l = 0; l < this.layerSizes.length - 1; l++) {
            this.NN.push(data.weights[l].map(row => [...row]));
            this.NN.push(Array(this.layerSizes[l + 1]).fill(0));
        }

        this.input = this.NN[0];
        this.output = this.NN[this.NN.length - 1];
        return this;
    }
}