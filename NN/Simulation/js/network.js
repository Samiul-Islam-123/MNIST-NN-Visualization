/**
 * Neural Network Engine for MNIST Simulation
 * Exactly mirrors NN/Network.js with full browser & ES module compatibility.
 */

export class Network {
    constructor(layerSizes) {
        if (
            !Array.isArray(layerSizes) ||
            layerSizes.length < 2 ||
            !layerSizes.every(n => Number.isInteger(n) && n > 0)
        ) {
            throw new Error("Provide at least two positive integer layer sizes.");
        }

        this.layerSizes = [...layerSizes];
        this.NN = [];
        this.biases = [];
        this.init();
    }

    init() {
        this.NN = [];
        this.biases = [];

        // Input layer
        this.NN.push(Array(this.layerSizes[0]).fill(0));

        for (let l = 0; l < this.layerSizes.length - 1; l++) {
            const fanIn = this.layerSizes[l];
            const fanOut = this.layerSizes[l + 1];

            // Xavier / Glorot initialization
            const limit = Math.sqrt(6 / (fanIn + fanOut));

            const weights = Array.from(
                { length: fanOut },
                () => Array.from(
                    { length: fanIn },
                    () => (Math.random() * 2 - 1) * limit
                )
            );

            const neurons = Array(fanOut).fill(0);
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
        // Save the input activations
        this.NN[0] = Array.isArray(input) ? input : Array.from(input);

        // Feed forward through each layer
        for (let l = 0; l < this.layerSizes.length - 1; l++) {
            const weights = this.NN[2 * l + 1];
            const previousLayer = this.NN[2 * l];
            const currentLayer = [];

            for (let j = 0; j < weights.length; j++) {
                let sum = this.biases[l][j];
                const wRow = weights[j];
                for (let i = 0; i < previousLayer.length; i++) {
                    sum += previousLayer[i] * wRow[i];
                }
                currentLayer.push(this.sigmoid(sum));
            }

            this.NN[2 * l + 2] = currentLayer;
        }

        return this.NN[this.NN.length - 1];
    }

    backpropagation(target, learningRate = 0.1) {
        const layerCount = this.layerSizes.length - 1;
        const deltas = Array(layerCount).fill(null);

        // STEP 1: Output layer error
        const output = this.NN[this.NN.length - 1];
        deltas[layerCount - 1] = [];

        for (let j = 0; j < output.length; j++) {
            const activation = output[j];
            const error = activation - target[j];
            const derivative = activation * (1 - activation);
            const delta = error * derivative;
            deltas[layerCount - 1].push(delta);
        }

        // STEP 2: Hidden layer errors (backpropagate deltas)
        for (let l = layerCount - 2; l >= 0; l--) {
            const hiddenActivations = this.NN[2 * l + 2];
            const nextWeights = this.NN[2 * (l + 1) + 1];
            const nextDeltas = deltas[l + 1];

            deltas[l] = [];

            for (let j = 0; j < hiddenActivations.length; j++) {
                let error = 0;
                for (let k = 0; k < nextWeights.length; k++) {
                    error += nextWeights[k][j] * nextDeltas[k];
                }
                const activation = hiddenActivations[j];
                const derivative = activation * (1 - activation);
                const delta = error * derivative;
                deltas[l].push(delta);
            }
        }

        // STEP 3: Update weights and biases
        for (let l = 0; l < layerCount; l++) {
            const weights = this.NN[2 * l + 1];
            const previousLayer = this.NN[2 * l];
            const layerDeltas = deltas[l];

            for (let j = 0; j < weights.length; j++) {
                const deltaJ = layerDeltas[j];
                const wRow = weights[j];
                for (let i = 0; i < previousLayer.length; i++) {
                    wRow[i] -= learningRate * deltaJ * previousLayer[i];
                }
                this.biases[l][j] -= learningRate * deltaJ;
            }
        }

        return deltas;
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

    extractWeights() {
        return this.NN
            .filter((_, index) => index % 2 === 1)
            .map(matrix => matrix.map(row => [...row]));
    }

    extractBiases() {
        return this.biases.map(layer => [...layer]);
    }

    extractActivations() {
        return this.NN
            .filter((_, index) => index % 2 === 0)
            .map(layer => [...layer]);
    }

    toJSON() {
        return {
            architecture: [...this.layerSizes],
            layerSizes: [...this.layerSizes],
            weights: this.extractWeights(),
            biases: this.extractBiases()
        };
    }

    loadState(data) {
        const sizes = data.architecture || data.layerSizes;
        if (!sizes || !data.weights || !data.biases) {
            throw new Error("Invalid model object: missing architecture, weights, or biases.");
        }
        this.layerSizes = [...sizes];
        this.biases = data.biases.map(layer => [...layer]);

        this.NN = [];
        this.NN.push(Array(this.layerSizes[0]).fill(0));

        for (let l = 0; l < this.layerSizes.length - 1; l++) {
            this.NN.push(data.weights[l].map(row => [...row]));
            this.NN.push(Array(this.layerSizes[l + 1]).fill(0));
        }

        this.input = this.NN[0];
        this.output = this.NN[this.NN.length - 1];
        return this;
    }

    static fromJSON(data) {
        const sizes = data.architecture || data.layerSizes;
        const net = new Network(sizes);
        net.loadState(data);
        return net;
    }
}

if (typeof window !== "undefined") {
    window.Network = Network;
}
