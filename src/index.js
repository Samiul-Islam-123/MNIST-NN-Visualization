import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { extractInfo } from "../image-processor/BasicProcessor.js";
import { Network } from "../NN/Network.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DATASET_DIR = path.join(__dirname, "../dataset/");
const MODEL_FILE = path.join(__dirname, "../model.json");

/**
 * Predict on random images from the test set using a loaded model.
 */
async function runTestPredictions(network, numSamples = 10) {
    const testDir = path.join(DATASET_DIR, "test");
    if (!fs.existsSync(testDir)) {
        console.error(`Test directory not found at: ${testDir}`);
        return;
    }

    const testFiles = fs.readdirSync(testDir).filter(file => file.endsWith(".png"));
    if (testFiles.length === 0) {
        console.warn("No PNG test images found.");
        return;
    }

    console.log(`\n=========================================================================================`);
    console.log(`TEST EVALUATION: PREDICTIONS ON RANDOM TEST IMAGES`);
    console.log(`=========================================================================================`);
    console.log(`Selecting ${numSamples} random test images out of ${testFiles.length} available...\n`);

    // Shuffle and pick random test samples
    const shuffled = [...testFiles].sort(() => 0.5 - Math.random());
    const samples = shuffled.slice(0, Math.min(numSamples, shuffled.length));

    let correctCount = 0;
    console.log(`-----------------------------------------------------------------------------------------`);
    console.log(` #  | Image Filename        | True Target | Predicted | Output Conf | Evaluation`);
    console.log(`-----------------------------------------------------------------------------------------`);

    for (let i = 0; i < samples.length; i++) {
        const file = samples[i];
        const filePath = path.join(testDir, file);
        const target = Number(file.split("_")[0]);

        const input = await extractInfo(filePath);
        const output = network.predict(input);
        const predicted = network.getPrediction(output);
        const isCorrect = predicted === target;

        if (isCorrect) correctCount++;

        const confidence = (output[predicted] * 100).toFixed(2);
        const statusIcon = isCorrect ? "✓ CORRECT" : "✗ MISMATCH";

        console.log(
            ` ${(i + 1).toString().padStart(2)} | ` +
            `${file.padEnd(21)} | ` +
            `${target.toString().padEnd(11)} | ` +
            `${predicted.toString().padEnd(9)} | ` +
            `${(confidence + "%").padEnd(11)} | ` +
            `${statusIcon}`
        );
    }

    console.log(`-----------------------------------------------------------------------------------------`);
    const accuracy = ((correctCount / samples.length) * 100).toFixed(1);
    console.log(`Sample Accuracy: ${correctCount}/${samples.length} (${accuracy}%)\n`);
}

async function main() {
    const args = process.argv.slice(2);
    const isTestOnly = args.includes("--test-only") || args.includes("--skip-train");

    let brain;

    if (isTestOnly) {
        console.log(`\n[--test-only flag detected: Skipping training phase]`);
        if (!fs.existsSync(MODEL_FILE)) {
            console.error(`Cannot run test-only: Saved model file not found at ${MODEL_FILE}`);
            process.exit(1);
        }
    } else {
        // ====================================================================
        // STEP 1: TRAIN ON TRAINING DATASET
        // ====================================================================
        const trainDir = path.join(DATASET_DIR, "train");
        console.log(`\nLoading training dataset from: ${trainDir}...`);

        const files = fs.readdirSync(trainDir)
            .filter(file => file.endsWith(".png"));

        console.log(`Found ${files.length} training images.`);

        const trainingData = files.map(file => ({
            filePath: path.join(trainDir, file),
            target: Number(file.split("_")[0])
        }));

        brain = new Network([784, 3, 3, 10]);

        console.log(`Network architecture initialized: ${brain.layerSizes.join(" → ")}`);
        console.log(`Beginning training across ${trainingData.length} samples...\n`);

        await brain.train(
            5,
            trainingData,
            async (sample) => {
                return await extractInfo(sample.filePath);
            },
            0.1
        );

        // ====================================================================
        // STEP 2: SAVE TRAINED MODEL ARCHITECTURE, WEIGHTS, AND BIASES
        // ====================================================================
        console.log(`\n=========================================================================================`);
        console.log(`SAVING MODEL TO DISK`);
        console.log(`=========================================================================================`);
        console.log(`Destination: ${MODEL_FILE}`);

        brain.save(MODEL_FILE);

        const stats = fs.statSync(MODEL_FILE);
        console.log(`Model successfully saved! (${(stats.size / 1024).toFixed(2)} KB)`);
    }

    // ========================================================================
    // STEP 3: LOAD MODEL FROM JSON FILE
    // ========================================================================
    console.log(`\n=========================================================================================`);
    console.log(`LOADING SAVED MODEL`);
    console.log(`=========================================================================================`);
    console.log(`Loading model file from: ${MODEL_FILE}...`);

    const loadedBrain = Network.load(MODEL_FILE);

    console.log(`Model successfully loaded into memory!`);
    console.log(`Loaded Architecture: ${loadedBrain.layerSizes.join(" → ")}`);
    console.log(`Weight Matrices: ${loadedBrain.layerSizes.length - 1} layers`);

    // ========================================================================
    // STEP 4: PREDICT ON RANDOM TEST IMAGES
    // ========================================================================
    await runTestPredictions(loadedBrain, 10);
}

main().catch(console.error);