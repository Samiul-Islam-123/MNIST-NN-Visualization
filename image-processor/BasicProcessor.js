import sharp from "sharp";

export async function extractInfo(filepath) {
    const { data, info } = await sharp(filepath)
        .grayscale()
        .resize(28, 28)
        .raw()
        .toBuffer({ resolveWithObject: true });

    return Array.from(data, pixel => pixel / 255);
}