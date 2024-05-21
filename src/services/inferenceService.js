const tf = require("@tensorflow/tfjs-node");
const InputError = require("../exceptions/InputError");

async function predictClassification(model, image) {
  try {
    const tensor = tf.node.decodeJpeg(image).resizeNearestNeighbor([224, 224]).expandDims().toFloat();
    const classes = ["Cancer", "Non-cancer"];

    const prediction = model.predict(tensor);
    const score = await prediction.data(); // Ambil hasil prediksi sebagai array

    if (score.length === 0) {
      throw new Error("Model Tidak Mengembalikan Prediksi Apapun.");
    }

    const confidenceScore = Math.max(...score) * 100;
    const classResult = tf.argMax(prediction, 1).dataSync()[0];
    let label = classes[classResult];

    if (confidenceScore < 50) {
      label = "Non-cancer";
    }

    let explanation, suggestion;

    if (label === "Cancer") {
      suggestion = "Segera konsultasi dengan dokter terdekat.";
    } else {
      suggestion = "Anda Sehat";
    }

    return { confidenceScore, label, explanation, suggestion };
  } catch (error) {
    throw new InputError(`Terjadi kesalahan input: ${error.message}`);
  }
}

module.exports = predictClassification;