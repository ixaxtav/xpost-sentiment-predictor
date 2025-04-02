const tf = require("@tensorflow/tfjs-node");

// This function calculates the cosine similarity between two tensors.
function cosineSimilarity(a, b) {
  const aFlat = a.flatten();
  const bFlat = b.flatten();

  const dotProduct = tf.dot(aFlat, bFlat).dataSync()[0];
  const normA = tf.norm(aFlat).dataSync()[0];
  const normB = tf.norm(bFlat).dataSync()[0];

  return dotProduct / (normA * normB);
}

module.exports = { cosineSimilarity };
