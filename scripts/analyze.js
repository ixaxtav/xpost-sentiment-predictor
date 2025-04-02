const tf = require("@tensorflow/tfjs-node");
const use = require("@tensorflow-models/universal-sentence-encoder");
const { cosineSimilarity } = require("../utils/similarity");

const userInput = process.argv.slice(2).join(" ");

if (!userInput) {
  console.error("❗ Please enter a sentence to analyze:");
  console.error('Example: node scripts/analyze.js "I love this app"');
  process.exit(1);
}

const reference = {
  positive: [
    "I love this so much.",
    "This is amazing and wonderful.",
    "I'm really happy with it.",
    "I absolutely enjoyed it.",
  ],
  negative: [
    "I hate this so much.",
    "This is terrible and awful.",
    "I'm really upset with it.",
    "I absolutely hated it.",
  ],
};

function averageEmbedding(tensorGroup) {
  return tf.mean(tensorGroup, 0).reshape([1, -1]);
}

async function analyzeSentiment(text) {
  const model = await use.load();

  const allReference = [...reference.positive, ...reference.negative];
  const allSentences = [text, ...allReference];
  const embeddings = await model.embed(allSentences);

  const inputEmbedding = embeddings.slice([0, 0], [1, -1]);
  const positiveEmbeddings = embeddings.slice(
    [1, 0],
    [reference.positive.length, -1]
  );
  const negativeEmbeddings = embeddings.slice(
    [1 + reference.positive.length, 0],
    [reference.negative.length, -1]
  );

  const posAvg = averageEmbedding(positiveEmbeddings);
  const negAvg = averageEmbedding(negativeEmbeddings);

  const posScore = cosineSimilarity(inputEmbedding, posAvg);
  const negScore = cosineSimilarity(inputEmbedding, negAvg);

  let sentiment = "Neutral";
  if (posScore - negScore > 0.1) sentiment = "Positive";
  else if (negScore - posScore > 0.1) sentiment = "Negative";

  console.log(`\n"${text}" ➜ ${sentiment}`);
  console.log(`Positive Score: ${posScore.toFixed(3)}`);
  console.log(`Negative Score: ${negScore.toFixed(3)}\n`);
}

analyzeSentiment(userInput);
