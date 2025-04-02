const tf = require("@tensorflow/tfjs-node");
const use = require("@tensorflow-models/universal-sentence-encoder");
const { cosineSimilarity } = require("../utils/similarity");

const TEXTS = [
  "I love this so much!",
  "This is terrible and awful.",
  "It's okay, I guess.",
];

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

async function analyzeSentiment(texts) {
  const model = await use.load();

  const allReference = [...reference.positive, ...reference.negative];
  const allSentences = [...texts, ...allReference];
  const embeddings = await model.embed(allSentences);

  const inputEmbeddings = embeddings.slice([0, 0], [texts.length, -1]);
  const positiveEmbeddings = embeddings.slice(
    [texts.length, 0],
    [reference.positive.length, -1]
  );
  const negativeEmbeddings = embeddings.slice(
    [texts.length + reference.positive.length, 0],
    [reference.negative.length, -1]
  );

  const posAvg = averageEmbedding(positiveEmbeddings);
  const negAvg = averageEmbedding(negativeEmbeddings);

  const results = [];

  for (let i = 0; i < texts.length; i++) {
    const inputEmbedding = inputEmbeddings.slice([i, 0], [1, -1]);

    const posScore = cosineSimilarity(inputEmbedding, posAvg);
    const negScore = cosineSimilarity(inputEmbedding, negAvg);

    let sentiment = "Neutral";
    if (posScore - negScore > 0.1) sentiment = "Positive";
    else if (negScore - posScore > 0.1) sentiment = "Negative";

    results.push({
      text: texts[i],
      sentiment,
      scores: {
        pos: posScore.toFixed(3),
        neg: negScore.toFixed(3),
      },
    });
  }

  return results;
}

analyzeSentiment(TEXTS).then((results) => {
  results.forEach((r) => {
    console.log(
      `"${r.text}" ➜ ${r.sentiment} (pos: ${r.scores.pos}, neg: ${r.scores.neg})`
    );
  });
});
