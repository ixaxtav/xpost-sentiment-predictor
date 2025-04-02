const use = require("@tensorflow-models/universal-sentence-encoder");
const { cosineSimilarity } = require("../utils/similarity");

const TEXTS = [
  "I love this so much!",
  "This is terrible and awful.",
  "It's okay, I guess.",
];

async function analyzeSentiment(texts) {
  const model = await use.load();

  const reference = {
    positive: "I am feeling very happy and excited.",
    negative: "I am so angry and disappointed.",
  };

  const allSentences = [...texts, reference.positive, reference.negative];
  const embeddings = await model.embed(allSentences);

  const results = [];

  for (let i = 0; i < texts.length; i++) {
    const inputEmbedding = embeddings.slice([i, 0], [1, -1]);
    const positiveEmbedding = embeddings.slice([texts.length, 0], [1, -1]);
    const negativeEmbedding = embeddings.slice([texts.length + 1, 0], [1, -1]);

    const posScore = cosineSimilarity(inputEmbedding, positiveEmbedding);
    const negScore = cosineSimilarity(inputEmbedding, negativeEmbedding);

    let sentiment = "Neutral";
    if (posScore - negScore > 0.1) sentiment = "Positive";
    else if (negScore - posScore > 0.1) sentiment = "Negative";

    results.push({
      text: texts[i],
      sentiment,
      scores: { pos: posScore.toFixed(3), neg: negScore.toFixed(3) },
    });
  }

  return results;
}

// Run the analysis
analyzeSentiment(TEXTS).then((results) => {
  results.forEach((r) => {
    console.log(
      `"${r.text}" ➜ ${r.sentiment} (pos: ${r.scores.pos}, neg: ${r.scores.neg})`
    );
  });
});
