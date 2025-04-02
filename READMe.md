# X-Post Sentimental Analysis Predictor with TensorFlow.js

A lightweight Node.js script that analyzes the sentiment of a user-provided sentence using the Universal Sentence Encoder (USE) from TensorFlow.js. It compares the input to predefined positive, negative, and neutral reference sentences using cosine similarity.

## Features

- Classifies sentiment as "Positive," "Negative," or "Neutral."
- Uses pre-trained embeddings from the Universal Sentence Encoder.
- Simple command-line interface.

## Prerequisites

- [Node.js](https://nodejs.org/) (v14 or higher recommended).
- npm (comes with Node.js).

## Installation

1. Clone or download this repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install dependencies:
   ```bash
   npm install @tensorflow/tfjs-node @tensorflow-models/universal-sentence-encoder
   ```

## Usage

Run the script from the command line by providing a sentence:

```bash
node scripts/analyze.js "I love this app"
```

Example output:

```
"I love this app" ➜ Positive
Positive Score: 0.823
Negative Score: 0.412
Neutral Score: 0.567
```

If no input is provided, an error message will guide you:

```bash
node scripts/analyze.js
❗ Please enter a sentence to analyze:
Example: node scripts/analyze.js "I love this app"
```

## How It Works

1. The script loads the Universal Sentence Encoder model.
2. It embeds the user input and reference sentences into vector representations.
3. It calculates average embeddings for positive, negative, and neutral reference groups.
4. Cosine similarity is used to compare the input embedding to each group.
5. Sentiment is determined based on score differences with a threshold of 0.1.

## Customization

- Edit the `reference` object in `analyze.js` to adjust the benchmark sentences.
- Modify the threshold in the sentiment logic for stricter or looser classification.

## Dependencies

- `@tensorflow/tfjs-node`: TensorFlow.js for Node.js.
- `@tensorflow-models/universal-sentence-encoder`: Pre-trained sentence embedding model.

## License

This project is unlicensed and free to use or modify as you see fit.
