# LLM Preference Classification – Kaggle Competition

This repository contains the code and pipeline for the [Kaggle LLM Classification Competition](https://www.kaggle.com/). The task is to predict human preferences between responses from two large language models (LLMs), given a prompt and their generated responses.

## 📂 Files

- `llm.ipynb`: Main notebook that performs the following:
  - Loads and preprocesses the dataset
  - Extracts embeddings from LLM responses
  - Trains a neural network classifier
  - Outputs predictions in `submission.csv`

- `submission.csv`: Output file in the required submission format.

## 🧠 Model Details

- Framework: PyTorch
- Feature Extraction: Sentence Embeddings (e.g., via SentenceTransformers)
- Model: Simple Feedforward Neural Network
- Optimizer: Adam
- Loss Function: CrossEntropyLoss

## 🧪 Evaluation

The model outputs 3-class softmax probabilities for:
- `winner_model_a`
- `winner_model_b`
- `winner_tie`

Submission is evaluated using log loss.

## 🚀 How to Use

1. Clone the repository or open the `llm.ipynb` file in [Kaggle Kernels](https://www.kaggle.com/).
2. Ensure your embeddings are loaded correctly.
3. Run all cells to train the model and generate `submission.csv`.
4. Submit `submission.csv` to the competition.

## 📌 Notes

- Ensure your notebook **outputs `submission.csv`** at the root level.
- Watch for silent crashes or hangs due to memory usage during prediction.
- You can batch the inference process to avoid timeouts or GPU OOM errors.

## 🛠️ Future Improvements

- Add model ensembling
- Use transformer fine-tuning instead of static embeddings
- Explore ranking-based loss functions

---

📫 For questions or suggestions, feel free to open an issue or reach out.

