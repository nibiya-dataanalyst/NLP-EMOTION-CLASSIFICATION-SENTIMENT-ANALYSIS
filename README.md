🎭 NLP Emotion Classification & Sentiment Analysis

A Machine Learning pipeline for multi-class emotion classification on text data using **TF-IDF Vectorization**, **Multinomial Naive Bayes**, and **Support Vector Machines (LinearSVC)**.

---

## 📌 Project Overview

This project processes raw textual feedback/comments and classifies them into specific emotion categories (`anger`, `fear`, and `joy`). It demonstrates an end-to-end Natural Language Processing (NLP) workflow—from text cleaning and preprocessing to feature extraction, model training, evaluation, and comparative visualization.

---

## 📊 Dataset

The dataset (`nlp_dataset.csv`) contains unstructured text comments labeled with target emotion classes.
* **Text Column:** `Comment` / Raw Text
* **Target Column:** `Emotion` (`anger`, `fear`, `joy`)

---

## ⚙️ Methodology & Workflow

Raw Text Data
│
▼
[ Text Preprocessing ] ──► Lowercasing, Regex URL/Punctuation Removal, Stopword Filtering
│
▼
[ Feature Extraction ] ──► TF-IDF Vectorization (Unigrams & Bigrams, Max Features: 5000)
│
▼
[ Model Training ]    ──► Multinomial Naive Bayes vs. LinearSVC
│
▼
[ Evaluation ]         ──► Accuracy, Weighted F1-Score, Classification Report, Confusion Matrices


1. **Data Preprocessing:**
   * Text normalization (lowercasing).
   * Noise removal via regular expressions (URLs, digits, punctuation, special characters).
   * Tokenization and removal of English NLTK stopwords.
2. **Feature Extraction:**
   * Transformed clean text into TF-IDF feature matrices (`ngram_range=(1, 2)`).
   * Evaluated on an 80/20 stratified train-test split.
3. **Model Training & Comparison:**
   * **Multinomial Naive Bayes (MultinomialNB):** Probabilistic baseline model.
   * **Support Vector Machine (LinearSVC):** High-dimensional boundary optimization model.

---

## 📈 Performance & Results

| Model | Accuracy | Weighted F1-Score |
| :--- | :---: | :---: |
| **Multinomial Naive Bayes** | ~91.16% | ~91.16% |
| **Support Vector Machine (LinearSVC)** | **~94.61%** | **~94.61%** |

* **Key Takeaway:** The LinearSVC model outperforms Naive Bayes due to its ability to handle sparse, high-dimensional TF-IDF vectors without relying on strict feature independence assumptions.

---

## 📂 Repository Structure

```text
.
├── nlp_classification.ipynb   # Main Jupyter Notebook containing code & evaluation
├── nlp_dataset.csv            # Dataset file
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
