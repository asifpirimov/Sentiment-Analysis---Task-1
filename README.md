# Sentiment Analysis – Task 1

This repository contains my solution for **Task 1 (Sentiment Analysis)** prepared for the **Artificial Intelligence Internship Program**.

The objective of this task is to build a machine learning model that can analyze a user-written sentence and determine the **emotion** expressed in the text.

---

## 📌 Task Overview

The task requires:
- Researching sentiment analysis
- Performing exploratory data analysis (EDA)
- Applying text preprocessing techniques
- Training a sentiment (emotion) classification model
- Evaluating and interpreting results
- Providing a working Python implementation (no ready-made templates)

This project follows all the above requirements and is implemented from scratch using Python.

---

## 🎯 Emotion Classes

The model predicts one of the following **six emotions**:

- **anger**
- **fear**
- **joy**
- **love**
- **sadness**
- **surprise**

---

## 🧠 1. Sentiment Analysis – Research Summary

Sentiment Analysis is a Natural Language Processing (NLP) technique used to determine the emotional tone of a text.  
In this project, sentiment analysis is formulated as a **multi-class classification problem**, where each sentence belongs to one emotion class.

A traditional machine learning approach was chosen because it:
- is interpretable,
- trains quickly,
- and clearly demonstrates understanding of the full ML pipeline.

---

## 📂 2. Dataset

The dataset consists of three CSV files:

- `training.csv` – 16,000 samples  
- `validation.csv` – 2,000 samples  
- `test.csv`

Each dataset contains:
- `text` – a user-written sentence
- `label` – an encoded emotion class

The dataset is **imbalanced**, meaning some emotions appear more frequently than others.

---

## 📊 3. Exploratory Data Analysis (EDA)

### Training Set
- Shape: **(16000, 2)**
- Average text length: **~96 characters**
- Maximum text length: **300 characters**

### Validation Set
- Shape: **(2000, 2)**
- Average text length: **~95 characters**
- Maximum text length: **295 characters**

**Observation:**  
Classes such as *joy* and *sadness* dominate the dataset, while *surprise* and *love* have fewer samples.

---

## 🧹 4. Text Preprocessing

The following preprocessing steps were applied:

- Conversion to **lowercase**
- Removal of **numbers**
- Removal of **punctuation**
- Whitespace normalization

These steps reduce noise and improve model generalization.

---

## 🔢 5. Feature Engineering

Text data is converted into numerical vectors using:

- **TF-IDF (Term Frequency – Inverse Document Frequency)**
- Unigrams and bigrams (`ngram_range = (1, 2)`)
- Maximum features: **8000**

TF-IDF helps the model focus on informative words while reducing the impact of very common terms.

---

## 🤖 6. Model

The classification model used is:

- **Logistic Regression**

**Why Logistic Regression?**
- Works well with sparse TF-IDF features
- Fast and efficient
- Provides a strong and interpretable baseline for text classification

---

## 📈 7. Evaluation Results (Validation Set)

- **Accuracy:** `0.851`

### Key Observations:
- **Joy** and **Sadness** achieve high recall due to clearer linguistic patterns.
- **Love** has high precision but lower recall, likely due to semantic overlap with joyful expressions.
- **Fear** and **Surprise** are harder to classify due to fewer samples.
- Overall performance is strong for a baseline sentiment analysis model.

---

## 🧪 8. Example Prediction

```bash
python -m src.predict --text "I feel very happy today"
