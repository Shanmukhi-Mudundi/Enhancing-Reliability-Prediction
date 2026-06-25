# Enhancing Reliability in Predicting Amazon Reviews
[![License](https://img.shields.io/badge/License-Academic-lightgrey)]()
[![Python](https://img.shields.io/badge/Python-3.12-blue)]()
[![MLOps](https://img.shields.io/badge/Pipeline-MLOps-green)]()

> An end-to-end pipeline that scrapes Amazon reviews, scores them with an LLM, and trains a classifier to flag unreliable or fake reviews.

---

## Overview

Fake and manipulated reviews erode consumer trust on e-commerce platforms. This project builds a multi-stage system to detect unreliable Amazon reviews by combining web scraping, LLM-powered sentiment analysis, and classical machine learning classification.

The pipeline is designed to be modular — each stage is a standalone script that feeds into the next.

---

## Pipeline

```
WebScraping(1).py  →  SentimentAnalysis(2).py  →  EDA(3).py  →  ModelTraining(4).py  →  Classification(5).py
     │                        │                        │                  │                        │
  Scrapes up             Cleans data,            Exploratory         Trains & evaluates       Loads saved
  to 1,000 Amazon        runs LLM sentiment      analysis,           7 classifiers on         model, predicts
  chair reviews          (Mixtral-8x7B)          generates           TF-IDF features,         reliability of
  via Selenium           per-sentence,           visualisations,     saves best (Random       new reviews
                         labels RELIABLE /        saves              Forest) to .pkl
                         UNRELIABLE              cleaned_modeling.csv
```

---

## Scripts

### 1. `WebScraping(1).py`
Uses Selenium to automate a Chrome browser on Amazon.in. Searches for "chair", navigates to a product's review section, and scrapes up to 1,000 reviews across multiple pages.

**Collects per review:**
- Review text
- Review date
- Helpful vote count
- Verified purchase status (Yes / No)

**Output:** `amazon_reviews.csv`

---

### 2. `SentimentAnalysis(2).py`
The core enrichment stage. Takes the raw CSV, cleans it, and sends reviews in batches of 20 to the Together AI API (Mixtral-8x7B-Instruct) using prompt engineering.

**Processing steps:**
1. Clean data — drop nulls, duplicates, and reviews under 10 characters
2. Detect gibberish using spaCy POS tagging (high non-alpha ratio, too few meaningful words)
3. Split reviews into sentences using spaCy
4. Prompt the LLM to label each sentence as `POSITIVE`, `NEGATIVE`, or `NEUTRAL` with a score
5. Aggregate sentence scores to a review-level sentiment score
6. Label reviews `RELIABLE` if `|score| > 0.2` and text is non-gibberish; otherwise `UNRELIABLE`

**Output:** `sentiment_batched.csv`, `finetune_data.jsonl`

---

### 3. `EDA(3).py`
Reads `sentiment_batched.csv` and produces visualisations to understand the data before modelling.

**Generates:**
- Review length distribution (word count histogram)
- Sentiment label distribution
- Verified purchase distribution
- Correlation heatmap across numeric and encoded features

**Output:** `cleaned_modeling.csv` (cleaned, ready for training)

---

### 4. `ModelTraining(4).py`
Trains and benchmarks seven classifiers on TF-IDF vectors (top 5,000 features) of the review text.

**Models evaluated:**

| Model | Notes |
|---|---|
| Logistic Regression | Fast baseline |
| Decision Tree | Interpretable |
| Random Forest | Best performer — saved to disk |
| Support Vector Classifier | High-dimensional text classifier |
| XGBoost | Gradient boosted trees |
| Naive Bayes | Probabilistic, fast |
| KNN | Distance-based |

Outputs accuracy, precision, recall, F1, and training time per model. Saves the best model:

```
random_model.pkl
tfidf_vectorizer.pkl
```

---

### 5. `Classification(5).py`
Loads the saved Random Forest model and TF-IDF vectorizer to classify new reviews at inference time.

```python
new_reviews = [
    "This object is so good, the quality is top notch",
    "sjdfbnjkewfbgbgn",
    "very poor quality, amazon needs to change its vendors, these are all fake"
]
# → Reliable, Unreliable, Reliable
```

---

## Data Files

| File | Description |
|---|---|
| `amazon_reviews.csv` | Raw scraped reviews |
| `sentiment_batched.csv` | Reviews with sentiment scores and reliability labels |
| `cleaned_modeling.csv` | Final cleaned dataset used for model training |

---

## Setup

### Requirements

```bash
pip install selenium spacy pandas scikit-learn xgboost joblib matplotlib seaborn requests
python -m spacy download en_core_web_sm
```

You will also need:
- **Google Chrome** and the matching **ChromeDriver** on your PATH
- A **Together AI API key** — set it in `SentimentAnalysis(2).py`:
  ```python
  TOGETHER_API_KEY = "your_key_here"
  ```
  Get one at [together.ai](https://www.together.ai)

---

## Running the Pipeline

Run scripts in order:

```bash
python "WebScraping(1).py"
python "SentimentAnalysis(2).py"
python "EDA(3).py"
python "ModelTraining(4).py"
python "Classification(5).py"
```

> **Note:** `WebScraping(1).py` may prompt you to log in to Amazon manually. Follow the on-screen instruction and press Enter to continue once logged in.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Scraping | Selenium, ChromeDriver |
| NLP | spaCy (`en_core_web_sm`) |
| LLM | Together AI — Mixtral-8x7B-Instruct-v0.1 |
| Feature extraction | scikit-learn `TfidfVectorizer` |
| Classification | scikit-learn, XGBoost |
| Visualisation | matplotlib, seaborn |
| Model persistence | joblib |

---

## Reliability Logic

A review is labelled **UNRELIABLE** if any of the following are true:

- It is gibberish (high non-alphabetic character ratio, fewer than 2 meaningful words)
- It has fewer than 3 words
- Its LLM sentiment score falls in the range `(-0.2, +0.2)` — too neutral to carry genuine opinion

Otherwise it is labelled **RELIABLE**.

---

## Project Structure

```
Enhancing-Reliability-Prediction/
├── WebScraping(1).py          # Stage 1: Scrape reviews from Amazon
├── SentimentAnalysis(2).py    # Stage 2: Clean + LLM sentiment labelling
├── EDA(3).py                  # Stage 3: Exploratory data analysis
├── ModelTraining(4).py        # Stage 4: Train and evaluate classifiers
├── Classification(5).py       # Stage 5: Inference on new reviews
├── amazon_reviews.csv         # Raw scraped data
├── sentiment_batched.csv      # Labelled data
├── cleaned_modeling.csv       # Final training data
└── README.md
```

---

## Known Limitations

- The reliability threshold (`±0.2`) is a fixed heuristic, not learned from data.
- The API key is stored in plain text in the script — use environment variables in production:
  ```python
  import os
  TOGETHER_API_KEY = os.environ["TOGETHER_API_KEY"]
  ```
- No `requirements.txt` is currently included — see the Setup section above for dependencies.
- Amazon's page structure may change over time, requiring XPath updates in the scraper.
