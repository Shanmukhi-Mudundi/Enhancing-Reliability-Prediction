# Enhancing Reliability in Amazon Review Prediction

## Overview

This project develops an advanced system to predict the reliability of Amazon reviews by combining:
- **Web Scraping**: Collect reviews from Amazon using Selenium
- **Sentiment Analysis**: Analyze review sentiment using LLM APIs
- **Data Analysis**: Perform exploratory data analysis
- **Machine Learning**: Train models to classify review reliability

## Project Structure

```
src/
├── scraping/              # Web scraping module
├── preprocessing/         # Sentiment analysis
├── analysis/              # EDA module
├── models/                # ML models and training
└── utils/                 # Utility functions

data/
├── raw/                   # Raw scraped data
└── processed/             # Processed data

models/                    # Trained models
tests/                     # Unit tests
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Shanmukhi-Mudundi/Enhancing-Reliability-Prediction.git
cd Enhancing-Reliability-Prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

## Usage

### Web Scraping
```python
from src.scraping import scrape_amazon_reviews

scrape_amazon_reviews(
    output_file="data/raw/amazon_reviews.csv",
    target_reviews=1000
)
```

### Sentiment Analysis
```python
from src.preprocessing.sentiment_analyzer import SentimentAnalyzer
from src.utils.data_cleaning import clean_data

df = clean_data("data/raw/amazon_reviews.csv")
analyzer = SentimentAnalyzer(batch_size=20)
df_sentiment = analyzer.analyze_batches(df)
df_sentiment.to_csv("data/processed/sentiment_batched.csv", index=False)
```

### Model Training
```python
from src.models.trainer import ModelTrainer

trainer = ModelTrainer(test_size=0.2)
X_train, X_test, y_train, y_test = trainer.prepare_data(df)
models_results = trainer.train_models(X_train, y_train)
results = trainer.evaluate_models(models_results, X_test, y_test)
trainer.save_best_model(models_results)
```

### Make Predictions
```python
from src.models.classifier import ReliabilityClassifier

classifier = ReliabilityClassifier(
    model_path="models/best_model.pkl",
    vectorizer_path="models/tfidf_vectorizer.pkl"
)

reviews = ["This product is amazing!", "Poor quality"]
predictions = classifier.predict(reviews)
```

## Testing

Run unit tests:
```bash
pytest tests/ -v
```

## Security

- **Never commit `.env` files** - they contain sensitive API keys
- Use environment variables for all API keys
- The `.gitignore` file protects sensitive data

## License

MIT License

## Author

Shanmukhi Mudundi
