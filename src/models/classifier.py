"""Reliability classification using trained models."""

import joblib
import pandas as pd
from typing import List, Dict

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class ReliabilityClassifier:
    """Classify review reliability using trained model."""
    
    def __init__(self, model_path: str, vectorizer_path: str):
        """
        Initialize classifier with trained model and vectorizer.
        
        Args:
            model_path: Path to trained model
            vectorizer_path: Path to TF-IDF vectorizer
        """
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            logger.info(f"Model loaded from {model_path}")
            logger.info(f"Vectorizer loaded from {vectorizer_path}")
        except FileNotFoundError as e:
            logger.error(f"Could not load model or vectorizer: {str(e)}")
            raise
    
    def predict(self, reviews: List[str]) -> List[Dict[str, str]]:
        """
        Predict reliability for reviews.
        
        Args:
            reviews: List of review texts
            
        Returns:
            List of predictions with review and reliability
        """
        try:
            X = self.vectorizer.transform(reviews)
            predictions = self.model.predict(X)
            label_map = {0: "Unreliable", 1: "Reliable"}
            
            results = []
            for review, prediction in zip(reviews, predictions):
                results.append({
                    "Review": review,
                    "Predicted Reliability": label_map[prediction]
                })
            
            logger.info(f"Predicted reliability for {len(reviews)} reviews")
            return results
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
