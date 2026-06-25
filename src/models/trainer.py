"""Model training module."""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelTrainer:
    """Train and evaluate classification models."""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize model trainer.
        
        Args:
            test_size: Test set size ratio
            random_state: Random state for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.vectorizer = None
        self.results = None
        logger.info(f"ModelTrainer initialized (test_size={test_size})")
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        text_column: str = "Review Text",
        label_column: str = "Reliability",
        max_features: int = 5000
    ) -> Tuple:
        """
        Prepare data for training.
        
        Args:
            df: Input DataFrame
            text_column: Column with text data
            label_column: Column with labels
            max_features: Maximum features for TF-IDF
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Preparing data with max_features={max_features}")
        
        label_mapping = {'UNRELIABLE': 0, 'RELIABLE': 1}
        df_filtered = df[df[label_column].isin(label_mapping)]
        logger.info(f"Filtered to {len(df_filtered)} samples")
        
        y = df_filtered[label_column].map(label_mapping)
        X_text = df_filtered[text_column]
        
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
        X = self.vectorizer.fit_transform(X_text)
        logger.info(f"Vectorization complete. Shape: {X.shape}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )
        
        logger.info(f"Train set size: {X_train.shape}, Test set size: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, y_train) -> Dict:
        """
        Train multiple models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of model results
        """
        models_config = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=self.random_state),
            "Decision Tree": DecisionTreeClassifier(random_state=self.random_state),
            "Random Forest": RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            "Support Vector Classifier": SVC(random_state=self.random_state),
            "XGBoost Classifier": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
            "Naive Bayes": MultinomialNB(),
            "KNN Classifier": KNeighborsClassifier(n_jobs=-1)
        }
        
        results = []
        logger.info(f"Training {len(models_config)} models...")
        
        for name, model in models_config.items():
            try:
                logger.info(f"Training {name}...")
                start = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - start
                
                results.append({
                    "Model": name,
                    "Model Object": model,
                    "Train Time (s)": round(train_time, 3)
                })
                logger.info(f"✓ {name} trained in {train_time:.2f}s")
            except Exception as e:
                logger.error(f"✗ {name} failed: {str(e)}")
        
        return results
    
    def evaluate_models(self, models_results: List[Dict], X_test, y_test) -> pd.DataFrame:
        """
        Evaluate models on test set.
        
        Args:
            models_results: List of model results
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with evaluation metrics
        """
        logger.info("Evaluating models on test set...")
        evaluation_results = []
        
        for result in models_results:
            model = result["Model Object"]
            name = result["Model"]
            
            try:
                y_pred = model.predict(X_test)
                metrics = {
                    "Model": name,
                    "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                    "Precision": round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
                    "Recall": round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
                    "F1 Score": round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4),
                    "Train Time (s)": result["Train Time (s)"]
                }
                evaluation_results.append(metrics)
                logger.info(f"{name}: F1={metrics['F1 Score']}")
            except Exception as e:
                logger.error(f"Error evaluating {name}: {str(e)}")
        
        results_df = pd.DataFrame(evaluation_results).sort_values(by="F1 Score", ascending=False)
        self.results = results_df
        logger.info(f"\nTop Models:\n{results_df.head(3).to_string(index=False)}")
        return results_df
    
    def save_best_model(self, models_results: List[Dict], model_path: str = "models/best_model.pkl") -> None:
        """
        Save the best model and vectorizer.
        
        Args:
            models_results: List of model results
            model_path: Path to save model
        """
        if self.results is None or self.results.empty:
            logger.error("No results available")
            return
        
        best_model_name = self.results.iloc[0]['Model']
        best_model = next(r["Model Object"] for r in models_results if r["Model"] == best_model_name)
        
        joblib.dump(self.vectorizer, "models/tfidf_vectorizer.pkl")
        joblib.dump(best_model, model_path)
        logger.info(f"Best model ({best_model_name}) saved to {model_path}")
