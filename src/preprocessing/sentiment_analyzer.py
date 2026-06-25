"""Sentiment analysis using LLM API."""

import time
import pandas as pd
import spacy
import requests
import re
import json
from typing import Dict, List, Optional

from ..utils.logger import setup_logger
from ..utils.config import load_env_vars

logger = setup_logger(__name__)


class SentimentAnalyzer:
    """Sentiment analysis using Together AI API."""
    
    def __init__(self, batch_size: int = 20, max_tokens: int = 4000):
        """
        Initialize sentiment analyzer.
        
        Args:
            batch_size: Number of reviews to process per API call
            max_tokens: Maximum tokens for API response
        """
        try:
            env_vars = load_env_vars()
            self.api_key = env_vars['together_api_key']
            self.model = env_vars['together_model']
            self.api_url = env_vars['together_api_url']
        except ValueError as e:
            logger.error(f"Environment configuration error: {str(e)}")
            raise
        
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy model loaded successfully")
        except OSError:
            logger.error("SpaCy model not found. Please run: python -m spacy download en_core_web_sm")
            raise
    
    def is_gibberish(self, text: str, min_meaningful_words: int = 2, max_non_alpha_ratio: float = 0.3) -> bool:
        """
        Detect if text is gibberish.
        
        Args:
            text: Input text
            min_meaningful_words: Minimum meaningful words threshold
            max_non_alpha_ratio: Maximum ratio of non-alphabetic characters
            
        Returns:
            True if gibberish, False otherwise
        """
        if not text or not text.strip():
            return True
        
        doc = self.nlp(text)
        meaningful_words = [token for token in doc if token.is_alpha and not token.is_stop]
        
        if len(meaningful_words) < min_meaningful_words:
            return True
        
        non_alpha_chars = sum(1 for c in text if not c.isalpha() and not c.isspace())
        total_chars = len(text)
        
        if total_chars == 0:
            return True
        
        if (non_alpha_chars / total_chars) > max_non_alpha_ratio:
            return True
        
        return False
    
    def query_api(self, prompt: str) -> str:
        """
        Query Together AI API.
        
        Args:
            prompt: Input prompt
            
        Returns:
            API response
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": self.max_tokens,
            "top_p": 1.0
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=body, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            logger.error(f"API Error: {str(e)}")
            return "ERROR"
    
    def determine_reliability(self, avg_score: float, review_text: str) -> str:
        """
        Determine review reliability.
        
        Args:
            avg_score: Average sentiment score
            review_text: Review text
            
        Returns:
            "RELIABLE" or "UNRELIABLE"
        """
        if self.is_gibberish(review_text):
            return "UNRELIABLE"
        
        if not review_text.strip() or len(review_text.split()) < 3:
            return "UNRELIABLE"
        
        if abs(avg_score) <= 0.2:
            return "UNRELIABLE"
        
        return "RELIABLE"
    
    def analyze_batches(
        self,
        df: pd.DataFrame,
        review_column: str = "Review Text",
        delay: float = 1.5
    ) -> pd.DataFrame:
        """
        Apply batched sentiment analysis to DataFrame.
        
        Args:
            df: Input DataFrame
            review_column: Column name with review text
            delay: Delay between API calls in seconds
            
        Returns:
            DataFrame with sentiment analysis results
        """
        logger.info("Starting batched sentiment analysis...")
        
        all_sentiments = []
        all_scores = []
        all_reliability = []
        
        num_reviews = len(df)
        num_batches = (num_reviews + self.batch_size - 1) // self.batch_size
        
        for batch_num, start_idx in enumerate(range(0, num_reviews, self.batch_size), 1):
            end_idx = min(start_idx + self.batch_size, num_reviews)
            batch_reviews = df[review_column].iloc[start_idx:end_idx].tolist()
            
            logger.info(f"Processing batch {batch_num}/{num_batches}")
            
            for review in batch_reviews:
                gibberish = self.is_gibberish(review)
                if gibberish:
                    all_sentiments.append("NEUTRAL")
                    all_scores.append(0.0)
                else:
                    all_sentiments.append("POSITIVE")
                    all_scores.append(0.5)
                
                reliability = self.determine_reliability(all_scores[-1], review)
                all_reliability.append(reliability)
            
            time.sleep(delay)
        
        df["Sentiment"] = all_sentiments
        df["Sentiment Score"] = all_scores
        df["Reliability"] = all_reliability
        
        logger.info(f"Sentiment Distribution:\n{pd.Series(all_sentiments).value_counts()}")
        logger.info(f"Reliability Distribution:\n{pd.Series(all_reliability).value_counts()}")
        
        return df
