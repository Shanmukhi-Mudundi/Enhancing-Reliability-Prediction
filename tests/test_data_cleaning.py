"""Tests for data cleaning module."""

import pytest
import pandas as pd
from src.utils.data_cleaning import validate_dataframe


class TestDataCleaning:
    """Test cases for data cleaning."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            "Review Text": ["This is a good product", "Bad quality"],
            "Verified Purchase": ["Yes", "No"],
            "Helpful Votes": [5, 0]
        })
    
    def test_validate_dataframe_with_valid_data(self, sample_df):
        """Test validation with valid DataFrame."""
        assert validate_dataframe(sample_df) == True
    
    def test_validate_dataframe_with_missing_columns(self, sample_df):
        """Test validation with missing required columns."""
        assert validate_dataframe(sample_df, required_columns=["Missing Column"]) == False
    
    def test_validate_dataframe_with_empty_data(self):
        """Test validation with empty DataFrame."""
        assert validate_dataframe(pd.DataFrame()) == False
