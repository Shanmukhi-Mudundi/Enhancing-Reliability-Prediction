"""Exploratory Data Analysis module."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from typing import Optional

from ..utils.logger import setup_logger
from ..utils.data_cleaning import clean_data, validate_dataframe

logger = setup_logger(__name__)


def perform_eda(csv_file_path: str, output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Perform exploratory data analysis.
    
    Args:
        csv_file_path: Path to input CSV
        output_file: Path to save processed data
        
    Returns:
        Cleaned DataFrame
    """
    # Load and clean data
    df = clean_data(csv_file_path)
    
    # Validate data
    if not validate_dataframe(df):
        raise ValueError("Data validation failed")
    
    # Basic info
    logger.info("\n" + "="*50)
    logger.info("DATASET INFORMATION")
    logger.info("="*50)
    logger.info(f"Shape: {df.shape}")
    
    # Save processed data
    if output_file:
        df.to_csv(output_file, index=False)
        logger.info(f"Processed data saved to {output_file}")
    
    logger.info("\n✅ EDA complete")
    return df
