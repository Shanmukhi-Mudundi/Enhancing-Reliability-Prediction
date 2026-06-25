"""Data cleaning utilities."""

import pandas as pd
from typing import List, Optional
from .logger import setup_logger

logger = setup_logger(__name__)


def clean_data(
    csv_file_path: str,
    na_values: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load and clean CSV data.
    
    Args:
        csv_file_path: Path to CSV file
        na_values: List of values to treat as NA
        
    Returns:
        Cleaned DataFrame
    """
    if na_values is None:
        na_values = ["None", "none", "NA", "N/A", "n/a", "null", "NULL", "-", ""]
    
    try:
        logger.info(f"Reading file: {csv_file_path}")
        df = pd.read_csv(csv_file_path, na_values=na_values)
        logger.info(f"File loaded successfully. Shape: {df.shape}")
        
        # Clean Verified Purchase column
        if "Verified Purchase" in df.columns:
            df["Verified Purchase"] = df["Verified Purchase"].apply(
                lambda x: "Yes" if str(x).strip().lower() == "yes" else "No"
            )
            logger.info("Cleaned 'Verified Purchase' column")
        
        # Drop missing values
        initial_shape = df.shape
        df.dropna(inplace=True)
        logger.info(f"Dropped missing values. Shape: {initial_shape} -> {df.shape}")
        
        # Filter short reviews
        if "Review Text" in df.columns:
            original_len = len(df)
            df = df[df["Review Text"].str.len() > 10]
            logger.info(f"Filtered short reviews: {original_len - len(df)} removed")
        
        # Drop duplicates
        original_len = len(df)
        df.drop_duplicates(inplace=True)
        logger.info(f"Dropped duplicates: {original_len - len(df)} removed")
        
        return df
        
    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        raise


def validate_dataframe(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> bool:
    """
    Validate DataFrame structure.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if valid, False otherwise
    """
    if df is None or df.empty:
        logger.error("DataFrame is None or empty")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
    
    logger.info(f"DataFrame validation passed. Shape: {df.shape}")
    return True
