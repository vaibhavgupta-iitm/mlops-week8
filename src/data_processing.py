"""
Data processing module for IRIS pipeline.
Handles data loading, preprocessing, and train-test splitting.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data processing operations for the IRIS dataset."""
    
    def __init__(self, test_size: float = 0.4, random_state: int = 42):
        """
        Initialize DataProcessor.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        self.target_column = 'species'
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Loaded DataFrame
        """
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Loaded data with shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate the loaded data.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        # Check if required columns exist
        required_columns = self.feature_columns + [self.target_column]
        missing_columns = set(required_columns) - set(data.columns)
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for null values
        if data.isnull().any().any():
            logger.error("Data contains null values")
            return False
        
        # Check data types
        for col in self.feature_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                logger.error(f"Column {col} is not numeric")
                return False
        
        # Check target column has expected values
        expected_species = ['setosa', 'versicolor', 'virginica']
        if not set(data[self.target_column].unique()).issubset(set(expected_species)):
            logger.error(f"Unexpected species values: {data[self.target_column].unique()}")
            return False
        
        logger.info("Data validation passed")
        return True
    
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            train, test = train_test_split(
                data, 
                test_size=self.test_size, 
                stratify=data[self.target_column], 
                random_state=self.random_state
            )
            
            X_train = train[self.feature_columns]
            y_train = train[self.target_column]
            X_test = test[self.feature_columns]
            y_test = test[self.target_column]
            
            logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise
    
    def augment_data(self, data: pd.DataFrame, n_extra_rows: int = 10) -> pd.DataFrame:
        """
        Augment data by adding extra rows through sampling with replacement.
        
        Args:
            data: Original DataFrame
            n_extra_rows: Number of extra rows to add
            
        Returns:
            Augmented DataFrame
        """
        try:
            extra_rows = data.sample(n=n_extra_rows, replace=True, random_state=self.random_state).reset_index(drop=True)
            augmented_data = pd.concat([data, extra_rows], ignore_index=True)
            logger.info(f"Data augmented - Original: {data.shape[0]}, Augmented: {augmented_data.shape[0]}")
            return augmented_data
        except Exception as e:
            logger.error(f"Error augmenting data: {e}")
            raise
