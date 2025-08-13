"""
UFC Fighter Segmentation - Data Pre-Processing Module
This module handles basic validation and data quality fixes before main processing:
1. Calculate total fight time and drop misleading columns
2. Calculate total strikes landed/attempted and drop erroneous columns
3. Basic data validation and quality checks
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreProcessor:
    """
    Data Pre-Processor to fix data quality issues before main processing step.
    
    Handles:
    - Fight time calculation corrections
    - Strike statistics recalculation 
    - Method standardization
    - Winner validation and filtering
    - Column cleanup and data type standardization
    - Basic validation
    """
    
    def __init__(self, fight_details: pd.DataFrame, event_details: pd.DataFrame = None):
        """
        Initialize with fight details DataFrame and optional event details.
        
        Parameters:
        fight_details (pd.DataFrame): Raw fight details data
        event_details (pd.DataFrame): Event details with winner information
        """
        self.fight_details = fight_details.copy()  # Work on a copy to avoid mutating original
        self.event_details = event_details.copy() if event_details is not None else None
        
        # Method categories for standardization
        self.method_categories = ['ko_tko', 'submission', 'decision', 'other']
        
        logger.info(f"PreProcessor initialized with {len(self.fight_details)} fights")

    def _calculate_fight_time(self) -> None:
        """Calculate correct total fight time in seconds."""
        logger.info("Calculating fight time corrections...")
        
        # Calculate correct total fight time
        self.fight_details['fight_time_sec'] = self.fight_details.apply(
            self._calculate_fight_time_helper, axis=1
        )
        
        # Drop the problematic column if it exists
        if 'match_time_sec' in self.fight_details.columns:
            self.fight_details.drop(columns=['match_time_sec'], inplace=True)
            logger.info("Dropped 'match_time_sec' column")


    @staticmethod
    def _calculate_fight_time_helper(row) -> Optional[int]:
        """
        Helper function to calculate total fight time in seconds based on finish round and match time.
        
        Parameters:
        row: DataFrame row containing 'finish_round' and 'match_time_sec'
        
        Returns:
        Optional[int]: Total fight time in seconds, or NaN for invalid values
        """
        if pd.isna(row['finish_round']) or pd.isna(row['match_time_sec']):
            return np.nan
            
        finish_round = int(row['finish_round'])
        match_time_sec = row['match_time_sec']
        
        if finish_round == 1:
            return match_time_sec
        elif finish_round == 2:
            return 300 + match_time_sec  # 5 minutes = 300 seconds
        elif finish_round == 3:
            return 600 + match_time_sec  # 10 minutes = 600 seconds
        elif finish_round == 4:
            return 900 + match_time_sec  # 15 minutes = 900 seconds
        elif finish_round == 5:
            return 1200 + match_time_sec  # 20 minutes = 1200 seconds
        else:
            logger.warning(f"Unexpected finish_round value: {finish_round}")
            return np.nan

    def _standardize_methods(self) -> None:
        """Standardize fight method categories."""
        logger.info("Standardizing fight methods...")
        
        self.fight_details['method_category'] = self.fight_details['method'].apply(self._categorize_method)
        logger.info(f"Method categories: {self.fight_details['method_category'].value_counts().to_dict()}")

    @staticmethod
    def _categorize_method(method: str) -> str:
        """Categorize fight methods into standard categories."""
        if pd.isna(method):
            return 'other'
        
        method = str(method).strip().lower()

        # KO/TKO category (including all stoppages)
        if any(x in method for x in ['ko', 'tko', 'doctor', 'stoppage']):
            return 'ko_tko'
        
        # Submission category
        elif 'submission' in method:
            return 'submission'
        
        # Decision category (all types)
        elif 'decision' in method:
            return 'decision'
        
        # Everything else (DQ, Could not continue, etc.)
        else:
            return 'other'

    def _merge_and_validate_winners(self) -> None:
        """Merge with event details and validate winner information."""
        if self.event_details is None:
            logger.warning("No event details provided - skipping winner validation")
            return
            
        logger.info("Merging event details and validating winners...")
        
        # Merge with event details
        initial_count = len(self.fight_details)
        self.fight_details = self.fight_details.merge(
            self.event_details[['winner', 'winner_id', 'fight_id', 'date']],
            on='fight_id',
            how='left'
        )
        
        # Filter out fights with no clear winner (null winner_id)
        fights_with_winners = self.fight_details[self.fight_details['winner_id'].notna()]
        excluded_count = len(self.fight_details) - len(fights_with_winners)
        
        self.fight_details = fights_with_winners
        
        logger.info(f"Excluded {excluded_count} fights with non-binary outcomes (draws/no contests/etc.)")
        logger.info(f"Retained {len(self.fight_details)} fights with clear winners")

    def _standardize_data_types(self) -> None:
        """Fix data types and standardize problematic values."""
        logger.info("Standardizing data types and values...")
        
        # Convert numeric columns that might be stored as strings
        numeric_cols = [col for col in self.fight_details.columns if 
                       any(x in col for x in ['landed', 'atmpted', 'kd', 'td_', 'ctrl', 'sig_st', 'total_st'])]
        
        for col in numeric_cols:
            if col in self.fight_details.columns:
                self.fight_details[col] = pd.to_numeric(self.fight_details[col], errors='coerce')
        
        # Handle any specific value standardizations
        # (You can add specific fixes here based on your EDA findings)
        
        logger.info("Data type standardization completed")

    def _validate_data(self) -> None:
        """Perform basic data validation checks."""
        logger.info("Performing data validation...")
        
        # Check for negative values in strike stats
        strike_cols = [col for col in self.fight_details.columns if 
                      any(x in col for x in ['landed', 'atmpted', 'kd', 'td_landed', 'td_atmpted'])]
        
        for col in strike_cols:
            if col in self.fight_details.columns:
                negative_count = (self.fight_details[col] < 0).sum()
                if negative_count > 0:
                    logger.warning(f"Found {negative_count} negative values in {col}")
        
        # Check fight time validity
        invalid_time = (self.fight_details['fight_time_sec'] <= 0).sum()
        if invalid_time > 0:
            logger.warning(f"Found {invalid_time} fights with invalid fight time")
        
        # Check for extremely long fights (over 25 minutes = 1500 seconds)
        very_long_fights = (self.fight_details['fight_time_sec'] > 1500).sum()
        if very_long_fights > 0:
            logger.warning(f"Found {very_long_fights} unusually long fights (>25 minutes)")

    def pre_process_data(self) -> pd.DataFrame:
        """
        Run the complete pre-processing pipeline.
        
        Returns:
        pd.DataFrame: Pre-processed fight details data
        """
        logger.info("Starting pre-processing pipeline...")
        
        original_shape = self.fight_details.shape
        
        # Run pre-processing steps in logical order
        self._standardize_data_types()
        self._calculate_fight_time()
        self._standardize_methods()
        
        # Handle event data if provided
        if self.event_details is not None:
            self._merge_and_validate_winners()
        
        self._validate_data()
        
        final_shape = self.fight_details.shape
        logger.info(f"Pre-processing complete. Shape: {original_shape} -> {final_shape}")
        
        return self.fight_details


def test_preprocessor(data_path:str = 'data/raw/'):
    """Test the PreProcessor with sample data."""
    print("Testing PreProcessor...")
    event_data = pd.read_csv(f"{data_path}event_details.csv", low_memory=False)
    fight_data = pd.read_csv(f"{data_path}fight_details.csv", low_memory=False)

    preprocessor = PreProcessor(fight_data, event_data)
    processed_data = preprocessor.pre_process_data()
    
    print("✓ PreProcessor test completed successfully!")
    print(f"✓ Processed data shape: {processed_data.shape}")
    
    return processed_data


if __name__ == "__main__":
    test_preprocessor()