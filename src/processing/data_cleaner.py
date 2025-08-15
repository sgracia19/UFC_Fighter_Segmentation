import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Clean structured UFC data based on EDA insights and business rules.
    This class handles filtering, outlier removal, and data quality improvements.
    """

    def __init__(self):
        pass

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the input DataFrame by applying various cleaning steps.

        Parameters:
        df (pd.DataFrame): The raw input DataFrame.

        Returns:
        pd.DataFrame: The cleaned DataFrame.
        """
        logger.info("Starting data cleaning process.")

        df_cleaned = df.copy()

        initial_len = len(df_cleaned)

        # Clean weigth classes
        df_cleaned = self.clean_weight_class(df_cleaned)

        # Domain knowledge filtering
        df_cleaned = self.domain_filtering(df_cleaned)

        # Imputation and additional filtering
        df_cleaned = self.handle_missing_values(df_cleaned)

        # Add additional metrics
        df_cleaned = self.add_relative_physical_metrics(df_cleaned)

        final_len = len(df_cleaned)

        logger.info(f"Data cleaning process completed: dropped {initial_len - final_len} fighters")
        return df_cleaned
    
    def domain_filtering(self, df : pd.DataFrame) -> pd.DataFrame:
        """
        Filter and filter data based on domain knowledge.

        Parameters:
        df (pd.DataFrame): The DataFrame to process.

        Returns:
        pd.DataFrame: The DataFrame with invalid entries fixed and filtered      
        """
        # Replace invalid stances with 'Orthodox'
        df['stance'] = df['stance'].replace({'Open Stance':'Orthodox', 'Sideways':'Orthodox'})

        # Set 4 fight minimum threshold for statistcal balance between sample size and data stability
        df_filtered = df[df['total_UFC_fights'] >= 4]

        # Drop catchweight and unknown weight classes for clustering
        df_filtered = df_filtered[~df_filtered['weight_class'].isin(['Catchweight', 'Unknown', "Women's Featherweight"])]

        df_filtered = df_filtered.loc[df_filtered['sig_str_landed'] > 0 ]

        return df_filtered
    
    def clean_weight_class(self, df):
        """
        Clean and infer weight classes from division text and weight data.
        
        Args:
            df: DataFrame with 'division' and 'weight' columns
            
        Returns:
            DataFrame with added 'weight_class' column
        """
        def parse_weight_class_from_text(division):
            """Parse weight class from division text"""
            if pd.isna(division):
                return 'Unknown'
            
            division = division.lower()
            
            if "women" in division or "woman" in division:
                if "flyweight" in division:
                    return "Women's Flyweight"
                elif "bantamweight" in division:
                    return "Women's Bantamweight"
                elif "featherweight" in division:
                    return "Women's Featherweight"
                elif "strawweight" in division:
                    return "Women's Strawweight"
                else:
                    return "Unknown"
            else:
                if "flyweight" in division:
                    return "Flyweight"
                elif "bantamweight" in division:
                    return "Bantamweight"
                elif "featherweight" in division:
                    return "Featherweight"
                elif "lightweight" in division:
                    return "Lightweight"
                elif "welterweight" in division:
                    return "Welterweight"
                elif "middleweight" in division:
                    return "Middleweight"
                elif "light heavyweight" in division:
                    return "Light Heavyweight"
                elif "heavyweight" in division:
                    return "Heavyweight"
                else:
                    return "Unknown"
        
        def infer_from_weight(parsed_class, weight):
            """Infer weight class from weight when text parsing fails"""
            if parsed_class != 'Unknown':
                return parsed_class
            
            if pd.isna(weight):
                return 'Unknown'
                
            # Weight-based classification (may misclassify some women fighters)
            if weight <= 52.2:
                return "Women's Strawweight"
            elif weight <= 56.7:
                return "Flyweight"
            elif weight <= 61.2:
                return "Bantamweight"
            elif weight <= 65.8:
                return "Featherweight"
            elif weight <= 70.3:
                return "Lightweight"
            elif weight <= 77.1:
                return "Welterweight"
            elif weight <= 83.9:
                return "Middleweight"
            elif weight <= 93.0:
                return "Light Heavyweight"
            elif weight <= 120.2:
                return "Heavyweight"
            else:
                return "Catchweight"
        
        # Apply parsing and inference
        df = df.copy()
        df['parsed_weight_class'] = df['division'].apply(parse_weight_class_from_text)
        df['weight_class'] = df.apply(
            lambda row: infer_from_weight(row['parsed_weight_class'], row['weight']), 
            axis=1
        )
        
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame to process.

        Returns:
        pd.DataFrame: The DataFrame with missing values handled.
        """
        logger.info("Handling missing values...")

        reach_medians = df.groupby('weight_class')['reach'].median()

        def impute_reach(row):
            if pd.isna(row['reach']):
                return reach_medians.get(row['weight_class'], df['reach'].median())
            return row['reach']
        
        df = df.copy()  # Make sure it's a copy, not a view
        df['reach'] = df.apply(impute_reach, axis=1)
        
        return df   
    
    def add_relative_physical_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add relative height and reach metrics based on weight class averages."""
        logger.info("Adding relative height and reach metrics...")
        
        # Calculate division averages
        division_stats = df.groupby('division')[['height', 'reach']].agg(['mean', 'std'])
        division_stats.columns = ['_'.join(col).strip() for col in division_stats.columns]
        
        # Merge division stats
        df = df.merge(
            division_stats, 
            left_on='division', 
            right_index=True, 
            how='left'
        )
        
        # Calculate relative metrics
        df['relative_height'] = df['height'] - df['height_mean']
        df['relative_reach'] = df['reach'] - df['reach_mean']
        df['height_z_score'] = (df['height'] - df['height_mean']) / df['height_std']
        df['reach_z_score'] = (df['reach'] - df['reach_mean']) / df['reach_std']
        
        # Clean up
        df.drop(columns=['height_mean', 'height_std', 'reach_mean', 'reach_std'], inplace=True)
        
        return df