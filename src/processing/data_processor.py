"""
UFC Fighter Segmentation - Data Processing Module
This module handles the processing of raw UFC fighter data into a structured format suitable for analysis.
It includes functions for cleaning, transforming, and aggregating fighter statistics.
It also ensures that the data is consistent and ready for downstream tasks such as modeling and visualization.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging
from .pre_processor import PreProcessor # Data PreProcessor
from .metric_calculator import UFCMetricCalculator # Adding Metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UFCDataProcessor:
    """Simplified processor focusing on binary win/loss outcomes."""

    def __init__(self, data_path: str = 'data/raw/'):
        """
        Initializes the DataProcessor with path to raw data files.
        
        Parameters:
        data_path (str): Path to the directory containing raw data files.
        """
        self.data_path = data_path
        self.event_details = None
        self.fighter_details = None
        self.fight_details = None
        self.processed_data = None

        # Intitialize metric calculator
        self.metric_calculator = UFCMetricCalculator()
        
        # Simplified method categories
        self.method_categories = [
            'ko_tko', 'submission', 'decision', 'other'
        ]
        
        logger.debug(f"DataProcessor initialized with data path: {data_path}")

    def load_raw_data(self) -> None:
        """Load raw data files into pandas DataFrames."""
        try:
            self.event_details = pd.read_csv(f"{self.data_path}event_details.csv", low_memory=False)
            self.event_details['date'] = pd.to_datetime(self.event_details['date'])
            self.fighter_details = pd.read_csv(f"{self.data_path}fighter_details.csv", low_memory=False)
            self.fight_details = pd.read_csv(f"{self.data_path}fight_details.csv", low_memory=False)
            
            logger.info("Data loaded successfully. {} events, {} fighters, {} fights.".format(
                len(self.event_details), len(self.fighter_details), len(self.fight_details)))
            
            # Preprocess the data
            preprocessor = PreProcessor(self.fight_details, self.event_details)
            self.fight_details = preprocessor.pre_process_data()

            # Add strikes absorbed to fight_details
            self.fight_details = self.metric_calculator.add_complementary_metrics(self.fight_details)
                        
        except Exception as e:
            logger.error(f"Error loading/preprocessing data files: {e}")
            raise

    def prepare_fight_level_data(self) -> pd.DataFrame:
        """
        Transform preprocessed fight data from red/blue corner format to fighter-level records.
        
        Returns:
            DataFrame with one row per fighter per fight
        """
        logger.info("Preparing fight-level data from preprocessed inputs...")

        # Create binary win/loss indicators (business logic - stays in main processor)
        self.fight_details['r_win'] = np.where(
            self.fight_details['winner_id'] == self.fight_details['r_id'], 1, 0
        )
        self.fight_details['b_win'] = np.where(
            self.fight_details['winner_id'] == self.fight_details['b_id'], 1, 0
        )
        self.fight_details['r_loss'] = np.where(
            self.fight_details['winner_id'] == self.fight_details['b_id'], 1, 0
        )
        self.fight_details['b_loss'] = np.where(
            self.fight_details['winner_id'] == self.fight_details['r_id'], 1, 0
        )

        # Identify column categories
        all_cols = set(self.fight_details.columns)
        red_cols = {col for col in all_cols if col.startswith('r_')}
        blue_cols = {col for col in all_cols if col.startswith('b_')}
        shared_cols = all_cols - red_cols - blue_cols
        
        logger.debug(f"Found {len(red_cols)} red columns, {len(blue_cols)} blue columns, {len(shared_cols)} shared columns")

        # Create separate dataframes for each corner
        red_fighter_df = self.fight_details[list(red_cols) + list(shared_cols)].copy()
        blue_fighter_df = self.fight_details[list(blue_cols) + list(shared_cols)].copy()

        # Remove prefixes to standardize column names
        red_fighter_df.rename(columns={col: col[2:] if col.startswith('r_') else col 
                                    for col in red_fighter_df.columns}, inplace=True)
        blue_fighter_df.rename(columns={col: col[2:] if col.startswith('b_') else col 
                                    for col in blue_fighter_df.columns}, inplace=True)

        # Combine red and blue fighter records
        fighter_df = pd.concat([red_fighter_df, blue_fighter_df], ignore_index=True)

        # Clean up unnecessary columns
        cols_to_drop = [
            'event_name', 'event_id', 'title_fight', 'referee',
            'winner', 'winner_id', 'total_rounds'
        ]
        existing_cols_to_drop = [col for col in cols_to_drop if col in fighter_df.columns]
        if existing_cols_to_drop:
            fighter_df.drop(columns=existing_cols_to_drop, inplace=True)
            logger.debug(f"Dropped {len(existing_cols_to_drop)} unnecessary columns")

        logger.info(f"✓ Prepared fighter-level data: {len(fighter_df)} records, {len(fighter_df.columns)} columns")
        
        return fighter_df
    
    def create_method_columns(self, fighter_stats: pd.DataFrame, 
                            wins_by_method: pd.DataFrame, 
                            losses_by_method: pd.DataFrame) -> pd.DataFrame:
        """Ensure all method categories have corresponding win/loss columns."""
        logger.debug("Creating method columns...")
        
        # Method categories now come from preprocessing
        method_categories = ['ko_tko', 'submission', 'decision', 'other']
        
        for category in method_categories:
            win_col = f'wins_by_{category}'
            loss_col = f'losses_by_{category}'
            
            # Add wins column
            if win_col in wins_by_method.columns:
                fighter_stats = fighter_stats.merge(
                    wins_by_method[[win_col]], 
                    left_on='id', right_index=True, how='left'
                )
                fighter_stats[win_col] = fighter_stats[win_col].fillna(0)
            else:
                fighter_stats[win_col] = 0
                
            # Add losses column
            if loss_col in losses_by_method.columns:
                fighter_stats = fighter_stats.merge(
                    losses_by_method[[loss_col]], 
                    left_on='id', right_index=True, how='left'
                )
                fighter_stats[loss_col] = fighter_stats[loss_col].fillna(0)
            else:
                fighter_stats[loss_col] = 0
        
        return fighter_stats

    def aggregate_fighter_stats(self, fighter_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate fight-level data to fighter statistics."""
        logger.info("Aggregating fighter statistics...")
        
        # Add method categories
        df_copy = fighter_df.copy()

        # Aggregation dictionary from metric calculator
        agg_dict = self.metric_calculator.get_enhanced_aggregation_dict()

        # Group by fighter and aggregate
        fighter_stats = df_copy.groupby('id').agg(agg_dict).reset_index()

        # Create win/loss by method breakdown
        wins_by_method = df_copy[df_copy['win'] == 1].groupby(['id', 'method_category']).size().unstack(fill_value=0)
        wins_by_method.columns = [f'wins_by_{col}' for col in wins_by_method.columns]
        
        losses_by_method = df_copy[df_copy['loss'] == 1].groupby(['id', 'method_category']).size().unstack(fill_value=0)
        losses_by_method.columns = [f'losses_by_{col}' for col in losses_by_method.columns]
        
        # Add method columns
        fighter_stats = self.create_method_columns(fighter_stats, wins_by_method, losses_by_method)
        
        # Rename columns for clarity
        fighter_stats = fighter_stats.rename(columns={
            'fight_id': 'total_UFC_fights',
            'win': 'UFC_wins', 
            'loss': 'UFC_losses',
            'finish_round': 'total_rounds_fought',
            'fight_time_sec': 'total_fight_time_sec',
            'date' : 'last_fight'
        })
        
        logger.info(f"Aggregated data for {len(fighter_stats)} fighters")
        return fighter_stats

    def add_fighter_details(self, fighter_stats: pd.DataFrame) -> pd.DataFrame:
        """Merge with fighter metadata."""
        logger.debug("Merging fighter details...")
        
        complete_fighter_stats = fighter_stats.merge(
            self.fighter_details,
            on=['id', 'name'],
            how='left'
        )
        
        # Clean up redundant columns
        redundant_columns = ['wins', 'losses', 'draws']
        cols_to_drop = [col for col in redundant_columns if col in complete_fighter_stats.columns]
        if cols_to_drop:
            complete_fighter_stats.drop(columns=cols_to_drop, inplace=True)
        
        return complete_fighter_stats

    def process_data(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """Run the full processing pipeline."""
        logger.debug("Starting processing pipeline...")
        
        self.load_raw_data()
        fighter_df = self.prepare_fight_level_data()
        fighter_stats = self.aggregate_fighter_stats(fighter_df)
        # fighter_stats = self.calculate_derived_metrics(fighter_stats)
        fighter_stats = self.metric_calculator.calculate_derived_metrics(fighter_stats)
        complete_fighter_stats = self.add_fighter_details(fighter_stats)
        
        self.processed_data = complete_fighter_stats

        # Validation check
        total_outcomes = complete_fighter_stats['UFC_wins'] + complete_fighter_stats['UFC_losses']
        expected_total = complete_fighter_stats['total_UFC_fights']
        
        if not (total_outcomes == expected_total).all():
            logger.warning("Win + Loss doesn't equal total fights for some fighters - check data quality")
        else:
            logger.info("✓ Data validation passed: wins + losses = total fights")

        if save_path:
            complete_fighter_stats.to_csv(save_path, index=False)
            logger.info(f"Processed data saved to {save_path}")
            
        logger.info(f"Processing complete! Final dataset shape: {complete_fighter_stats.shape}")
        
        return complete_fighter_stats

def quick_test():
    """Test the simplified processor."""
    print("Testing Simplified UFCDataProcessor...")
    processor = UFCDataProcessor(data_path='data/raw/')
    
    try:
        complete_data = processor.process_data()
        method_cols = [col for col in complete_data.columns if 'wins_by_' in col or 'losses_by_' in col]
        
        print(f"✓ Complete processing successful!")
        print(f"✓ Method columns: {method_cols}")
        print(f"✓ Final dataset shape: {complete_data.shape}")
        
        # Show some stats
        total_wins = complete_data['UFC_wins'].sum()
        total_losses = complete_data['UFC_losses'].sum()
        total_fights = complete_data['total_UFC_fights'].sum()
        
        print(f"✓ Total fights processed: {total_fights}")
        print(f"✓ Total wins: {total_wins}, Total losses: {total_losses}")
        print(f"✓ Win/Loss balance check: {total_wins == total_losses}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()