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
from pre_processor import PreProcessor # Data PreProcessor

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
        
        # Simplified method categories
        self.method_categories = [
            'ko_tko', 'submission', 'decision', 'other'
        ]
        
        logger.info(f"DataProcessor initialized with data path: {data_path}")

    def load_raw_data(self) -> None:
        """Load raw data files into pandas DataFrames."""
        try:
            self.event_details = pd.read_csv(f"{self.data_path}event_details.csv", low_memory=False)
            self.fighter_details = pd.read_csv(f"{self.data_path}fighter_details.csv", low_memory=False)
            self.fight_details = pd.read_csv(f"{self.data_path}fight_details.csv", low_memory=False)
            
            logger.info("Data loaded successfully. {} events, {} fighters, {} fights.".format(
                len(self.event_details), len(self.fighter_details), len(self.fight_details)))
            
            # Preprocess the data
            preprocessor = PreProcessor(self.fight_details, self.event_details)
            self.fight_details = preprocessor.pre_process_data()
            
            logger.info(f"Data preprocessing completed. {len(self.fight_details)} fights ready for processing")
            
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
        
        logger.info(f"Found {len(red_cols)} red columns, {len(blue_cols)} blue columns, {len(shared_cols)} shared columns")

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
            'winner', 'winneid', 'date', 'total_rounds'
        ]
        existing_cols_to_drop = [col for col in cols_to_drop if col in fighter_df.columns]
        if existing_cols_to_drop:
            fighter_df.drop(columns=existing_cols_to_drop, inplace=True)
            logger.info(f"Dropped {len(existing_cols_to_drop)} unnecessary columns")

        logger.info(f"✓ Prepared fighter-level data: {len(fighter_df)} records, {len(fighter_df.columns)} columns")
        
        return fighter_df
    
    def create_method_columns(self, fighter_stats: pd.DataFrame, 
                            wins_by_method: pd.DataFrame, 
                            losses_by_method: pd.DataFrame) -> pd.DataFrame:
        """Ensure all method categories have corresponding win/loss columns."""
        logger.info("Creating method columns...")
        
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
        
        # Debug info
        logger.info(f"Method categories found: {df_copy['method_category'].unique()}")

        # Aggregation dictionary
        agg_dict = {
            # Basic fight information
            'fight_id': 'count',  # Total number of fights
            'win': 'sum',  # Total wins
            'loss': 'sum',  # Total losses
            
            # Time-based metrics - sum totals, then can calculate averages later
            'fight_time_sec': 'sum',  # Total fight time in seconds
            'finish_round': 'sum',  # Total rounds fought
            
            # Event counts - sum across all fights
            'kd': 'sum',  # Total knockdowns dealt
            
            # Strike totals - sum for career totals
            'sig_str_landed': 'sum', # Significant Strikes landed
            'sig_str_atmpted': 'sum', # Significant Strikes landed
            'total_str_landed': 'sum',  # Total strikes landed
            'total_str_atmpted': 'sum',  # Total strikes attempted
            
            # Takedown totals - sum for career totals
            'td_landed': 'sum',  # Total takedowns landed
            'td_atmpted': 'sum',  # Total takedowns attempted
            
            # Control time - sum for total career control time
            'ctrl': 'sum',  # Total control time
            
            # Strike location totals - sum for career totals
            'head_landed': 'sum',
            'head_atmpted': 'sum',
            'body_landed': 'sum',
            'body_atmpted': 'sum',
            'leg_landed': 'sum',
            'leg_atmpted': 'sum',
            
            # Strike position totals - sum for career totals
            'dist_landed': 'sum',  # Distance strikes landed
            'dist_atmpted': 'sum',  # Distance strikes attempted
            'clinch_landed': 'sum',  # Clinch strikes landed
            'clinch_atmpted': 'sum',  # Clinch strikes attempted
            'ground_landed': 'sum',  # Ground strikes landed
            'ground_atmpted': 'sum',  # Ground strikes attempted
            
            # Submission attempts
            'sub_att': 'sum',  # Total submission attempts
            
            # Percentages and accuracy - we'll recalculate these from totals
            # Don't aggregate the existing percentage columns directly as they're per-fight
            
            # Keep fighter metadata (take first occurrence)
            'name': 'first',
            'division': 'last',  # Use last in case fighter changed divisions
        }

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
            'fight_time_sec': 'total_fight_time_sec'
        })
        
        logger.info(f"Aggregated data for {len(fighter_stats)} fighters")
        return fighter_stats

    def calculate_derived_metrics(self, fighter_stats: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived metrics from aggregated stats."""
        logger.info("Calculating derived metrics...")

        # Accuracy percentages
        accuracy_metrics = [
            ('career_total_sig_str_acc', 'sig_str_landed', 'sig_str_atmpted'),
            ('career_total_str_acc', 'total_str_landed', 'total_str_atmpted'),
            ('career_td_acc', 'td_landed', 'td_atmpted'),
            ('career_head_acc', 'head_landed', 'head_atmpted'),
            ('career_body_acc', 'body_landed', 'body_atmpted'),
            ('career_leg_acc', 'leg_landed', 'leg_atmpted'),
            ('career_dist_acc', 'dist_landed', 'dist_atmpted'),
            ('career_clinch_acc', 'clinch_landed', 'clinch_atmpted'),
            ('career_ground_acc', 'ground_landed', 'ground_atmpted'),
        ]
        
        for acc_name, landed_col, attempted_col in accuracy_metrics:
            fighter_stats[acc_name] = np.where(
                fighter_stats[attempted_col] > 0,
                (fighter_stats[landed_col] / fighter_stats[attempted_col]) * 100,
                0
            )

        # Per-minute rates
        fighter_stats['str_landed_per_min'] = np.where(
            fighter_stats['total_fight_time_sec'] > 0,
            (fighter_stats['total_str_landed'] / fighter_stats['total_fight_time_sec']) * 60,
            0
        )
        
        fighter_stats['td_landed_per_min'] = np.where(
            fighter_stats['total_fight_time_sec'] > 0,
            (fighter_stats['td_landed'] / fighter_stats['total_fight_time_sec']) * 60,
            0
        )
        
        fighter_stats['kd_per_min'] = np.where(
            fighter_stats['total_fight_time_sec'] > 0,
            (fighter_stats['kd'] / fighter_stats['total_fight_time_sec']) * 60,
            0
        )

        fighter_stats['ctrl_per_min'] = np.where(
            fighter_stats['total_fight_time_sec'] > 0,
            (fighter_stats['ctrl'] / fighter_stats['total_fight_time_sec']) * 60,
            0
        )

        # Basic metrics
        fighter_stats['avg_fight_time_sec'] = fighter_stats['total_fight_time_sec'] / fighter_stats['total_UFC_fights']
        fighter_stats['win_percentage'] = (fighter_stats['UFC_wins'] / fighter_stats['total_UFC_fights']) * 100

        # Finish rates
        fighter_stats['finish_rate'] = np.where(
            fighter_stats['total_UFC_fights'] > 0,
            ((fighter_stats['wins_by_ko_tko'] + fighter_stats['wins_by_submission']) / 
             fighter_stats['total_UFC_fights']) * 100,
            0
        )

        # Method-specific rates
        method_categories = ['ko_tko', 'submission', 'decision', 'other']
        for method in method_categories:
            # Win rates
            fighter_stats[f'{method}_win_rate'] = np.where(
                fighter_stats['total_UFC_fights'] > 0,
                (fighter_stats[f'wins_by_{method}'] / fighter_stats['total_UFC_fights']) * 100,
                0
            )
            
            # Loss vulnerability
            fighter_stats[f'{method}_vulnerability'] = np.where(
                fighter_stats['total_UFC_fights'] > 0,
                (fighter_stats[f'losses_by_{method}'] / fighter_stats['total_UFC_fights']) * 100,
                0
            )

        # Strike distribution
        total_strikes_landed = (
            fighter_stats['head_landed'] + 
            fighter_stats['body_landed'] + 
            fighter_stats['leg_landed']
        )
        
        for location in ['head', 'body', 'leg']:
            fighter_stats[f'{location}_strike_percentage'] = np.where(
                total_strikes_landed > 0,
                (fighter_stats[f'{location}_landed'] / total_strikes_landed) * 100,
                0
            )
        
        return fighter_stats

    def add_fighter_details(self, fighter_stats: pd.DataFrame) -> pd.DataFrame:
        """Merge with fighter metadata."""
        logger.info("Merging fighter details...")
        
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
        logger.info("Starting simplified processing pipeline (binary outcomes only)...")
        
        self.load_raw_data()
        fighter_df = self.prepare_fight_level_data()
        fighter_stats = self.aggregate_fighter_stats(fighter_df)
        fighter_stats = self.calculate_derived_metrics(fighter_stats)
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
        logger.info(f"Excluded rare outcomes to focus on {len(complete_fighter_stats)} fighters with clear win/loss records")
        
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