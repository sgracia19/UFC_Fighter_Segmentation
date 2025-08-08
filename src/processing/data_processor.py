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
            
            # Log how many null winner_ids we have
            null_winners = self.event_details['winner_id'].isna().sum()
            total_fights = len(self.event_details)
            logger.info(f"Found {null_winners} fights ({null_winners/total_fights*100:.2f}%) with null winner_id - these will be excluded")
            
        except Exception as e:
            logger.error(f"Error loading data files: {e}")
            raise

    def prepare_fight_level_data(self) -> pd.DataFrame:
        """ 
        Transform fight data from red/blue corner format to fighter-level records.
        
        Returns:
            DataFrame with one row per fighter per fight
        """
        logger.info("Preparing fight-level data with binary outcomes only...")

        self.fight_details['fight_time_sec'] = self.fight_details.apply(self.calculate_fight_time, axis=1)

        # Merge fight details with event details
        fights_with_winners = self.fight_details.merge(
            self.event_details[['winner', 'winner_id', 'fight_id', 'date']],
            on='fight_id'
        )
        
        # Filter out fights with no clear winner (null winner_id)
        initial_count = len(fights_with_winners)
        fights_with_winners = fights_with_winners[fights_with_winners['winner_id'].notna()]
        excluded_count = initial_count - len(fights_with_winners)
        
        logger.info(f"Excluded {excluded_count} fights with non-binary outcomes (draws/no contests/etc.)")

        # Create binary win/loss indicators
        fights_with_winners['r_win'] = np.where(
            fights_with_winners['winner_id'] == fights_with_winners['r_id'], 1, 0
        )
        fights_with_winners['b_win'] = np.where(
            fights_with_winners['winner_id'] == fights_with_winners['b_id'], 1, 0
        )
        fights_with_winners['r_loss'] = np.where(
            fights_with_winners['winner_id'] == fights_with_winners['b_id'], 1, 0
        )
        fights_with_winners['b_loss'] = np.where(
            fights_with_winners['winner_id'] == fights_with_winners['r_id'], 1, 0
        )

        # Split into red and blue corner dataframes
        red_fighter_df = fights_with_winners.drop(columns=[
            col for col in fights_with_winners.columns if 'b_' in col
        ])
        blue_fighter_df = fights_with_winners.drop(columns=[
            col for col in fights_with_winners.columns if 'r_' in col
        ])

        # Rename columns
        red_fighter_df.columns = [col.replace('r_', '') for col in red_fighter_df.columns]
        blue_fighter_df.columns = [col.replace('b_', '') for col in blue_fighter_df.columns]

        # Combine datasets
        fighter_df = pd.concat([red_fighter_df, blue_fighter_df], ignore_index=True)
        fighter_df = fighter_df.loc[:, ~fighter_df.columns.duplicated()]

        # Clean up columns
        cols_to_drop = [
            'event_name', 'event_id', 'title_fight', 'referee',
            'winner', 'winneid', 'date', 'total_rounds', 'mathch_time_sec'
        ]
        cols_to_drop = [col for col in cols_to_drop if col in fighter_df.columns]
        fighter_df.drop(cols_to_drop, axis=1, inplace=True)

        logger.info(f"Prepared fighter-level data with {len(fighter_df)} records from binary outcomes")
        return fighter_df

    @staticmethod
    def categorize_method(method: str) -> str:
        """Simplified method categorization focusing on main finish types."""
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
        
    @staticmethod
    def calculate_fight_time(row) -> Optional[int]:
        """Calculate total fight time in seconds based on finish round and match time."""
        if row['finish_round'] == 1:
            return row['match_time_sec']
        elif row['finish_round'] == 2:
            return 300 + row['match_time_sec']  # 5 minutes = 300 seconds
        elif row['finish_round'] == 3:
            return 600 + row['match_time_sec']  # 10 minutes = 600 seconds
        elif row['finish_round'] == 4:
            return 900 + row['match_time_sec']  # 15 minutes = 900 seconds
        elif row['finish_round'] == 5:
            return 1200 + row['match_time_sec']  # 20 minutes = 1200 seconds
        else:
            return np.nan  # Handle unexpected values

    def create_method_columns(self, fighter_stats: pd.DataFrame, 
                            wins_by_method: pd.DataFrame, 
                            losses_by_method: pd.DataFrame) -> pd.DataFrame:
        """Ensure all method categories have corresponding win/loss columns."""
        logger.info("Creating method columns...")
        
        for category in self.method_categories:
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
        df_copy['method_category'] = df_copy['method'].apply(self.categorize_method)
        
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
            'sig_stlanded': 'sum',  # Total significant strikes landed
            'sig_statmpted': 'sum',  # Total significant strikes attempted
            'total_stlanded': 'sum',  # Total strikes landed
            'total_statmpted': 'sum',  # Total strikes attempted
            
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
            'suatt': 'sum',  # Total submission attempts
            
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
            ('career_sig_st_acc', 'sig_stlanded', 'sig_statmpted'),
            ('career_total_st_acc', 'total_stlanded', 'total_statmpted'),
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
        fighter_stats['sig_st_landed_per_min'] = np.where(
            fighter_stats['total_fight_time_sec'] > 0,
            (fighter_stats['sig_stlanded'] / fighter_stats['total_fight_time_sec']) * 60,
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
        for method in self.method_categories:
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