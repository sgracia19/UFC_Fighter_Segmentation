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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UFCDataProcessor:
    """Main class for processing UFC fighter data."""

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
        logger.info(f"DataProcessor initialized with data path: {data_path}")

    def load_raw_data(self) -> None:
        """ Load raw data files into pandas DataFrames. """
        try:
            self.event_details = pd.read_csv(f"{self.data_path}event_details.csv")
            self.fighter_details = pd.read_csv(f"{self.data_path}fighter_details.csv")
            self.fight_details = pd.read_csv(f"{self.data_path}fight_details.csv")
            logger.info("Data loaded successfully. {} events, {} fighters, {} fights.".format(
                len(self.event_details), len(self.fighter_details), len(self.fight_details)))
        except Exception as e:
            logger.error(f"Error loading data files: {e}")
            raise

    def prepare_fight_level_data(self) -> pd.DataFrame:
        """ 
        Transform fight data from red/blue corner format to fighter-level records.
        
        Returns:
            DataFrame with one row per fighter per fight
        """
        logger.info("Preparing fight-level data...")

        # Merge fight details with event details to get winner information
        fights_with_winners = self.fight_details.merge(
            self.event_details[['winner', 'winner_id', 'fight_id', 'date']],
            on='fight_id'
        )

        # Create red corner and blue corner dataframes
        fights_with_winners['r_win'] = np.where(fights_with_winners['winner_id'] == fights_with_winners['r_id'], 1, 0)
        fights_with_winners['b_win'] = np.where(fights_with_winners['winner_id'] == fights_with_winners['b_id'], 1, 0)
        fights_with_winners['r_loss'] = np.where(fights_with_winners['winner_id'] == fights_with_winners['b_id'], 1, 0)
        fights_with_winners['b_loss'] = np.where(fights_with_winners['winner_id'] == fights_with_winners['r_id'], 1, 0)

        # Split into red and blue corner dataframes
        red_fighter_df = fights_with_winners.drop(columns = [col for col in fights_with_winners.columns if 'b_' in col])
        blue_fighter_df = fights_with_winners.drop(columns = [col for col in fights_with_winners.columns if 'r_' in col])

        # Rename columns for clarity
        red_fighter_df.columns  = [col.replace('r_', '') for col in red_fighter_df.columns]
        blue_fighter_df.columns = [col.replace('b_', '') for col in blue_fighter_df.columns]

        # Combine datasets
        fighter_df = pd.concat([red_fighter_df, blue_fighter_df], ignore_index=True)
        fighter_df = fighter_df.loc[:, ~fighter_df.columns.duplicated()]  # Remove duplicate columns

        # Clean up columns
        cols_to_drop = ['event_name', 'event_id', 'title_fight', 'referee',
                         'winner' , 'winneid', 'date','finish_round']
        fighter_df.drop(cols_to_drop, axis=1, inplace = True)

        logger.info(f"Prepared fighter-level data with {len(fighter_df)} records.")
        return fighter_df
    
    @staticmethod
    def categorize_method(method:str) -> str:
        """ Method to categorize fight finish methods. """
        if pd.isna(method):
            return 'Other'
        method = str(method).strip()

        if 'KO' in method or 'TKO' in method:
            return 'KO/TKO'
        elif 'Submission' in method:
            return 'Submission'
        elif 'Decision' in method:
            return 'Decision'
        elif method in ['Could Not Continue', "TKO - Doctor's Stoppage"]:
            return 'Stoppage'
        elif method in ['DQ', 'Overturned']:
            return 'DQ/Overturned'
        else:
            return 'Other'
    
    def aggregate_fighter_stats(self, fighter_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate fight-level data to compute cumulative fighter statistics.
        
        Parameters:
            fighter_df (DataFrame): DataFrame with one row per fighter per fight.
        
        Returns:
            DataFrame with aggregated fighter statistics.
        """
        logger.info("Aggregating fighter statistics...")
        
        # Add method categories
        df_copy = fighter_df.copy()
        df_copy['method_category'] = df_copy['method'].apply(self.categorize_method)

        # Define aggregation functions
        agg_dict = {
            # Basic fight information
            'fight_id': 'count',  # Total number of fights
            'win': 'sum',  # Total wins
            'loss': 'sum',  # Total losses
            
            # Time-based metrics - sum totals, then can calculate averages later
            'match_time_sec': 'sum',  # Total fight time in seconds
            'total_rounds': 'sum',  # Total rounds fought
            
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

        fighter_stats = df_copy.groupby('id').agg(agg_dict).reset_index()

        # Add win/loss by method features
        wins_by_method = df_copy[df_copy['win'] == 1].groupby(['id', 'method_category']).size().unstack(fill_value=0)
        wins_by_method.columns = [f'wins_by_{col.lower().replace("/", "_").replace(" ", "_")}' 
                                 for col in wins_by_method.columns]
        
        losses_by_method = df_copy[df_copy['loss'] == 1].groupby(['id', 'method_category']).size().unstack(fill_value=0)
        losses_by_method.columns = [f'losses_by_{col.lower().replace("/", "_").replace(" ", "_")}' 
                                   for col in losses_by_method.columns]
        
        # Merge method breakdowns
        fighter_stats = fighter_stats.merge(wins_by_method, left_on='id', right_index=True, how='left')
        fighter_stats = fighter_stats.merge(losses_by_method, left_on='id', right_index=True, how='left')
        
        # Fill NaN values for method columns
        method_columns = [col for col in fighter_stats.columns if 'wins_by_' in col or 'losses_by_' in col]
        fighter_stats[method_columns] = fighter_stats[method_columns].fillna(0)
        
        # Rename fight count column and win/loss columns for clarity
        fighter_stats = fighter_stats.rename(columns={
            'fight_id': 'total_UFC_fights',
            'win': 'UFC_wins', 
            'loss': 'UFC_losses'
        })

        # Drop columns that are not needed for analysis or have too many missing values
        cols_to_drop = ['wins', 'losses', 'draws']
        fighter_stats.drop(columns=cols_to_drop, inplace=True)
        
        logger.info(f"Aggregated data for {len(fighter_stats)} fighters")
        
        return fighter_stats

    def calculate_derived_metrics(self, fighter_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived metrics such as accuracy percentages and average fight time.
        
        Parameters:
            fighter_stats (DataFrame): DataFrame with aggregated fighter statistics.
        
        Returns:
            DataFrame with additional derived metrics.
        """
        logger.info("Calculating derived metrics...")

        # Accuracy percentages
        fighter_stats['career_sig_st_acc'] = np.where(
            fighter_stats['sig_statmpted'] > 0,
            (fighter_stats['sig_stlanded'] / fighter_stats['sig_statmpted']) * 100,
            0
        )
        
        fighter_stats['career_total_st_acc'] = np.where(
            fighter_stats['total_statmpted'] > 0,
            (fighter_stats['total_stlanded'] / fighter_stats['total_statmpted']) * 100,
            0
        )
        
        fighter_stats['career_td_acc'] = np.where(
            fighter_stats['td_atmpted'] > 0,
            (fighter_stats['td_landed'] / fighter_stats['td_atmpted']) * 100,
            0
        )
        
        fighter_stats['career_head_acc'] = np.where(
            fighter_stats['head_atmpted'] > 0,
            (fighter_stats['head_landed'] / fighter_stats['head_atmpted']) * 100,
            0
        )
        
        fighter_stats['career_body_acc'] = np.where(
            fighter_stats['body_atmpted'] > 0,
            (fighter_stats['body_landed'] / fighter_stats['body_atmpted']) * 100,
            0
        )
        
        fighter_stats['career_leg_acc'] = np.where(
            fighter_stats['leg_atmpted'] > 0,
            (fighter_stats['leg_landed'] / fighter_stats['leg_atmpted']) * 100,
            0
        )
        
        fighter_stats['career_dist_acc'] = np.where(
            fighter_stats['dist_atmpted'] > 0,
            (fighter_stats['dist_landed'] / fighter_stats['dist_atmpted']) * 100,
            0
        )
        
        fighter_stats['career_clinch_acc'] = np.where(
            fighter_stats['clinch_atmpted'] > 0,
            (fighter_stats['clinch_landed'] / fighter_stats['clinch_atmpted']) * 100,
            0
        )
        
        fighter_stats['career_ground_acc'] = np.where(
            fighter_stats['ground_atmpted'] > 0,
            (fighter_stats['ground_landed'] / fighter_stats['ground_atmpted']) * 100,
            0
        )
        
        # Per-minute rates (using total fight time)
        fighter_stats['sig_st_landed_per_min'] = np.where(
            fighter_stats['match_time_sec'] > 0,
            (fighter_stats['sig_stlanded'] / fighter_stats['match_time_sec']) * 60,
            0
        )
        
        # fighter_stats['sig_st_absorbed_per_min'] = np.where(
        #     fighter_stats['match_time_sec'] > 0,
        #     (fighter_stats['sig_stlanded'] / fighter_stats['match_time_sec']) * 60,  # This would need opponent data
        #     0
        # )
        
        fighter_stats['td_landed_per_min'] = np.where(
            fighter_stats['match_time_sec'] > 0,
            (fighter_stats['td_landed'] / fighter_stats['match_time_sec']) * 60,
            0
        )
        
        fighter_stats['kd_per_min'] = np.where(
            fighter_stats['match_time_sec'] > 0,
            (fighter_stats['kd'] / fighter_stats['match_time_sec']) * 60,
            0
        )
        
        # Average fight time
        fighter_stats['avg_fight_time_sec'] = fighter_stats['match_time_sec'] / fighter_stats['total_UFC_fights']
        
        # Win percentage
        fighter_stats['win_percentage'] = (fighter_stats['UFC_wins'] / fighter_stats['total_UFC_fights']) * 100
        
        # Calculate finish rates (useful for clustering)
        fighter_stats['finish_rate'] = np.where(
            fighter_stats['total_UFC_fights'] > 0,
            ((fighter_stats.get('wins_by_ko_tko', 0) + 
            fighter_stats.get('wins_by_submission', 0)) / fighter_stats['total_UFC_fights']) * 100,
            0
        )
        
        fighter_stats['ko_tko_rate'] = np.where(
            fighter_stats['total_UFC_fights'] > 0,
            (fighter_stats.get('wins_by_ko_tko', 0) / fighter_stats['total_UFC_fights']) * 100,
            0
        )
        
        fighter_stats['submission_rate'] = np.where(
            fighter_stats['total_UFC_fights'] > 0,
            (fighter_stats.get('wins_by_submission', 0) / fighter_stats['total_UFC_fights']) * 100,
            0
        )
        
        fighter_stats['decision_rate'] = np.where(
            fighter_stats['total_UFC_fights'] > 0,
            (fighter_stats.get('wins_by_decision', 0) / fighter_stats['total_UFC_fights']) * 100,
            0
        )
        
        # Calculate loss vulnerability rates
        fighter_stats['ko_tko_vulnerability'] = np.where(
            fighter_stats['total_UFC_fights'] > 0,
            (fighter_stats.get('losses_by_ko_tko', 0) / fighter_stats['total_UFC_fights']) * 100,
            0
        )
        
        fighter_stats['submission_vulnerability'] = np.where(
            fighter_stats['total_UFC_fights'] > 0,
            (fighter_stats.get('losses_by_submission', 0) / fighter_stats['total_UFC_fights']) * 100,
            0
        )
        
        # Strike distribution percentages (of total strikes landed)
        total_strikes_landed = (
            fighter_stats['head_landed'] + 
            fighter_stats['body_landed'] + 
            fighter_stats['leg_landed']
        )
        
        fighter_stats['head_strike_percentage'] = np.where(
            total_strikes_landed > 0,
            (fighter_stats['head_landed'] / total_strikes_landed) * 100,
            0
        )
        
        fighter_stats['body_strike_percentage'] = np.where(
            total_strikes_landed > 0,
            (fighter_stats['body_landed'] / total_strikes_landed) * 100,
            0
        )
        
        fighter_stats['leg_strike_percentage'] = np.where(
            total_strikes_landed > 0,
            (fighter_stats['leg_landed'] / total_strikes_landed) * 100,
            0
        )
        
        return fighter_stats
    
    def add_fighter_details(self, fighter_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Merge aggregated fighter stats with fighter metadata.
        
        Parameters:
            fighter_stats (DataFrame): DataFrame with aggregated fighter statistics.
        
        Returns:
            DataFrame with fighter metadata included.
        """
        logger.info("Merging fighter details...")
        
        # Merge on fighter ID
        complete_fighter_stats = fighter_stats.merge(
            self.fighter_details,
            on=['id', 'name'],
            how='left'
        )
                
        return complete_fighter_stats
    

    def process_data(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Run the full data processing pipeline.
        Parameters:
            save_path (str, optional): If provided, saves the processed data to this path as CSV.
        Returns:
            DataFrame with fully processed fighter statistics.
        """
        logger.info("Starting full data processing pipeline...")
        # Load raw data
        self.load_raw_data()

        # Spilt data to fighter level from fight level
        fighter_df = self.prepare_fight_level_data()

        # Aggregate data to one row per fighter
        fighter_stats = self.aggregate_fighter_stats(fighter_df)

        # Calculate derived metrics
        fighter_stats = self.calculate_derived_metrics(fighter_stats)

        # Merge with fighter details
        complete_fighter_stats = self.add_fighter_details(fighter_stats)

        # Store processed data
        self.processed_data = complete_fighter_stats

        # Save to CSV if path provided
        if save_path:
            try:
                complete_fighter_stats.to_csv(save_path, index=False)
                logger.info(f"Processed data saved to {save_path}")
            except Exception as e:
                logger.error(f"Error saving processed data: {e}")
                raise
        logger.info(f"Processing complete! Final dataset shape: {complete_fighter_stats.shape}")

        return complete_fighter_stats

        
def quick_test():
    """Quick test function."""
    print("Testing UFCDataProcessor...")
    processor = UFCDataProcessor(data_path='data/raw/')
    
    try:
        processor.load_raw_data()
        print("✓ Data loading successful!")
        print(f"✓ Loaded {len(processor.event_details)} events")
        print(f"✓ Loaded {len(processor.fighter_details)} fighters") 
        print(f"✓ Loaded {len(processor.fight_details)} fights")
    except Exception as e:
        print(f"✗ Error: {e}")

    try:
        fighter_data = processor.prepare_fight_level_data()
        print(f"✓ Prepared fighter-level data with {len(fighter_data)} records.")
        print(fighter_data.head())
    except Exception as e:
        print(f"✗ Error preparing fighter-level data: {e}")

# Run quick test if this script is executed directly
if __name__ == "__main__":
    quick_test()


