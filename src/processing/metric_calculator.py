"""
UFC Fighter Metrics Calculator
Handles all metric creation including offensive, defensive, and derived statistics.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class UFCMetricCalculator:
    """
    Calculates comprehensive metrics for UFC fighters including:
    - Complementary defensive metrics
    - Performance ratios and differentials  
    - Derived efficiency measures
    """
    
    def __init__(self):
        pass
    
    def add_complementary_metrics(self, fights_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add complementary defensive metrics to fight-level data.
                
        Parameters:
        fights_df (pd.DataFrame): Fight data with red/blue corners
        
        Returns:
        pd.DataFrame: Enhanced fight data with complementary metrics
        """
        logger.info("Adding complementary defensive metrics...")
        
        df = fights_df.copy()
        
        # RED corner - what they absorbed from BLUE corner
        df['r_strikes_absorbed'] = df['b_sig_stlanded']
        df['r_total_strikes_absorbed'] = df['b_total_stlanded']
        df['r_takedowns_allowed'] = df['b_td_landed']
        df['r_time_controlled'] = df['b_ctrl']
        df['r_knockdowns_absorbed'] = df['b_kd']
        
        # Strike absorption by location
        df['r_head_absorbed'] = df['b_head_landed']
        df['r_body_absorbed'] = df['b_body_landed']
        df['r_leg_absorbed'] = df['b_leg_landed']
        
        # Strike absorption by position
        df['r_dist_absorbed'] = df['b_dist_landed']
        df['r_clinch_absorbed'] = df['b_clinch_landed']
        df['r_ground_absorbed'] = df['b_ground_landed']
        
        # Takedown defense
        df['r_takedowns_defended'] = df['b_td_atmpted'] - df['b_td_landed']
        df['r_subs_defended'] = df['b_suatt']
        
        # BLUE corner - what they absorbed from RED corner  
        df['b_strikes_absorbed'] = df['r_sig_stlanded']
        df['b_total_strikes_absorbed'] = df['r_total_stlanded']
        df['b_takedowns_allowed'] = df['r_td_landed']
        df['b_time_controlled'] = df['r_ctrl']
        df['b_knockdowns_absorbed'] = df['r_kd']
        
        # Strike absorption by location
        df['b_head_absorbed'] = df['r_head_landed']
        df['b_body_absorbed'] = df['r_body_landed']
        df['b_leg_absorbed'] = df['r_leg_landed']
        
        # Strike absorption by position
        df['b_dist_absorbed'] = df['r_dist_landed']
        df['b_clinch_absorbed'] = df['r_clinch_landed']
        df['b_ground_absorbed'] = df['r_ground_landed']
        
        # Takedown defense
        df['b_takedowns_defended'] = df['r_td_atmpted'] - df['r_td_landed']
        df['b_subs_defended'] = df['r_suatt']
        
        # Performance differentials
        df['r_ctrl_differential'] = df['r_ctrl'] - df['b_ctrl']
        df['b_ctrl_differential'] = df['b_ctrl'] - df['r_ctrl']
        df['r_strike_differential'] = df['r_sig_stlanded'] - df['b_sig_stlanded']
        df['b_strike_differential'] = df['b_sig_stlanded'] - df['r_sig_stlanded']
        
        # Opponent quality indicators
        df['r_opp_sig_strikes_landed'] = df['b_sig_stlanded']
        df['r_opp_takedowns_landed'] = df['b_td_landed']
        df['b_opp_sig_strikes_landed'] = df['r_sig_stlanded']
        df['b_opp_takedowns_landed'] = df['r_td_landed']
        
        logger.info("Added complementary metrics to fight-level data")
        return df
    
    def get_enhanced_aggregation_dict(self) -> dict:
        """
        Return the enhanced aggregation dictionary including defensive metrics.
        Use this in your data_processor aggregation step.
        """
        return {
            # Basic fight information
            'fight_id': 'count',
            'win': 'sum',
            'loss': 'sum',
            
            # Time-based metrics
            'fight_time_sec': 'sum',
            'finish_round': 'sum',
            
            # OFFENSIVE STATS
            'kd': 'sum',
            'sig_stlanded': 'sum',
            'sig_statmpted': 'sum',
            'total_stlanded': 'sum',
            'total_statmpted': 'sum',
            'td_landed': 'sum',
            'td_atmpted': 'sum',
            'ctrl': 'sum',
            'suatt': 'sum',
            
            # Strike locations (offensive)
            'head_landed': 'sum', 'head_atmpted': 'sum',
            'body_landed': 'sum', 'body_atmpted': 'sum',
            'leg_landed': 'sum', 'leg_atmpted': 'sum',
            
            # Strike positions (offensive)
            'dist_landed': 'sum', 'dist_atmpted': 'sum',
            'clinch_landed': 'sum', 'clinch_atmpted': 'sum',
            'ground_landed': 'sum', 'ground_atmpted': 'sum',
            
            # DEFENSIVE STATS  
            'strikes_absorbed': 'sum',
            'total_strikes_absorbed': 'sum',
            'takedowns_allowed': 'sum',
            'time_controlled': 'sum',
            'knockdowns_absorbed': 'sum',
            'takedowns_defended': 'sum',
            'subs_defended': 'sum',
            
            # Strike absorption by location
            'head_absorbed': 'sum', 'body_absorbed': 'sum', 'leg_absorbed': 'sum',
            
            # Strike absorption by position  
            'dist_absorbed': 'sum', 'clinch_absorbed': 'sum', 'ground_absorbed': 'sum',
            
            # Performance differentials
            'ctrl_differential': 'sum',
            'strike_differential': 'sum',
            
            # Opponent quality
            'opp_sig_strikes_landed': 'sum',
            'opp_takedowns_landed': 'sum',
            
            # Fighter metadata
            'name': 'first',
            'division': 'last',
        }
    
    def calculate_defensive_metrics(self, fighter_stats: pd.DataFrame) -> pd.DataFrame:
        """Calculate all defensive and efficiency metrics."""
        logger.info("Calculating defensive efficiency metrics...")
        
        # Takedown defense percentage
        total_tds_faced = fighter_stats['takedowns_allowed'] + fighter_stats['takedowns_defended']
        fighter_stats['career_td_defense'] = np.where(
            total_tds_faced > 0,
            (fighter_stats['takedowns_defended'] / total_tds_faced) * 100,
            100  # If never faced takedowns, perfect defense
        )
        
        # Strike absorption rates
        fighter_stats['strikes_absorbed_per_min'] = np.where(
            fighter_stats['total_fight_time_sec'] > 0,
            (fighter_stats['strikes_absorbed'] / fighter_stats['total_fight_time_sec']) * 60,
            0
        )
        
        # Knockdown vulnerability
        fighter_stats['knockdowns_per_fight'] = (
            fighter_stats['knockdowns_absorbed'] / fighter_stats['total_UFC_fights']
        )
        
        # Control resistance
        fighter_stats['time_controlled_per_fight'] = (
            fighter_stats['time_controlled'] / fighter_stats['total_UFC_fights']
        )
        
        return fighter_stats
    
    def calculate_efficiency_ratios(self, fighter_stats: pd.DataFrame) -> pd.DataFrame:
        """Calculate performance efficiency ratios."""
        logger.info("Calculating efficiency ratios...")
        
        # Strike efficiency (output vs absorption)
        fighter_stats['strike_efficiency_ratio'] = np.where(
            fighter_stats['strikes_absorbed'] > 0,
            fighter_stats['sig_stlanded'] / fighter_stats['strikes_absorbed'],
            fighter_stats['sig_stlanded']
        )
        
        # Control dominance (control given vs taken)
        fighter_stats['control_dominance_ratio'] = np.where(
            fighter_stats['time_controlled'] > 0,
            fighter_stats['ctrl'] / fighter_stats['time_controlled'],
            10  # Cap at high value if never controlled
        )
        
        # Takedown dominance
        fighter_stats['takedown_dominance_ratio'] = np.where(
            fighter_stats['takedowns_allowed'] > 0,
            fighter_stats['td_landed'] / fighter_stats['takedowns_allowed'],
            fighter_stats['td_landed']
        )
        
        # Overall performance differential per minute
        fighter_stats['performance_differential_per_min'] = np.where(
            fighter_stats['total_fight_time_sec'] > 0,
            (fighter_stats['strike_differential'] + fighter_stats['ctrl_differential']) / 
            (fighter_stats['total_fight_time_sec'] / 60),
            0
        )
        
        return fighter_stats
    
    def calculate_absorption_percentages(self, fighter_stats: pd.DataFrame) -> pd.DataFrame:
        """Calculate where fighters absorb damage."""
        logger.info("Calculating strike absorption patterns...")
        
        total_absorbed = (
            fighter_stats['head_absorbed'] + 
            fighter_stats['body_absorbed'] + 
            fighter_stats['leg_absorbed']
        )
        
        # Where they get hit
        for location in ['head', 'body', 'leg']:
            fighter_stats[f'{location}_absorbed_percentage'] = np.where(
                total_absorbed > 0,
                (fighter_stats[f'{location}_absorbed'] / total_absorbed) * 100,
                0
            )
        
        # Position-based absorption
        total_position_absorbed = (
            fighter_stats['dist_absorbed'] +
            fighter_stats['clinch_absorbed'] + 
            fighter_stats['ground_absorbed']
        )
        
        for position in ['dist', 'clinch', 'ground']:
            fighter_stats[f'{position}_absorbed_percentage'] = np.where(
                total_position_absorbed > 0,
                (fighter_stats[f'{position}_absorbed'] / total_position_absorbed) * 100,
                0
            )
        
        return fighter_stats
    
    def calculate_all_metrics(self, fighter_stats: pd.DataFrame) -> pd.DataFrame:
        """Run all metric calculations in sequence."""
        logger.info("Calculating all enhanced metrics...")
        
        fighter_stats = self.calculate_defensive_metrics(fighter_stats)
        fighter_stats = self.calculate_efficiency_ratios(fighter_stats)
        fighter_stats = self.calculate_absorption_percentages(fighter_stats)
        
        logger.info("All metric calculations complete")
        return fighter_stats