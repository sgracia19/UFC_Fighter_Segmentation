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
        logger.info("Adding complementary fight metrics...")
        
        df = fights_df.copy()
        
        # RED corner - what they absorbed from BLUE corner
        df['r_sig_str_absorbed'] = df['b_sig_str_landed']
        df['r_total_str_absorbed'] = df['b_total_str_landed']
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
        df['r_subs_defended'] = df['b_sub_att']
        
        # BLUE corner - what they absorbed from RED corner  
        df['b_sig_str_absorbed'] = df['r_sig_str_landed']
        df['b_total_str_absorbed'] = df['r_total_str_landed']
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
        df['b_subs_defended'] = df['r_sub_att']
        
        # Performance differentials
        df['r_ctrl_differential'] = df['r_ctrl'] - df['b_ctrl']
        df['b_ctrl_differential'] = df['b_ctrl'] - df['r_ctrl']
        df['r_str_differential'] = df['r_sig_str_landed'] - df['b_sig_str_landed']
        df['b_str_differential'] = df['b_sig_str_landed'] - df['r_sig_str_landed']
        
        # Opponent quality indicators
        df['r_opp_sig_str_landed'] = df['b_sig_str_landed']
        df['r_opp_takedowns_landed'] = df['b_td_landed']
        df['b_opp_sig_str_landed'] = df['r_sig_str_landed']
        df['b_opp_takedowns_landed'] = df['r_td_landed']
        
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
            'date': 'last',
            
            # Time-based metrics
            'fight_time_sec': 'sum',
            'finish_round': 'sum',
            
            # OFFENSIVE STATS
            'kd': 'sum',
            'sig_str_landed': 'sum',
            'sig_str_atmpted': 'sum',
            'total_str_landed': 'sum',
            'total_str_atmpted': 'sum',
            'td_landed': 'sum',
            'td_atmpted': 'sum',
            'ctrl': 'sum',
            'sub_att': 'sum',
            
            # Strike locations (offensive)
            'head_landed': 'sum', 'head_atmpted': 'sum',
            'body_landed': 'sum', 'body_atmpted': 'sum',
            'leg_landed': 'sum', 'leg_atmpted': 'sum',
            
            # Strike positions (offensive)
            'dist_landed': 'sum', 'dist_atmpted': 'sum',
            'clinch_landed': 'sum', 'clinch_atmpted': 'sum',
            'ground_landed': 'sum', 'ground_atmpted': 'sum',
            
            # DEFENSIVE STATS  
            'sig_str_absorbed': 'sum',
            'total_str_absorbed': 'sum',
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
            'str_differential': 'sum',
            
            # Opponent quality
            'opp_sig_str_landed': 'sum',
            'opp_takedowns_landed': 'sum',
            
            # Fighter metadata
            'name': 'first',
            'division': 'last',
        }
    
    def calculate_derived_metrics(self, fighter_stats: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived metrics from aggregated stats."""
        logger.info("Calculating derived metrics...")

        initial_len = len(fighter_stats.columns)

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

        # Per-minute metrics for standardization of metrics
        per_min_metrics = [
            ('str_landed_per_min', 'total_str_landed'),
            ('td_per_min', 'td_landed'), 
            ('td_att_per_min', 'td_atmpted'),                  
            ('kd_per_min', 'kd'),                           
            ('ctrl_per_min', 'ctrl'),                       
            ('sig_str_absorbed_per_min', 'sig_str_absorbed'),
            ('times_knockedown_per_min', 'knockdowns_absorbed'),
            ('time_controlled_per_min', 'time_controlled')
        ]

        for per_min_name, metric in per_min_metrics:
            fighter_stats[per_min_name] = np.where(
                fighter_stats['total_fight_time_sec'] >0,
                (fighter_stats[metric] / fighter_stats['total_fight_time_sec']) *60,
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
        
        # Takedown defense percentage
        total_tds_faced = fighter_stats['takedowns_allowed'] + fighter_stats['takedowns_defended']
        fighter_stats['career_td_defense'] = np.where(
            total_tds_faced > 0,
            (fighter_stats['takedowns_defended'] / total_tds_faced) * 100,
            100  # If never faced takedowns, perfect defense
        )

        # Strike distribution
        total_strikes_landed = (
            fighter_stats['head_landed'] + 
            fighter_stats['body_landed'] + 
            fighter_stats['leg_landed']
        )
        
        for location in ['head', 'body', 'leg']:
            fighter_stats[f'{location}_str_percentage'] = np.where(
                total_strikes_landed > 0,
                (fighter_stats[f'{location}_landed'] / total_strikes_landed) * 100,
                0
            )

        # Strike efficiency (output vs absorption)
        fighter_stats['str_efficiency_ratio'] = np.where(
            fighter_stats['sig_str_absorbed'] > 0,
            fighter_stats['sig_str_landed'] / fighter_stats['sig_str_absorbed'],
            fighter_stats['sig_str_landed']
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
            (fighter_stats['str_differential'] + fighter_stats['ctrl_differential']) / 
            (fighter_stats['total_fight_time_sec'] / 60),
            0
        )

        # Knockdown differential (given vs received)
        fighter_stats['knockdown_differential'] = fighter_stats['kd'] - fighter_stats['knockdowns_absorbed']

        # "Granite chin" metric - ability to avoid knockdowns relative to strikes absorbed
        fighter_stats['chin_durability'] = np.where(
            fighter_stats['sig_str_absorbed'] > 0,
            1 - (fighter_stats['knockdowns_absorbed'] / (fighter_stats['sig_str_absorbed'] / 100)),
            1
        )

        # Absorbtion Rates
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


        # Style and versatility 

        # Fighting range preference (distance vs clinch vs ground)
        total_strikes = fighter_stats['dist_landed'] + fighter_stats['clinch_landed'] + fighter_stats['ground_landed']

        # Calculate range diversity (higher = more well-rounded)
        range_proportions = []
        for pos in ['dist', 'clinch', 'ground']:
            prop = fighter_stats[f'{pos}_landed'] / total_strikes
            range_proportions.append(prop)

        # Shannon entropy for style diversity (0 = one-dimensional, higher = more diverse)
        fighter_stats['style_diversity_index'] = -sum(p * np.log(p + 1e-10) for p in range_proportions)

        # Submission threat level
        fighter_stats['submission_threat'] = np.where(
            fighter_stats['ground_landed'] + fighter_stats['td_landed'] > 0,
            fighter_stats['sub_att'] / (fighter_stats['ground_landed'] + fighter_stats['td_landed']),
            0
        )

        # "Power" efficiency - knockdowns per significant strike
        fighter_stats['power_efficiency'] = np.where(
            fighter_stats['sig_str_landed'] > 0,
            fighter_stats['kd'] / fighter_stats['sig_str_landed'],
            0
        )

        # Control fighter variable
        fighter_stats['ctr_time_per_td_atmpt'] = np.where(
            fighter_stats['td_atmpted'] > 0,
            fighter_stats['ctrl'] / fighter_stats['td_atmpted'],  # Control time per takedown attempt
            0
        )

        # Submission efficiency
        fighter_stats['submission_conversion_rate'] = np.where(
            fighter_stats['sub_att'] > 0,
            fighter_stats['wins_by_submission'] / fighter_stats['sub_att'] * 100,
            0
        )

        new_len = len(fighter_stats.columns)

        logger.info(f"Added {new_len - initial_len} new fight metric columns")
        return fighter_stats
    
    def calculate_all_metrics(self, fighter_stats: pd.DataFrame) -> pd.DataFrame:
        """Run metric calculations """        
        fighter_stats = self.calculate_derived_metrics(fighter_stats)
        
        logger.info("All metric calculations complete")
        return fighter_stats