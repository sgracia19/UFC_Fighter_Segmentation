"""
UFC Fighter Segmentation - Simple Data Pipeline
Simple pipeline for personal project with static Kaggle data.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Optional

from .processing.data_processor import UFCDataProcessor
from .processing.data_cleaner import DataCleaner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_process_ufc_data(raw_data_path: str = None,
                             save_processed: bool = True,
                             processed_data_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple function to load and process UFC data.
    
    Parameters:
    raw_data_path (str): Path to raw CSV files
    save_processed (bool): Whether to save processed data to CSV
    processed_data_path (str): Where to save processed data
    
    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: (raw_processed_data, cleaned_data)
    """
    logger.info("Processing UFC fighter data...")
    
    # Auto-detect paths if not provided
    if raw_data_path is None:
        # Look for data directory from current working directory or script location
        current_dir = Path.cwd()
        script_dir = Path(__file__).parent.parent  # Go up from src/
        
        for base_path in [current_dir, script_dir]:
            potential_raw = base_path / 'data' / 'raw'
            if potential_raw.exists():
                raw_data_path = str(potential_raw) + '/'
                break
        
        if raw_data_path is None:
            raise FileNotFoundError("Could not find data/raw/ directory. Please specify raw_data_path.")
    
    if processed_data_path is None:
        # Use same base directory as raw data
        raw_base = Path(raw_data_path).parent
        processed_data_path = str(raw_base / 'processed')
    
    # Clean up paths for logging (remove personal directory info)
    raw_display = Path(raw_data_path).name if raw_data_path else "auto-detected"
    processed_display = Path(processed_data_path).name if processed_data_path else "auto-detected"
    logger.info(f"Using raw data: {raw_display}")
    logger.info(f"Using processed data: {processed_display}")
    
    # Check if processed data already exists
    processed_path = Path(processed_data_path)
    raw_file = processed_path / 'ufc_fighters_processed.csv'
    clean_file = processed_path / 'ufc_fighters_clean.csv'
    
    if raw_file.exists() and clean_file.exists():
        logger.info("Loading existing processed data...")
        raw_data = pd.read_csv(raw_file)
        clean_data = pd.read_csv(clean_file)
        logger.info(f"Loaded {len(clean_data)} fighters from cache")
        return raw_data, clean_data
    
    # Process data
    logger.info("Processing raw data...")
    processor = UFCDataProcessor(raw_data_path)
    raw_data = processor.process_data()
    
    logger.info("Cleaning data...")
    cleaner = DataCleaner()
    clean_data = cleaner.clean_data(raw_data)
    
    # Save if requested
    if save_processed:
        processed_path.mkdir(parents=True, exist_ok=True)
        raw_data.to_csv(raw_file, index=False)
        clean_data.to_csv(clean_file, index=False)
    
    logger.info(f"Processing complete! {len(clean_data)} fighters ready for analysis")
    return raw_data, clean_data

def get_clean_data(raw_data_path: str = None,
                   processed_data_path: str = None) -> pd.DataFrame:
    """
    Get clean UFC data ready for analysis (most common use case).
    
    Returns:
    pd.DataFrame: Clean fighter data
    """
    _, clean_data = load_and_process_ufc_data(raw_data_path, True, processed_data_path)
    return clean_data

def force_reprocess(raw_data_path: str = None,
                   processed_data_path: str = None) -> pd.DataFrame:
    """
    Force reprocessing (delete cached files and reprocess).
    
    Returns:
    pd.DataFrame: Freshly processed clean data
    """
    # Auto-detect processed path if needed
    if processed_data_path is None:
        # Look for data directory from current working directory or script location
        current_dir = Path.cwd()
        script_dir = Path(__file__).parent.parent
        
        for base_path in [current_dir, script_dir]:
            potential_processed = base_path / 'data' / 'processed'
            if potential_processed.parent.exists():  # data/ directory exists
                processed_data_path = str(potential_processed)
                break
        
        if processed_data_path is None:
            processed_data_path = 'data/processed'  # fallback
    
    # Remove existing processed files
    processed_path = Path(processed_data_path)
    for file in processed_path.glob('ufc_fighters_*.csv'):
        file.unlink()
        
    logger.info("Cleared cached data - reprocessing...")
    return get_clean_data(raw_data_path, processed_data_path)

# Convenience function for quick analysis
def quick_load() -> pd.DataFrame:
    """Quick load for analysis - uses default paths."""
    return get_clean_data()

def debug_paths():
    """Debug function to help find your data files."""
    import os
    
    print("Debugging file paths...")
    print(f"Current working directory: {Path.cwd()}")
    print(f"Script location: {Path(__file__).parent}")
    print(f"Project root (assumed): {Path(__file__).parent.parent}")
    
    # Look for data directories
    search_paths = [
        Path.cwd(),
        Path(__file__).parent.parent,  # project root
        Path.cwd().parent,  # if running from notebooks/
    ]
    
    for base in search_paths:
        data_dir = base / 'data'
        raw_dir = data_dir / 'raw'
        
        print(f"\nChecking {base}:")
        print(f" {data_dir} exists: {data_dir.exists()}")
        print(f"{raw_dir} exists: {raw_dir.exists()}")
        
        if raw_dir.exists():
            csv_files = list(raw_dir.glob('*.csv'))
            print(f"CSV files found: {[f.name for f in csv_files]}")
            
            # Check for the expected files
            expected = ['event_details.csv', 'fighter_details.csv', 'fight_details.csv']
            missing = [f for f in expected if not (raw_dir / f).exists()]
            if missing:
                print(f"Missing files: {missing}")
            else:
                print(f"All expected files found!")
                return str(raw_dir) + '/'
    
    print("\n Could not find data/raw/ directory with required CSV files")
    return None

if __name__ == "__main__":
    # Simple CLI
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--force':
        data = force_reprocess()
    else:
        data = get_clean_data()
    
    print(f"Dataset: {data.shape[0]} fighters, {data.shape[1]} features")
    print(f"Ready for analysis!")