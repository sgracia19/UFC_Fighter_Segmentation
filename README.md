# UFC Fighter Segmentation

A data science project to analyze UFC fighter performance data and develop meaningful fighter archetypes through clustering.

## Overview

This project scrapes, processes, and analyzes UFC fight statistics to segment fighters into meaningful groups based on their fighting styles and performance metrics. The goal is to identify distinct fighter archetypes (e.g., knockout artists, wrestlers, decision fighters) using unsupervised machine learning.

## Data Sources

Three raw datasets from the [UFC Datasets 1994–2025](https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025) Kaggle dataset, originally sourced from the UFC stats website:

- `event_details.csv` — Fight outcomes by event (winner, location, date)
- `fighter_details.csv` — Fighter biographical and career summary stats (height, weight, reach, stance, win/loss record, etc.)
- `fight_details.csv` — Per-fight statistics for both corners (strikes, takedowns, control time, accuracy, etc.)

## Project Structure

```
├── data/
│   ├── raw/                    # Raw CSV files from Kaggle
│   │   ├── event_details.csv
│   │   ├── fighter_details.csv
│   │   └── fight_details.csv
│   └── processed/              # Cleaned and aggregated datasets
│       ├── ufc_fighters_clean.csv
│       └── ufc_fighters_processed.csv
├── notebooks/
│   ├── 01_Data_Processing.ipynb    # Data cleaning, merging, aggregation
│   ├── 02_Data_Exploration.ipynb   # Feature analysis and preprocessing
│   ├── 03_Clustering.ipynb         # PCA + K-Means clustering analysis
│   └── 04_Visualizations.ipynb     # Cluster visualization and insights
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py            # Main data processing pipeline
│   └── processing/                 # Modular processing components
│       ├── __init__.py
│       ├── data_cleaner.py
│       ├── data_processor.py
│       ├── metric_calculator.py
│       └── pre_processor.py
├── requirements.txt                # Python dependencies
└── README.md
```

## Data Processing Pipeline 

The processing notebook transforms raw fight-level data into one row per fighter:

1. **Merging** — Joins `fight_details` with `event_details` to attach fight outcomes (winner, date) to each fight row.
2. **Corner splitting** — Separates red corner and blue corner into a unified fighter-level format with standardized column names.
3. **Win/loss labeling** — Adds binary win/loss columns per fighter per fight based on matched IDs.
4. **Aggregation** — Rolls up fight-level data to career totals per fighter, including:
   - Strike totals and accuracy (head, body, leg, distance, clinch, ground)
   - Takedown totals and accuracy
   - Submission attempts, knockdowns, control time
   - Win/loss breakdowns by method (KO/TKO, Submission, Decision, etc.)
5. **Derived features** — Computes per-minute rates, finish rates, win percentage, and strike distribution percentages.
6. **Fighter details join** — Merges career stats with biographical data (height, weight, reach, stance, DOB).

> **Note:** UFC fight totals differ from fighter summary records because the dataset only includes UFC fights, while fighter profiles reflect full career records.

## Data Exploration 

The exploration notebook analyzes the processed fighter data to understand distributions, identify patterns, and prepare features for clustering:

### Key Preprocessing Decisions
- **Weight class consolidation**: Reduced 89 raw weight classes (including tournament-specific divisions) into 8 standard UFC weight classes
- **Stance normalization**: Consolidated rare stances (Open Stance, Sideways) into Orthodox for analysis
- **Feature filtering**: Focused on active fighters with meaningful UFC experience

### Core Features for Clustering
The analysis identified 22 key performance metrics across three categories:

**Striking Metrics:**
- Strike accuracy, landed strikes per minute, power efficiency
- KO/TKO win rate, strike efficiency ratio, strike attempts per minute
- Strike distribution (head/body/leg percentages)

**Grappling Metrics:**
- Takedown accuracy, takedowns per minute, submission win rate
- Takedown defense, submission vulnerability

**Physical & Style Metrics:**
- Height and reach (z-scores), style diversity index
- Finish rate, defensive vulnerabilities (knockdowns, absorbed strikes)
- Takedown dominance ratio, chin durability

### Data Quality Insights
- Dataset contains 2,591 fighters with complete UFC records
- Features span offensive capabilities, defensive resilience, and fighting style preferences
- Robust scaling applied to handle outliers in performance metrics

## Clustering Methodology (Notebook 03)

The clustering analysis uses unsupervised machine learning to identify fighter archetypes based on fighting style and performance patterns.

### Feature Engineering
- **22 core features** combined from striking offense/defense, grappling capabilities, and physical attributes
- **Robust scaling** applied to normalize features and reduce outlier influence
- **Missing value imputation** using weight-class-specific medians

### Dimensionality Reduction
- **Principal Component Analysis (PCA)** reduces 22 features to 8 components
- Captures ~80% of total variance while removing noise and multicollinearity
- First two components primarily represent striking vs. grappling preferences

### Clustering Algorithm
- **K-Means clustering** selected after evaluating silhouette scores across k=2-10 clusters
- **Optimal k=4** chosen based on highest silhouette score (0.237)
- Algorithm identifies natural groupings in the reduced feature space

### Validation Approach
- Silhouette analysis confirms cluster quality and separation
- Cross-validation ensures stable cluster assignments
### Validation Approach
- Silhouette analysis confirms cluster quality and separation
- Cross-validation ensures stable cluster assignments
- Visual inspection of PCA/t-SNE projections validates grouping patterns

## Results 

The clustering analysis identified 4 distinct fighter archetypes based on fighting style and performance patterns:

To be finished

## Visualizations 
The visualization notebook provides multiple views of the clustering results:

### PCA Scatter Plot
- **2D projection** of fighters in principal component space
- **First component** (horizontal axis) separates grappling vs. striking preferences
- **Second component** (vertical axis) captures physical attributes and defensive capabilities
- **Color-coded clusters** show clear separation between archetypes

### t-SNE Embedding
- **Non-linear dimensionality reduction** preserving local relationships
- **Alternative visualization** to PCA for complex pattern detection
- **Cluster boundaries** appear more distinct in t-SNE space
- **Perplexity=30** optimized for the dataset size

### Weight Class Distribution by Cluster
- **Stacked bar chart** showing archetype prevalence across weight classes
- **Reveals fighting style preferences** by division (e.g., grappling specialists in middleweight)
- **Identifies underrepresented combinations** (e.g., power punchers in lightweight)

### Key Visual Insights
- **Physical metrics** (height/reach) create strong cluster separation
- **Style diversity** varies significantly between archetypes
- **Weight class effects** show how fighting styles cluster by division
- **Defensive profiles** clearly differentiate technical strikers from power punchers

## Installation & Usage

### Prerequisites
- Python 3.8+
- Jupyter Notebook/Lab

### Dependencies
Install required packages:
```bash
pip install -r requirements.txt
```

**Key libraries:**
- `pandas`, `numpy` — Data manipulation
- `scikit-learn` — Machine learning (PCA, K-Means, preprocessing)
- `matplotlib`, `seaborn` — Visualization
- `jupyter` — Notebook environment

### Notebook Execution Order
1. **[01_Data_Processing.ipynb](notebooks/01_Data_Processing.ipynb)** — Clean and aggregate raw UFC data
2. **[02_Data_Exploration.ipynb](notebooks/02_Data_Exploration.ipynb)** — Analyze distributions and prepare features
3. **[03_Clustering.ipynb](notebooks/03_Clustering.ipynb)** — Perform PCA and K-Means clustering
4. **[04_Visualizations.ipynb](notebooks/04_Visualizations.ipynb)** — Create and analyze cluster visualizations

### Quick Start
```python
import pandas as pd
from src.data_pipeline import data_pipeline as dp

# Load processed fighter data
processor = dp.UFCDataProcessor('data/raw/')
fighters_df = processor.process_data()

# Access clustering results
print(f"Found {len(fighters_df)} fighters across {fighters_df['cluster'].nunique()} archetypes")
```

### Data Files
- **Raw data:** `data/raw/` (event_details.csv, fighter_details.csv, fight_details.csv)
- **Processed data:** `data/processed/` (ufc_fighters_clean.csv, ufc_fighters_processed.csv)

## Future Work

### Methodology Improvements
- **Hierarchical clustering** for more granular archetype detection
- **Feature engineering** with interaction terms and domain-specific metrics
- **Alternative algorithms** (DBSCAN, Gaussian Mixture Models) for comparison
- **Temporal analysis** incorporating fighter progression over time

### Analysis Extensions
- **Performance prediction** models using fighter archetypes as features
- **Matchup analysis** predicting outcomes based on archetype combinations
- **Career trajectory** analysis within each archetype
- **Style evolution** tracking how archetypes change over time

### Data Enhancements
- **Additional data sources** (strike locations, fight rounds, fighter injuries)
- **Longitudinal tracking** of fighter development and style changes
- **International comparisons** across different MMA promotions
- **Advanced metrics** (strike efficiency by target, defensive reactions)

### Applications
- **Scouting tools** for identifying talent in specific archetypes
- **Training recommendations** based on archetype strengths/weaknesses
- **Matchmaking optimization** for balanced and exciting fights
- **Fan engagement** features showing archetype-based fighter comparisons
- **Advanced metrics** (strike efficiency by target, defensive reactions)

### Applications
- **Scouting tools** for identifying talent in specific archetypes
- **Training recommendations** based on archetype strengths/weaknesses
- **Matchmaking optimization** for balanced and exciting fights
- **Fan engagement** features showing archetype-based fighter comparisons