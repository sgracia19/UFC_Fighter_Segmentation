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
│   └── raw/                    # Raw CSV files
├── notebooks/
│   └── 01_Data_Processing.ipynb  # Data cleaning, merging, and aggregation
├── src/
│   └── data/
│       └── data_loader.py      # (in progress) Data loading utilities
├── requirements.txt
└── README.md
```

## Data Processing Pipeline (Notebook 01)

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