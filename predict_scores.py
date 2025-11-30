#!/usr/bin/env python3
"""
predict_scores.py

Trains XGBoost regressors on past Premier League matches to predict
upcoming fixtures, including expected goals and top scorers.

Inputs:
  - output/matches_master.csv

Outputs:
  - output/predictions_upcoming.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from xgboost import XGBRegressor

MASTER_CSV = Path("output/matches_master.csv")
OUTPUT_PRED = Path("output/predictions_upcoming.csv")

def load_master():
    if not MASTER_CSV.exists():
        raise FileNotFoundError(f"Master file not found: {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV)
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True)
    return df

def split_past_upcoming(master_df):
    now = pd.Timestamp.utcnow()
    cond_past = (master_df['home_goals'].notna() & master_df['away_goals'].notna()) | \
                (master_df['datetime'].notna() & (master_df['datetime'] <= now))
    past_df = master_df[cond_past].copy()
    upcoming_df = master_df[~cond_past].copy()
    print(f"Loaded {len(master_df)} matches total.")
    print(f"Past matches: {len(past_df)}")
    print(f"Upcoming fixtures: {len(upcoming_df)}")
    return past_df, upcoming_df

def train_xgb_models(past_df):
    # Drop rows with NaN in goals
    past_df = past_df.dropna(subset=['home_goals', 'away_goals']).copy()

    # Encode teams as integers
    teams = pd.concat([past_df['home_team'], past_df['away_team']]).unique()
    team_map = {t: i for i, t in enumerate(teams)}

    past_df['home_id'] = past_df['home_team'].map(team_map)
    past_df['away_id'] = past_df['away_team'].map(team_map)

    # Features: home_id, away_id
    X_home = past_df[['home_id', 'away_id']]
    y_home = past_df['home_goals']

    X_away = past_df[['away_id', 'home_id']]
    y_away = past_df['away_goals']

    model_home = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05)
    model_home.fit(X_home, y_home)

    model_away = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05)
    model_away.fit(X_away, y_away)

    return model_home, model_away, team_map

def predict_scores(upcoming_df, model_home, model_away, team_map):
    # Map teams to IDs; unseen teams get -1
    upcoming_df['home_id'] = upcoming_df['home_team'].map(lambda t: team_map.get(t, -1))
    upcoming_df['away_id'] = upcoming_df['away_team'].map(lambda t: team_map.get(t, -1))

    # Replace -1 (unseen teams) with median of known IDs
    known_ids = [v for v in team_map.values() if pd.notna(v)]
    if known_ids:
        median_id = int(np.median(known_ids))
    else:
        median_id = 0  # fallback if team_map is empty

    upcoming_df['home_id'] = upcoming_df['home_id'].replace(-1, median_id)
    upcoming_df['away_id'] = upcoming_df['away_id'].replace(-1, median_id)

    X_home = upcoming_df[['home_id', 'away_id']]
    X_away = upcoming_df[['away_id', 'home_id']]

    upcoming_df['pred_home_goals'] = model_home.predict(X_home)
    upcoming_df['pred_away_goals'] = model_away.predict(X_away)

    # Clip negative predictions
    upcoming_df['pred_home_goals'] = upcoming_df['pred_home_goals'].clip(0)
    upcoming_df['pred_away_goals'] = upcoming_df['pred_away_goals'].clip(0)

    return upcoming_df

def predict_top_scorers(upcoming_df, past_df, top_n=3):
    # Simple historical scorer frequency
    scorer_dict = {}
    for team in past_df['home_team'].unique():
        scorers = past_df[past_df['home_team'] == team].get('home_scorers', pd.Series()).dropna().tolist()
        scorer_dict[team] = list(set(scorers))[:top_n]

    for team in past_df['away_team'].unique():
        scorers = past_df[past_df['away_team'] == team].get('away_scorers', pd.Series()).dropna().tolist()
        if team in scorer_dict:
            scorer_dict[team] += [s for s in scorers if s not in scorer_dict[team]]
        else:
            scorer_dict[team] = list(set(scorers))[:top_n]

    upcoming_df['top_home_scorers'] = upcoming_df['home_team'].map(lambda t: scorer_dict.get(t, []))
    upcoming_df['top_away_scorers'] = upcoming_df['away_team'].map(lambda t: scorer_dict.get(t, []))
    return upcoming_df

def main():
    master_df = load_master()
    past_df, upcoming_df = split_past_upcoming(master_df)

    if len(past_df) < 5:
        print("⚠ Not enough past data. Using heuristic averages.")
        upcoming_df['pred_home_goals'] = np.random.uniform(0.8, 2.2, len(upcoming_df))
        upcoming_df['pred_away_goals'] = np.random.uniform(0.8, 2.2, len(upcoming_df))
        upcoming_df['top_home_scorers'] = [[] for _ in range(len(upcoming_df))]
        upcoming_df['top_away_scorers'] = [[] for _ in range(len(upcoming_df))]
    else:
        print("--- Training XGBoost models on past matches ---")
        model_home, model_away, team_map = train_xgb_models(past_df)
        print("--- Predicting Upcoming Fixtures ---")
        upcoming_df = predict_scores(upcoming_df, model_home, model_away, team_map)
        upcoming_df = predict_top_scorers(upcoming_df, past_df, top_n=3)

    upcoming_df.to_csv(OUTPUT_PRED, index=False)
    print(f"✔ Saved upcoming predictions → {OUTPUT_PRED}")

if __name__ == "__main__":
    main()
