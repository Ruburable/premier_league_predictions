#!/usr/bin/env python3
"""
predict_scores.py

Trains model on historical data and predicts scores for upcoming fixtures.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# ---------------- CONFIG ---------------- #

HISTORICAL_DATA = Path("data/matches_master.csv")
UPCOMING_DATA = Path("data/upcoming_fixtures.csv")
OUT = Path("output/predictions_upcoming.csv")

RANDOM_STATE = 42

FEATURES = [
    "home_xg",
    "away_xg",
    "xg_diff",
    "home_adv",
]

TARGET_HOME = "home_goals"
TARGET_AWAY = "away_goals"


# ---------------------------------------- #


def load_historical_data():
    """Load historical matches for training."""
    if not HISTORICAL_DATA.exists():
        raise FileNotFoundError(
            f"Historical data not found at {HISTORICAL_DATA}\n"
            "Please run update_data.py first!"
        )

    df = pd.read_csv(HISTORICAL_DATA, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Drop any rows without goals (unplayed matches)
    df = df.dropna(subset=[TARGET_HOME, TARGET_AWAY])

    print(f"Loaded {len(df)} historical matches for training")
    if not df.empty:
        print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    return df


def load_upcoming_fixtures():
    """Load upcoming fixtures to predict."""
    if not UPCOMING_DATA.exists():
        raise FileNotFoundError(
            f"Upcoming fixtures not found at {UPCOMING_DATA}\n"
            "Please run update_data.py first!"
        )

    df = pd.read_csv(UPCOMING_DATA, parse_dates=["datetime"])

    print(f"\n📥 Raw upcoming fixtures file:")
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    if df.empty:
        print("\n⚠️  WARNING: Upcoming fixtures file is EMPTY!")
        print("  This means no future matches were found.")
        print("  Possible reasons:")
        print("    - The season has ended")
        print("    - All matches have been played")
        print("    - There's a break in fixtures")
        return df

    df = df.sort_values("datetime").reset_index(drop=True)

    print(f"\n📋 Upcoming fixtures details:")
    print(f"  Number of fixtures: {len(df)}")
    if not df.empty:
        print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"\n  First 5 fixtures:")
        for idx, row in df.head(5).iterrows():
            date_str = row['datetime'].strftime("%Y-%m-%d %H:%M") if pd.notna(row['datetime']) else "TBD"
            print(f"    {date_str} | {row['home_team']} vs {row['away_team']}")

    return df


def add_features(df):
    """Add engineered features."""
    df = df.copy()
    df["xg_diff"] = df["home_xg"] - df["away_xg"]
    df["home_adv"] = 1.0
    return df


def train_model(X, y, model_type="xgb"):
    """Train prediction model."""
    if model_type == "xgb":
        model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=4,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X, y)
    return model


def calculate_win_probabilities(pred_home, pred_away, margin=0.5):
    """
    Calculate win probabilities based on predicted scores.
    Uses a margin to determine draws.
    """
    diff = pred_home - pred_away

    # Simple probability model based on score difference
    if diff > margin:
        prob_home = 0.6 + min(0.3, (diff - margin) * 0.2)
        prob_draw = max(0.1, 0.3 - (diff - margin) * 0.15)
        prob_away = 1 - prob_home - prob_draw
    elif diff < -margin:
        prob_away = 0.6 + min(0.3, (-diff - margin) * 0.2)
        prob_draw = max(0.1, 0.3 - (-diff - margin) * 0.15)
        prob_home = 1 - prob_away - prob_draw
    else:
        # Close match - higher draw probability
        prob_draw = 0.35
        prob_home = 0.33
        prob_away = 0.32

    return prob_home, prob_draw, prob_away


def main():
    print("=" * 80)
    print("MATCH PREDICTION PIPELINE")
    print("=" * 80)

    # Load data
    print("\n1. Loading data...")
    historical = load_historical_data()
    upcoming = load_upcoming_fixtures()

    if upcoming.empty:
        print("\n" + "=" * 80)
        print("❌ NO UPCOMING FIXTURES TO PREDICT")
        print("=" * 80)
        print("\nCreating empty predictions file...")

        # Create empty output file
        empty_df = pd.DataFrame(columns=[
            "datetime", "home_team", "away_team",
            "pred_home_goals", "pred_away_goals",
            "prob_home", "prob_draw", "prob_away"
        ])
        OUT.parent.mkdir(parents=True, exist_ok=True)
        empty_df.to_csv(OUT, index=False)

        print(f"✓ Saved empty predictions to {OUT}")
        print("\nRun update_data.py to refresh fixture data.")
        return

    # Add features
    print("\n2. Engineering features...")
    historical = add_features(historical)
    upcoming = add_features(upcoming)

    # Prepare training data
    X_train = historical[FEATURES]
    y_home = historical[TARGET_HOME]
    y_away = historical[TARGET_AWAY]

    # Sanity check
    print("\n3. Training data sanity check:")
    print(historical[[TARGET_HOME, TARGET_AWAY]].describe())

    # Train models
    print("\n4. Training models...")
    model_home = train_model(X_train, y_home)
    model_away = train_model(X_train, y_away)

    # Evaluate on training data
    pred_home_train = model_home.predict(X_train)
    pred_away_train = model_away.predict(X_train)

    print("\nModel Performance (on training data):")
    print(f"  Home goals MAE: {mean_absolute_error(y_home, pred_home_train):.3f}")
    print(f"  Away goals MAE: {mean_absolute_error(y_away, pred_away_train):.3f}")

    # Predict upcoming matches
    print("\n5. Predicting upcoming fixtures...")
    X_upcoming = upcoming[FEATURES]

    upcoming["pred_home_goals"] = model_home.predict(X_upcoming)
    upcoming["pred_away_goals"] = model_away.predict(X_upcoming)

    # Calculate win probabilities
    probs = [
        calculate_win_probabilities(h, a)
        for h, a in zip(upcoming["pred_home_goals"], upcoming["pred_away_goals"])
    ]

    upcoming["prob_home"] = [p[0] for p in probs]
    upcoming["prob_draw"] = [p[1] for p in probs]
    upcoming["prob_away"] = [p[2] for p in probs]

    # Display predictions
    print("\n" + "=" * 80)
    print("PREDICTIONS FOR UPCOMING FIXTURES")
    print("=" * 80)
    for _, row in upcoming.iterrows():
        date_str = row['datetime'].strftime("%a %Y-%m-%d %H:%M") if pd.notna(row['datetime']) else "TBD"
        print(f"\n{date_str}")
        print(
            f"{row['home_team']:25s} {row['pred_home_goals']:4.2f} - {row['pred_away_goals']:4.2f}  {row['away_team']:25s}")
        print(
            f"  Probabilities: H {row['prob_home'] * 100:5.1f}% | D {row['prob_draw'] * 100:5.1f}% | A {row['prob_away'] * 100:5.1f}%")
    print("=" * 80)

    # Export
    output = upcoming[[
        "datetime",
        "home_team",
        "away_team",
        "pred_home_goals",
        "pred_away_goals",
        "prob_home",
        "prob_draw",
        "prob_away",
    ]]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUT, index=False)

    print(f"\n✅ Saved {len(output)} predictions to {OUT}")
    print("\nNext step: Run python visualise.py to create dashboard")


if __name__ == "__main__":
    main()