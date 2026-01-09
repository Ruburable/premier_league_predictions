#!/usr/bin/env python3
"""
predict_scores_enhanced.py

Enhanced prediction script that:
1. Makes in-sample predictions for historical matches (current season)
2. Predicts upcoming fixtures
3. Saves both for visualization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timezone

# ---------------- CONFIG ---------------- #

HISTORICAL_DATA = Path("data/matches_master.csv")
UPCOMING_DATA = Path("data/upcoming_fixtures.csv")
OUT_UPCOMING = Path("output/predictions_upcoming.csv")
OUT_HISTORICAL = Path("output/predictions_historical.csv")

RANDOM_STATE = 42


# Determine current season based on current date
# Premier League season runs from August to May
# e.g., 2024-25 season = 2024 (year when season starts)
def get_current_season():
    """
    Determine current Premier League season based on date.
    Season starts in August, so:
    - Jan-July: previous year's season (e.g., Jan 2025 = 2024 season)
    - Aug-Dec: current year's season (e.g., Aug 2024 = 2024 season)
    """
    now = datetime.now()
    if now.month >= 8:  # August or later
        return now.year
    else:  # January to July
        return now.year - 1


CURRENT_SEASON = get_current_season()

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

    print(f"Loaded {len(df)} historical matches")
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

    print(f"\nLoaded {len(df)} upcoming fixtures")
    if not df.empty:
        print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")

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
    """Calculate win probabilities based on predicted scores."""
    diff = pred_home - pred_away

    if diff > margin:
        prob_home = 0.6 + min(0.3, (diff - margin) * 0.2)
        prob_draw = max(0.1, 0.3 - (diff - margin) * 0.15)
        prob_away = 1 - prob_home - prob_draw
    elif diff < -margin:
        prob_away = 0.6 + min(0.3, (-diff - margin) * 0.2)
        prob_draw = max(0.1, 0.3 - (-diff - margin) * 0.15)
        prob_home = 1 - prob_away - prob_draw
    else:
        prob_draw = 0.35
        prob_home = 0.33
        prob_away = 0.32

    return prob_home, prob_draw, prob_away


def get_current_season_matches(df):
    """Filter for current season matches."""
    if df.empty:
        return df

    # Check if season column exists
    if "season" not in df.columns:
        print(f"\n⚠️  Warning: 'season' column not found in data")
        print(f"   Available columns: {list(df.columns)}")
        return pd.DataFrame()

    current_season_df = df[df["season"] == CURRENT_SEASON].copy()
    print(f"\nCurrent season ({CURRENT_SEASON}) has {len(current_season_df)} matches")

    if current_season_df.empty:
        print(f"⚠️  No matches found for season {CURRENT_SEASON}")
        print(f"   Available seasons in data: {sorted(df['season'].unique())}")

    return current_season_df


def main():
    print("=" * 80)
    print("ENHANCED MATCH PREDICTION PIPELINE")
    print("=" * 80)

    # Show current season
    print(f"\n🏆 Current season: {CURRENT_SEASON}/{str(CURRENT_SEASON + 1)[-2:]}")
    print(f"   (Determined from current date: {datetime.now().strftime('%Y-%m-%d')})")

    # Load data
    print("\n1. Loading data...")
    all_historical = load_historical_data()
    upcoming = load_upcoming_fixtures()

    # Add features to all data
    print("\n2. Engineering features...")
    all_historical = add_features(all_historical)
    upcoming = add_features(upcoming) if not upcoming.empty else upcoming

    # Prepare training data (all historical matches)
    X_train = all_historical[FEATURES]
    y_home = all_historical[TARGET_HOME]
    y_away = all_historical[TARGET_AWAY]

    # Sanity check
    print("\n3. Training data sanity check:")
    print(all_historical[[TARGET_HOME, TARGET_AWAY]].describe())

    # Train models
    print("\n4. Training models on all historical data...")
    model_home = train_model(X_train, y_home)
    model_away = train_model(X_train, y_away)

    # Evaluate on training data
    pred_home_train = model_home.predict(X_train)
    pred_away_train = model_away.predict(X_train)

    print("\nModel Performance (on all training data):")
    print(f"  Home goals MAE: {mean_absolute_error(y_home, pred_home_train):.3f}")
    print(f"  Away goals MAE: {mean_absolute_error(y_away, pred_away_train):.3f}")

    # Make in-sample predictions for current season
    print("\n5. Making in-sample predictions for current season...")
    current_season = get_current_season_matches(all_historical)

    if not current_season.empty:
        X_current = current_season[FEATURES]

        current_season["pred_home_goals"] = model_home.predict(X_current)
        current_season["pred_away_goals"] = model_away.predict(X_current)

        # Calculate probabilities for historical matches
        probs = [
            calculate_win_probabilities(h, a)
            for h, a in zip(current_season["pred_home_goals"], current_season["pred_away_goals"])
        ]

        current_season["prob_home"] = [p[0] for p in probs]
        current_season["prob_draw"] = [p[1] for p in probs]
        current_season["prob_away"] = [p[2] for p in probs]

        # Calculate prediction accuracy
        mae_home = mean_absolute_error(
            current_season[TARGET_HOME],
            current_season["pred_home_goals"]
        )
        mae_away = mean_absolute_error(
            current_season[TARGET_AWAY],
            current_season["pred_away_goals"]
        )

        print(f"\nCurrent season prediction accuracy:")
        print(f"  Home goals MAE: {mae_home:.3f}")
        print(f"  Away goals MAE: {mae_away:.3f}")

        # Save historical predictions
        historical_output = current_season[[
            "datetime",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "home_xg",
            "away_xg",
            "pred_home_goals",
            "pred_away_goals",
            "prob_home",
            "prob_draw",
            "prob_away",
        ]]

        OUT_HISTORICAL.parent.mkdir(parents=True, exist_ok=True)
        historical_output.to_csv(OUT_HISTORICAL, index=False)
        print(f"\n✅ Saved {len(historical_output)} historical predictions to {OUT_HISTORICAL}")

    # Predict upcoming matches
    if upcoming.empty:
        print("\n⚠️  No upcoming fixtures to predict")
        empty_df = pd.DataFrame(columns=[
            "datetime", "home_team", "away_team",
            "pred_home_goals", "pred_away_goals",
            "prob_home", "prob_draw", "prob_away"
        ])
        OUT_UPCOMING.parent.mkdir(parents=True, exist_ok=True)
        empty_df.to_csv(OUT_UPCOMING, index=False)
    else:
        print("\n6. Predicting upcoming fixtures...")
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
        for _, row in upcoming.head(10).iterrows():
            date_str = row['datetime'].strftime("%a %Y-%m-%d %H:%M") if pd.notna(row['datetime']) else "TBD"
            print(f"\n{date_str}")
            print(
                f"{row['home_team']:25s} {row['pred_home_goals']:4.2f} - {row['pred_away_goals']:4.2f}  {row['away_team']:25s}")
            print(
                f"  Probabilities: H {row['prob_home'] * 100:5.1f}% | D {row['prob_draw'] * 100:5.1f}% | A {row['prob_away'] * 100:5.1f}%")

        if len(upcoming) > 10:
            print(f"\n... and {len(upcoming) - 10} more fixtures")
        print("=" * 80)

        # Export
        upcoming_output = upcoming[[
            "datetime",
            "home_team",
            "away_team",
            "pred_home_goals",
            "pred_away_goals",
            "prob_home",
            "prob_draw",
            "prob_away",
        ]]

        OUT_UPCOMING.parent.mkdir(parents=True, exist_ok=True)
        upcoming_output.to_csv(OUT_UPCOMING, index=False)

        print(f"\n✅ Saved {len(upcoming_output)} upcoming predictions to {OUT_UPCOMING}")

    print("\n✨ Next step: Run python visualise.py to create enhanced dashboard")


if __name__ == "__main__":
    main()