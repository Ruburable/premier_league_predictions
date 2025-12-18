#!/usr/bin/env python3
"""
sandbox.py

Purpose:
- Experiment with different models safely
- Verify non-zero predictions
- Produce dashboard-compatible output

IMPORTANT:
- Treats last N matches as "upcoming" intentionally
"""

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# ---------------- CONFIG ---------------- #

DATA = Path("data/matches_master.csv")
OUT = Path("output/predictions_upcoming.csv")

SPLIT_N = 50        # number of matches treated as upcoming
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


def load_data():
    df = pd.read_csv(DATA, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def add_features(df):
    df = df.copy()

    df["xg_diff"] = df["home_xg"] - df["away_xg"]
    df["home_adv"] = 1.0

    return df


def train_model(X, y):
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
    model.fit(X, y)
    return model


def main():
    print("Loading data...")
    df = load_data()
    df = add_features(df)

    print(f"Total matches: {len(df)}")

    # ---------- split ----------
    train = df.iloc[:-SPLIT_N].copy()
    upcoming = df.iloc[-SPLIT_N:].copy()

    print(f"Training matches: {len(train)}")
    print(f"Upcoming matches: {len(upcoming)}")

    # ---------- sanity ----------
    print("\nLabel sanity check:")
    print(train[[TARGET_HOME, TARGET_AWAY]].describe())

    # ---------- train ----------
    X_train = train[FEATURES]
    y_home = train[TARGET_HOME]
    y_away = train[TARGET_AWAY]

    model_home = train_model(X_train, y_home)
    model_away = train_model(X_train, y_away)

    # ---------- eval ----------
    pred_home_train = model_home.predict(X_train)
    pred_away_train = model_away.predict(X_train)

    print("\nModel: XGBoost")
    print("Home goals MAE:", mean_absolute_error(y_home, pred_home_train))
    print("Away goals MAE:", mean_absolute_error(y_away, pred_away_train))

    print("\nSample predictions:")
    for i in range(5):
        print(f"{pred_home_train[i]:.2f} – {pred_away_train[i]:.2f}")

    # ---------- predict upcoming ----------
    X_up = upcoming[FEATURES]

    upcoming["pred_home_goals"] = model_home.predict(X_up)
    upcoming["pred_away_goals"] = model_away.predict(X_up)

    # crude win probs (for dashboard)
    upcoming["prob_home"] = (upcoming["pred_home_goals"] > upcoming["pred_away_goals"]).astype(float)
    upcoming["prob_draw"] = (np.abs(upcoming["pred_home_goals"] - upcoming["pred_away_goals"]) < 0.3).astype(float)
    upcoming["prob_away"] = 1 - upcoming["prob_home"] - upcoming["prob_draw"]

    # ---------- export ----------
    out = upcoming[
        [
            "datetime",
            "home_team",
            "away_team",
            "pred_home_goals",
            "pred_away_goals",
            "prob_home",
            "prob_draw",
            "prob_away",
        ]
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"\n✔ Saved predictions to {OUT}")


if __name__ == "__main__":
    main()
