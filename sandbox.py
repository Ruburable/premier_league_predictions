#!/usr/bin/env python3
"""
sandbox.py

Interactive experimentation environment for match score models.

Reads:
  - data/matches_master.csv   (historical only)

Allows:
  - Rapid feature selection
  - Model swapping (XGBoost, Poisson, Linear, RF)
  - Immediate diagnostics (label sanity, feature variance)
"""

import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# === MODELS ===
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = Path("data/matches_master.csv")

RANDOM_STATE = 42


# ---------------------------------------------------
# DATA LOADING
# ---------------------------------------------------

def load_data():
    df = pd.read_csv(DATA_PATH)

    # Parse datetime
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Drop future / missing labels
    df = df.dropna(subset=["home_goals", "away_goals"])

    # Ensure numeric
    for c in ["home_goals", "away_goals", "home_xg", "away_xg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna()

    print(f"Loaded {len(df)} historical matches")
    return df


# ---------------------------------------------------
# FEATURE ENGINEERING (MINIMAL, SAFE)
# ---------------------------------------------------

def build_features(df):
    """
    Keep this intentionally simple.
    Add / remove features freely.
    """

    features = pd.DataFrame(index=df.index)

    # Core strength signals
    features["home_xg"] = df["home_xg"]
    features["away_xg"] = df["away_xg"]
    features["xg_diff"] = df["home_xg"] - df["away_xg"]

    # Home advantage (binary)
    features["home_adv"] = 1.0

    return features


# ---------------------------------------------------
# MODEL FACTORY
# ---------------------------------------------------

def get_model(name: str):
    if name == "xgb":
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=600,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE
        )

    if name == "linear":
        return LinearRegression()

    if name == "rf":
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------
# TRAIN / EVAL LOOP
# ---------------------------------------------------

def train_and_eval(df, model_name="xgb"):
    X = build_features(df)
    y_home = df["home_goals"]
    y_away = df["away_goals"]

    print("\nLabel sanity check:")
    print(df[["home_goals", "away_goals"]].describe())

    print("\nFeature variance:")
    print(X.nunique())

    X_train, X_test, yh_train, yh_test = train_test_split(
        X, y_home, test_size=0.2, random_state=RANDOM_STATE
    )
    _, _, ya_train, ya_test = train_test_split(
        X, y_away, test_size=0.2, random_state=RANDOM_STATE
    )

    model_h = get_model(model_name)
    model_a = get_model(model_name)

    model_h.fit(X_train, yh_train)
    model_a.fit(X_train, ya_train)

    pred_h = model_h.predict(X_test)
    pred_a = model_a.predict(X_test)

    print(f"\nModel: {model_name}")
    print("Home goals MAE:", mean_absolute_error(yh_test, pred_h))
    print("Away goals MAE:", mean_absolute_error(ya_test, pred_a))

    print("\nSample predictions:")
    for i in range(5):
        print(f"{pred_h[i]:.2f} – {pred_a[i]:.2f}")

    return model_h, model_a


# ---------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------

def main():
    df = load_data()

    # CHANGE MODEL HERE:
    # options: "xgb", "linear", "rf"
    train_and_eval(df, model_name="xgb")


if __name__ == "__main__":
    main()
