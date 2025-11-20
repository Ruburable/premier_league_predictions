import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import os


def safe_fit(model, X_train, y_train, X_val, y_val):
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=30,
            verbose=False
        )
    except TypeError:
        print("XGBoost version does not support early_stopping_rounds. Training without early stopping.")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    return model


def train_goal_models(feats, save_models=True):

    target_home = "home_goals"
    target_away = "away_goals"

    feature_cols = [
        c for c in feats.columns
        if c not in [
            target_home, target_away,
            "match_id", "date", "season", "week",
            "h_team", "a_team"
        ]
    ]

    X = feats[feature_cols].fillna(0)
    y_home = feats[target_home]
    y_away = feats[target_away]

    X_train, X_val, yh_train, yh_val, ya_train, ya_val = train_test_split(
        X, y_home, y_away, test_size=0.15, shuffle=True, random_state=42
    )

    model_home = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=42
    )

    model_away = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=42
    )

    print("Training HOME goal model...")
    model_home = safe_fit(model_home, X_train, yh_train, X_val, yh_val)

    print("Training AWAY goal model...")
    model_away = safe_fit(model_away, X_train, ya_train, X_val, ya_val)

    if save_models:
        os.makedirs("models", exist_ok=True)
        model_home.save_model("models/model_home.json")
        model_away.save_model("models/model_away.json")
        print("Models saved in: models/")

    return model_home, model_away, feature_cols


def main():
    print("Loading master dataset...")

    master_path = "data/matches_master.csv"
    feats_path = "data/match_features.csv"

    if not os.path.exists(master_path):
        raise FileNotFoundError(f"Missing master dataset: {master_path}")

    if not os.path.exists(feats_path):
        raise FileNotFoundError(f"Missing feature dataset: {feats_path}")

    feats = pd.read_csv(feats_path)
    print(f"Loaded {len(feats)} rows of feature data.")

    print("Training models...")
    model_home, model_away, feature_cols = train_goal_models(feats)

    print("Training complete.")
    print(f"Features used: {len(feature_cols)}")


if __name__ == "__main__":
    main()
