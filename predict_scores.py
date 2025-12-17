import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from pathlib import Path

DATA_PATH = Path("data/matches_master.csv")
ROLLING_WINDOWS = [3, 5, 10]


# -------------------------------------------------------
# Load data
# -------------------------------------------------------
def load_matches():
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


# -------------------------------------------------------
# Build team-level event table
# -------------------------------------------------------
def build_team_events(df):
    home = df.rename(columns={
        "home_team": "team",
        "away_team": "opponent",
        "home_goals": "goals_for",
        "away_goals": "goals_against",
        "home_xg": "xg_for",
        "away_xg": "xg_against"
    }).assign(is_home=1)

    away = df.rename(columns={
        "away_team": "team",
        "home_team": "opponent",
        "away_goals": "goals_for",
        "home_goals": "goals_against",
        "away_xg": "xg_for",
        "home_xg": "xg_against"
    }).assign(is_home=0)

    events = pd.concat([home, away], ignore_index=True)
    events = events[[
        "datetime", "team", "opponent", "is_home",
        "goals_for", "goals_against", "xg_for", "xg_against"
    ]]

    return events.sort_values(["team", "datetime"])


# -------------------------------------------------------
# Rolling feature engineering
# -------------------------------------------------------
def add_rolling_features(events):
    for w in ROLLING_WINDOWS:
        events[f"xg_for_avg_{w}"] = (
            events.groupby("team")["xg_for"]
            .rolling(w, min_periods=1).mean().reset_index(0, drop=True)
        )

        events[f"xg_against_avg_{w}"] = (
            events.groupby("team")["xg_against"]
            .rolling(w, min_periods=1).mean().reset_index(0, drop=True)
        )

        events[f"goals_for_avg_{w}"] = (
            events.groupby("team")["goals_for"]
            .rolling(w, min_periods=1).mean().reset_index(0, drop=True)
        )

    return events


# -------------------------------------------------------
# Attach features back to match-level rows
# -------------------------------------------------------
def attach_match_features(matches, events):
    events = events.copy()
    events["match_id"] = events.groupby(["team", "datetime"]).cumcount()

    home_feats = events[events.is_home == 1].copy()
    away_feats = events[events.is_home == 0].copy()

    home_feats = home_feats.add_prefix("home_")
    away_feats = away_feats.add_prefix("away_")

    df = matches.copy()
    df = df.merge(
        home_feats,
        left_on=["home_team", "datetime"],
        right_on=["home_team", "home_datetime"],
        how="left"
    )

    df = df.merge(
        away_feats,
        left_on=["away_team", "datetime"],
        right_on=["away_team", "away_datetime"],
        how="left"
    )

    return df.dropna().reset_index(drop=True)


# -------------------------------------------------------
# Model training
# -------------------------------------------------------
def train_models(df):
    feature_cols = [
        c for c in df.columns
        if any(k in c for k in ["xg_for_avg", "xg_against_avg", "goals_for_avg"])
    ]

    X = df[feature_cols]
    y_home = df["home_goals"]
    y_away = df["away_goals"]

    model_home = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    model_away = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    model_home.fit(X, y_home)
    model_away.fit(X, y_away)

    return model_home, model_away, feature_cols


# -------------------------------------------------------
# Main pipeline
# -------------------------------------------------------
def main():
    print("Loading matches...")
    matches = load_matches()

    print("Building team events...")
    events = build_team_events(matches)

    print("Adding rolling features...")
    events = add_rolling_features(events)

    print("Attaching match features...")
    model_df = attach_match_features(matches, events)

    print(f"Training on {len(model_df)} matches")

    model_home, model_away, features = train_models(model_df)

    preds = model_df.copy()
    preds["pred_home_goals"] = model_home.predict(preds[features])
    preds["pred_away_goals"] = model_away.predict(preds[features])

    print(
        preds[[
            "home_team", "away_team",
            "home_goals", "away_goals",
            "pred_home_goals", "pred_away_goals"
        ]].tail(10)
    )


if __name__ == "__main__":
    main()
