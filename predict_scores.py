import pandas as pd
import numpy as np
from xgboost import XGBRegressor


DATA_FILE = "data/all_matches.csv"


# ----------------------------------------------------
# FEATURE ENGINEERING
# ----------------------------------------------------

def rolling_team_features(df, window=5):
    teams = pd.unique(df[["home_team", "away_team"]].values.ravel())
    frames = []

    for t in teams:
        sub = df[(df.home_team == t) | (df.away_team == t)].copy()
        sub = sub.sort_values("date")

        sub["team"] = t
        sub["goals_for"] = np.where(sub.home_team == t, sub.home_goals, sub.away_goals)
        sub["goals_against"] = np.where(sub.home_team == t, sub.away_goals, sub.home_goals)

        sub["form_gf"] = sub["goals_for"].rolling(window).mean()
        sub["form_ga"] = sub["goals_against"].rolling(window).mean()

        sub["form_xg"] = np.where(sub.home_team == t, sub.home_xg, sub.away_xg)
        sub["form_xga"] = np.where(sub.home_team == t, sub.away_xg, sub.home_xg)

        sub["form_xg"] = sub["form_xg"].rolling(window).mean()
        sub["form_xga"] = sub["form_xga"].rolling(window).mean()

        frames.append(sub)

    out = pd.concat(frames)
    return out


def attach_team_form(df):
    features = rolling_team_features(df)
    df = df.merge(
        features[[
            "season", "date", "home_team", "away_team",
            "team", "form_gf", "form_ga", "form_xg", "form_xga"
        ]],
        left_on=["season", "date", "home_team"],
        right_on=["season", "date", "team"],
        how="left"
    ).rename(columns={
        "form_gf": "home_form_gf",
        "form_ga": "home_form_ga",
        "form_xg": "home_form_xg",
        "form_xga": "home_form_xga"
    }).drop(columns=["team"])

    df = df.merge(
        features[[
            "season", "date", "home_team", "away_team",
            "team", "form_gf", "form_ga", "form_xg", "form_xga"
        ]],
        left_on=["season", "date", "away_team"],
        right_on=["season", "date", "team"],
        how="left"
    ).rename(columns={
        "form_gf": "away_form_gf",
        "form_ga": "away_form_ga",
        "form_xg": "away_form_xg",
        "form_xga": "away_form_xga"
    }).drop(columns=["team"])

    return df


# ----------------------------------------------------
# MODEL TRAINING
# ----------------------------------------------------

def train_xgboost(df):
    df = df.dropna(subset=["home_goals", "away_goals"])

    feature_cols = [
        "home_form_gf", "home_form_ga", "home_form_xg", "home_form_xga",
        "away_form_gf", "away_form_ga", "away_form_xg", "away_form_xga"
    ]

    X = df[feature_cols]
    y_home = df["home_goals"]
    y_away = df["away_goals"]

    model_home = XGBRegressor(objective="reg:squarederror")
    model_away = XGBRegressor(objective="reg:squarederror")

    model_home.fit(X, y_home)
    model_away.fit(X, y_away)

    return model_home, model_away


# ----------------------------------------------------
# PREDICT NEXT GAMEWEEK
# ----------------------------------------------------

def predict_next(df, model_home, model_away):
    future = df[df["home_goals"].isna()].copy()

    features = future[[
        "home_form_gf", "home_form_ga", "home_form_xg", "home_form_xga",
        "away_form_gf", "away_form_ga", "away_form_xg", "away_form_xga"
    ]]

    future["pred_home_goals"] = model_home.predict(features)
    future["pred_away_goals"] = model_away.predict(features)

    return future


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------

if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")

    df = attach_team_form(df)

    model_home, model_away = train_xgboost(df)

    predictions = predict_next(df, model_home, model_away)

    predictions.to_csv("data/predictions.csv", index=False)
    print("Predictions saved.")
