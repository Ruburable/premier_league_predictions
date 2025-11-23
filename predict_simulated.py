import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

MASTER = "output/matches_master.csv"
PLAYERS = "output/players_master.csv"
FIXTURES = "output/upcoming_fixtures.csv"
OUTFILE = "output/predictions.csv"


def load_data():
    matches = pd.read_csv(MASTER)
    players = pd.read_csv(PLAYERS)
    fixtures = pd.read_csv(FIXTURES)
    return matches, players, fixtures


# ------------------------------------------------------------
# Build simple rolling features for prediction
# ------------------------------------------------------------
def build_team_strength(matches):
    """Compute rolling team averages for goals & xG."""
    stats = []

    for team in pd.unique(matches[["home_team", "away_team"]].values.ravel()):
        df_home = matches[matches.home_team == team][["date", "home_goals", "home_xg"]].rename(
            columns={"home_goals": "goals", "home_xg": "xg"})
        df_away = matches[matches.away_team == team][["date", "away_goals", "away_xg"]].rename(
            columns={"away_goals": "goals", "away_xg": "xg"})

        df = pd.concat([df_home, df_away])
        df = df.sort_values("date")

        df["team"] = team
        df["goals_for_rolling"] = df["goals"].rolling(5, min_periods=1).mean()
        df["xg_for_rolling"] = df["xg"].rolling(5, min_periods=1).mean()

        stats.append(df)

    full = pd.concat(stats)
    return full[["team", "date", "goals_for_rolling", "xg_for_rolling"]]


# ------------------------------------------------------------
# Train simple regressors
# ------------------------------------------------------------
def train_models(matches, strength):

    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    strength["date"] = pd.to_datetime(strength["date"])

    # Merge rolling features
    matches = matches.merge(strength, left_on=["home_team", "date"],
                            right_on=["team", "date"], how="left")\
                     .rename(columns={"goals_for_rolling": "home_gf5",
                                      "xg_for_rolling": "home_xg5"})\
                     .drop(columns=["team"])

    matches = matches.merge(strength, left_on=["away_team", "date"],
                            right_on=["team", "date"], how="left")\
                     .rename(columns={"goals_for_rolling": "away_gf5",
                                      "xg_for_rolling": "away_xg5"})\
                     .drop(columns=["team"])

    feat_cols = ["home_gf5", "home_xg5", "away_gf5", "away_xg5"]

    # Fill missing for early season
    matches[feat_cols] = matches[feat_cols].fillna(matches[feat_cols].mean())

    X = matches[feat_cols]
    y_home = matches["home_goals"]
    y_away = matches["away_goals"]

    model_home = RandomForestRegressor(n_estimators=200, random_state=42)
    model_away = RandomForestRegressor(n_estimators=200, random_state=42)

    model_home.fit(X, y_home)
    model_away.fit(X, y_away)

    return model_home, model_away, feat_cols


# ------------------------------------------------------------
# Predict scorers from high-xG players
# ------------------------------------------------------------
def predict_scorers(players, team, n=2):
    df = players[players.team == team].sort_values("xG", ascending=False)
    return df.head(n)["player"].tolist()


# ------------------------------------------------------------
# Predict fixtures
# ------------------------------------------------------------
def predict_fixtures(fixtures, strength, model_home, model_away, feat_cols, players):

    fixtures = fixtures.copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"])

    # Merge strength
    fixtures = fixtures.merge(strength, left_on=["home_team", "date"],
                              right_on=["team", "date"], how="left")\
                       .rename(columns={"goals_for_rolling": "home_gf5",
                                        "xg_for_rolling": "home_xg5"})\
                       .drop(columns=["team"])

    fixtures = fixtures.merge(strength, left_on=["away_team", "date"],
                              right_on=["team", "date"], how="left")\
                       .rename(columns={"goals_for_rolling": "away_gf5",
                                        "xg_for_rolling": "away_xg5"})\
                       .drop(columns=["team"])

    # Fill missing values
    fixtures[feat_cols] = fixtures[feat_cols].fillna(fixtures[feat_cols].mean())

    # Predict goals
    X = fixtures[feat_cols]
    fixtures["pred_home_goals"] = model_home.predict(X)
    fixtures["pred_away_goals"] = model_away.predict(X)

    # Round to scoreline prediction
    fixtures["scoreline"] = fixtures.apply(
        lambda r: f"{int(round(r.pred_home_goals))}–{int(round(r.pred_away_goals))}", axis=1
    )

    # Simple outcome probabilities (heuristic)
    fixtures["p_home_win"] = 1 / (1 + np.exp(-(fixtures.pred_home_goals - fixtures.pred_away_goals)))
    fixtures["p_away_win"] = 1 - fixtures["p_home_win"]
    fixtures["p_draw"] = 1 - abs(fixtures["p_home_win"] - fixtures["p_away_win"])

    # Predict top scorers
    fixtures["expected_home_scorers"] = fixtures.apply(
        lambda r: predict_scorers(players, r.home_team), axis=1)
    fixtures["expected_away_scorers"] = fixtures.apply(
        lambda r: predict_scorers(players, r.away_team), axis=1)

    return fixtures


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    matches, players, fixtures = load_data()

    strength = build_team_strength(matches)
    model_home, model_away, feat_cols = train_models(matches, strength)

    preds = predict_fixtures(fixtures, strength, model_home, model_away, feat_cols, players)
    preds.to_csv(OUTFILE, index=False)

    print(f"\n✔ Predictions saved to {OUTFILE}\n")
    print(preds[[
        "date", "home_team", "away_team", "scoreline",
        "p_home_win", "p_draw", "p_away_win",
        "expected_home_scorers", "expected_away_scorers"
    ]])


if __name__ == "__main__":
    main()
