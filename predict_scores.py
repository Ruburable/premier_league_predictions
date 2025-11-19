#!/usr/bin/env python3
"""
predict_gameweek.py

Reads:
  - data/understat_matches_all.csv
  - data/understat_match_events/*.json

Outputs:
  - data/predictions_gameweek.csv  (one row per upcoming fixture with scores + scorers + probabilities)
  - prints short summary to stdout

Requirements:
  pip install pandas numpy scikit-learn xgboost joblib scipy
"""

import os
import json
import glob
import math
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib

# -----------------------
# Configuration
# -----------------------
DATA_DIR = Path("data")
MATCHES_FILE = DATA_DIR / "understat_matches_all.csv"
MATCH_EVENTS_DIR = DATA_DIR / "understat_match_events"
PREDICTIONS_FILE = DATA_DIR / "predictions_gameweek.csv"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Modeling hyperparams
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "objective": "reg:squarederror",
    "random_state": SEED,
}

# Rolling window for form (number of prior matches used)
FORM_WINDOW = 5

# Number of goals to consider for scorer sampling if predicted expected goals large
MAX_GOALS_SAMPLING = 6

# -----------------------
# Helpers
# -----------------------

def safe_read_matches(path):
    if not path.exists():
        raise FileNotFoundError(f"Matches file not found: {path}")
    df = pd.read_csv(path)
    # Normalize column names lower-case
    df.columns = [c.strip() for c in df.columns]
    # Ensure date parse if exists (Understat sometimes uses 'date' column)
    for col in ("date", "kickoff_time"):
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass
    return df

def list_event_files(events_dir):
    return sorted(glob.glob(str(events_dir / "*.json")))

def load_match_event_json(match_json_path):
    with open(match_json_path, "r", encoding="utf-8") as fh:
        return json.load(fh)

# -----------------------
# Player stats aggregation
# -----------------------

def build_player_stats(events_dir):
    """
    Walk per-match JSON files and aggregate player-level stats:
      - goals
      - shots
      - xG (sum of shot xG)
      - matches_played (count of matches where player had at least one event)
    Returns DataFrame indexed by player name with columns: team, goals, shots, xg, matches
    """
    player = defaultdict(lambda: {"team": None, "goals": 0, "shots": 0, "xg": 0.0, "matches": set()})
    files = list_event_files(events_dir)
    for fpath in files:
        try:
            data = load_match_event_json(fpath)
        except Exception:
            continue
        # shotsData commonly stored under "shotsData" key
        shots = data.get("shotsData") or data.get("shots") or []
        # Attach a match id for 'matches' set
        match_id = data.get("_match_id") or Path(fpath).stem
        # Understat shot objects often have: player, result ('goal'/'saved' etc.), xG, h_team/a_team info
        for s in shots:
            # fields may be named differently across scrapes: try multiple keys
            player_name = s.get("player") or s.get("player_name") or s.get("s") or None
            if not player_name:
                # sometimes structure is nested
                player_name = s.get("player_name") if isinstance(s, dict) else None
            if not player_name:
                continue
            team = s.get("h_team") or s.get("h") or s.get("team") or s.get("team_name") or None
            # determine if it's a home shot - some shot records include 'h_a' or similar; but team field is enough
            xg_val = s.get("xG") or s.get("xg") or s.get("shot_xg") or 0.0
            try:
                xg_val = float(xg_val)
            except Exception:
                xg_val = 0.0
            # result: goal
            result = s.get("result") or s.get("type") or s.get("situation") or ""
            was_goal = False
            # Understat marks goals with 'goal' or similar strings, or 'isGoal' boolean sometimes
            if isinstance(result, str) and "goal" in result.lower():
                was_goal = True
            if s.get("isGoal") or s.get("is_goal"):
                was_goal = True
            # increment stats
            rec = player[player_name]
            if team:
                rec["team"] = team
            if was_goal:
                rec["goals"] += 1
            rec["shots"] += 1
            rec["xg"] += xg_val
            rec["matches"].add(match_id)
    # Normalize to DataFrame
    rows = []
    for name, rec in player.items():
        matches_played = len(rec["matches"])
        rows.append({
            "player": name,
            "team": rec["team"],
            "goals": rec["goals"],
            "shots": rec["shots"],
            "xg": rec["xg"],
            "matches": matches_played,
            "goals_per90": (rec["goals"] / matches_played) if matches_played > 0 else 0.0,
            "xg_per_shot": (rec["xg"] / rec["shots"]) if rec["shots"] > 0 else 0.0
        })
    df = pd.DataFrame(rows)
    # drop players without a team (rare)
    df = df[df["team"].notna()].reset_index(drop=True)
    return df

# -----------------------
# Feature engineering for matches
# -----------------------

def compute_team_rolling_form(matches_df, window=FORM_WINDOW):
    """
    For each team produce rolling stats (goals for/against, xG for/against, points)
    Output: DataFrame with rows per team+match (pre-match stats), columns with rolling metrics.
    """
    # Prepare match-level canonical columns
    df = matches_df.copy()
    # Ensure date exists and sorted
    date_col = None
    if "date" in df.columns:
        date_col = "date"
    elif "kickoff_time" in df.columns:
        date_col = "kickoff_time"
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)
    else:
        # fallback: use index order
        df = df.reset_index(drop=True)

    # Build team-match rows
    rows = []
    for _, r in df.iterrows():
        home = r.get("home_team")
        away = r.get("away_team")
        season = r.get("season")
        idxdate = r.get(date_col) if date_col else None
        # only if team names exist
        if pd.isna(home) or pd.isna(away):
            continue
        rows.append({
            "match_id": r.get("match_id") or r.get("id") or r.get("match_id") or r.get("matchid") or r.get("id"),
            "team": home,
            "opponent": away,
            "is_home": 1,
            "goals_for": r.get("home_goals"),
            "goals_against": r.get("away_goals"),
            "home_xg": r.get("home_xg") if "home_xg" in r else r.get("hxG") if "hxG" in r else None,
            "away_xg": r.get("away_xg") if "away_xg" in r else r.get("axG") if "axG" in r else None,
            "date": idxdate,
            "season": season
        })
        rows.append({
            "match_id": r.get("match_id") or r.get("id"),
            "team": away,
            "opponent": home,
            "is_home": 0,
            "goals_for": r.get("away_goals"),
            "goals_against": r.get("home_goals"),
            "home_xg": r.get("home_xg") if "home_xg" in r else None,
            "away_xg": r.get("away_xg") if "away_xg" in r else None,
            "date": idxdate,
            "season": season
        })
    team_logs = pd.DataFrame(rows)
    if team_logs.empty:
        return pd.DataFrame()

    team_logs = team_logs.sort_values(["team", "date"])
    # rolling metrics shifted by 1 (exclude current match)
    aggregated = []
    for team, g in team_logs.groupby("team"):
        g = g.reset_index(drop=True)
        g["rolling_goals_for"] = g["goals_for"].shift(1).rolling(window, min_periods=1).mean()
        g["rolling_goals_against"] = g["goals_against"].shift(1).rolling(window, min_periods=1).mean()
        # points: compute from goals if possible
        def points_row(x):
            if pd.isna(x["goals_for"]) or pd.isna(x["goals_against"]):
                return np.nan
            if x["goals_for"] > x["goals_against"]:
                return 3
            if x["goals_for"] == x["goals_against"]:
                return 1
            return 0
        g["points"] = g.apply(points_row, axis=1)
        g["rolling_points"] = g["points"].shift(1).rolling(window, min_periods=1).sum()
        # rolling xg if available
        # approximate xg per team from home_xg/away_xg in the team row context:
        g["rolling_xg_for"] = g["home_xg"].where(g["is_home"]==1, g["away_xg"]).shift(1).rolling(window, min_periods=1).mean()
        aggregated.append(g)
    return pd.concat(aggregated, ignore_index=True)

def build_match_features(matches_df, team_rolling):
    """
    Produce a match-level feature frame suitable for regressing goals for home/away.
    Features include:
      - home_rolling_points, away_rolling_points
      - home_rolling_goals_for, away_rolling_goals_for
      - home_rolling_xg_for, away_rolling_xg_for
      - simple strength ratios
    """
    mf = matches_df.copy()
    # ensure match id present
    if "match_id" not in mf.columns:
        if "id" in mf.columns:
            mf["match_id"] = mf["id"]
        elif "match_id" in mf.columns:
            pass
        else:
            mf["match_id"] = mf.index.astype(str)

    # Merge rolling stats (take last row before match for each team)
    def lookup_team_stats(row, team_col, prefix):
        team = row.get(team_col)
        date = row.get("date") if "date" in row else row.get("kickoff_time")
        if pd.isna(team):
            return {f"{prefix}_rolling_points": 0.0,
                    f"{prefix}_rolling_gf": 0.0,
                    f"{prefix}_rolling_ga": 0.0,
                    f"{prefix}_rolling_xg": 0.0}
        # find team rows before date
        cand = team_rolling[(team_rolling["team"] == team) & (team_rolling["date"] < date)]
        if cand.empty:
            # fallback to most recent overall for team
            cand = team_rolling[(team_rolling["team"] == team)]
            if cand.empty:
                return {f"{prefix}_rolling_points": 0.0,
                        f"{prefix}_rolling_gf": 0.0,
                        f"{prefix}_rolling_ga": 0.0,
                        f"{prefix}_rolling_xg": 0.0}
        last = cand.iloc[-1]
        return {f"{prefix}_rolling_points": float(last.get("rolling_points", 0.0) or 0.0),
                f"{prefix}_rolling_gf": float(last.get("rolling_goals_for", 0.0) or 0.0),
                f"{prefix}_rolling_ga": float(last.get("rolling_goals_against", 0.0) or 0.0),
                f"{prefix}_rolling_xg": float(last.get("rolling_xg_for", 0.0) or 0.0)}
    feats = []
    for _, r in mf.iterrows():
        home_stats = lookup_team_stats(r, "home_team", "home")
        away_stats = lookup_team_stats(r, "away_team", "away")
        # baseline league averages for normalization
        home_xg = r.get("home_xg") if "home_xg" in r else np.nan
        away_xg = r.get("away_xg") if "away_xg" in r else np.nan
        row = {
            "match_id": r["match_id"],
            "home_team": r.get("home_team"),
            "away_team": r.get("away_team"),
            "date": r.get("date") if "date" in r else r.get("kickoff_time"),
            "home_xg_observed": home_xg,
            "away_xg_observed": away_xg,
            "home_goals": r.get("home_goals"),
            "away_goals": r.get("away_goals"),
        }
        row.update(home_stats)
        row.update(away_stats)
        # derived features
        row["diff_rolling_points"] = row["home_rolling_points"] - row["away_rolling_points"]
        row["diff_rolling_gf"] = row["home_rolling_gf"] - row["away_rolling_gf"]
        row["diff_rolling_xg"] = row["home_rolling_xg"] - row["away_rolling_xg"]
        feats.append(row)
    feat_df = pd.DataFrame(feats)
    # fill NaNs with zeros for modeling
    feat_df = feat_df.fillna(0.0)
    return feat_df

# -----------------------
# Modeling: goals prediction
# -----------------------

def train_goal_models(feature_df):
    """
    Train two XGBoost regressors: one for home goals, one for away goals.
    Returns (model_home, model_away).
    """
    train_df = feature_df.dropna(subset=["home_goals", "away_goals"])
    if train_df.shape[0] < 50:
        print("Warning: less than 50 labeled matches available; model quality may be poor.")
    X_cols = [
        "home_rolling_points", "away_rolling_points",
        "home_rolling_gf", "away_rolling_gf",
        "home_rolling_xg", "away_rolling_xg",
        "diff_rolling_points", "diff_rolling_gf", "diff_rolling_xg"
    ]
    X = train_df[X_cols].values
    y_home = train_df["home_goals"].astype(float).values
    y_away = train_df["away_goals"].astype(float).values

    X_train, X_val, y_train_home, y_val_home = train_test_split(X, y_home, test_size=0.12, random_state=SEED)
    _, _, y_train_away, y_val_away = train_test_split(X, y_away, test_size=0.12, random_state=SEED)

    model_home = xgb.XGBRegressor(**XGB_PARAMS)
    model_away = xgb.XGBRegressor(**XGB_PARAMS)
    model_home.fit(X_train, y_train_home, eval_set=[(X_val, y_val_home)], early_stopping_rounds=20, verbose=False)
    model_away.fit(X_train, y_train_away, eval_set=[(X_val, y_val_away)], early_stopping_rounds=20, verbose=False)

    # Save models
    joblib.dump(model_home, MODEL_DIR / "xgb_home_goals.joblib")
    joblib.dump(model_away, MODEL_DIR / "xgb_away_goals.joblib")
    return model_home, model_away, X_cols

def predict_expected_goals(match_feat_df, model_home, model_away, X_cols):
    """
    For provided match-level feature rows (including future fixtures with NaN goals),
    predict expected goals (floats) for home and away.
    """
    X = match_feat_df[X_cols].values
    pred_home = model_home.predict(X)
    pred_away = model_away.predict(X)
    match_feat_df = match_feat_df.copy()
    match_feat_df["pred_home_xg"] = np.clip(pred_home, 0.03, 5.0)  # clip to reasonable ranges
    match_feat_df["pred_away_xg"] = np.clip(pred_away, 0.03, 5.0)
    return match_feat_df

# -----------------------
# Scoreline probabilities + choose scorers
# -----------------------

def scoreline_probabilities(home_xg, away_xg, max_goals=6):
    """
    Compute probability table for scores 0..max_goals using independent Poisson
    Returns: dict with p_home_win, p_draw, p_away_win, most_likely_score (tuple)
    """
    probs = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            probs[i, j] = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
    p_home_win = probs[np.triu_indices(max_goals+1, k=1)].sum()
    p_draw = np.sum(np.diag(probs))
    p_away_win = probs[np.tril_indices(max_goals+1, k=-1)].sum()
    # most likely score = argmax prob
    idx = np.unravel_index(np.argmax(probs), probs.shape)
    most_likely = (int(idx[0]), int(idx[1]))
    return {
        "p_home_win": float(p_home_win),
        "p_draw": float(p_draw),
        "p_away_win": float(p_away_win),
        "most_likely_score": most_likely,
        "probs_matrix": probs
    }

def build_team_player_weights(player_stats_df, team_name, alpha_goals=1.0, beta_xg=1.2, min_players=3):
    """
    For a team, compute scoring weights for each player in player_stats_df.
    Score = alpha * goals_per90 + beta * xg_per_shot (or xg per match)
    Returns two arrays: players_list, probabilities (normalized)
    If insufficient players, create synthetic weights from team average.
    """
    team_players = player_stats_df[player_stats_df["team"] == team_name].copy()
    if team_players.empty:
        # fallback: use top scorers across league assigned to team as unknown (unlikely)
        return [], []
    # compute metric
    # goals per match ~ goals / matches ; use goals_per90 already computed earlier as goals / matches
    # xg_per_shot already computed as xg / shots
    team_players["score_metric"] = team_players["goals_per90"].fillna(0.0) * alpha_goals + team_players["xg_per_shot"].fillna(0.0) * beta_xg
    # if all zeros, fallback to goals raw or shots
    if team_players["score_metric"].sum() <= 0:
        team_players["score_metric"] = team_players["goals"].fillna(0.0) + 0.1 * team_players["shots"].fillna(0.0)
    # restrict to top N players by metric to limit noise
    team_players = team_players.sort_values("score_metric", ascending=False).head(max(min_players, 8))
    players = team_players["player"].tolist()
    weights = team_players["score_metric"].values.astype(float)
    # normalize
    if weights.sum() <= 0:
        probs = np.ones(len(weights)) / len(weights)
    else:
        probs = weights / weights.sum()
    return players, probs

def sample_scorers_for_team(n_goals, players, probs):
    """
    Sample n_goals scorers from players according to probs.
    Allow repeat scorers (same player can score multiple goals).
    Returns list of chosen player names.
    """
    if n_goals <= 0 or len(players) == 0:
        return []
    choices = list(np.random.choice(players, size=n_goals, p=probs))
    return choices

# -----------------------
# Main pipeline
# -----------------------

def main():
    print("Loading matches...")
    matches = safe_read_matches(MATCHES_FILE)

    # Normalize columns used below
    # Understat often contains 'home_team', 'away_team', 'home_goals', 'away_goals', 'home_xg', 'away_xg'
    # Ensure those columns exist or create placeholders
    for col in ["home_team", "away_team"]:
        if col not in matches.columns:
            raise KeyError(f"Required column missing in matches file: {col}")

    # Build player stats from per-match JSONs
    print("Aggregating player stats from per-match event JSONs...")
    player_stats = build_player_stats(MATCH_EVENTS_DIR)
    if player_stats.empty:
        print("Warning: no player event JSONs found. Goal-scorer predictions will be naive (team-level).")

    # Compute team rolling form
    print("Computing team rolling form...")
    team_rolling = compute_team_rolling_form(matches, window=FORM_WINDOW)

    # Build match-level features
    print("Constructing match-level features...")
    match_feats = build_match_features(matches, team_rolling)

    # Split into labeled (past) and upcoming (future)
    labeled = match_feats[(match_feats["home_goals"].notna()) & (match_feats["away_goals"].notna())]
    upcoming = match_feats[(match_feats["home_goals"].isna()) | (match_feats["away_goals"].isna())]

    if labeled.empty:
        print("No labeled matches found (no matches with goals). Cannot train. Exiting.")
        return

    # Train models
    print("Training goal prediction models (XGBoost)...")
    model_home, model_away, X_cols = train_goal_models(match_feats)

    # Predict expected goals for upcoming fixtures
    if upcoming.empty:
        print("No upcoming fixtures found in dataset (no rows with missing goals). Nothing to predict.")
        return

    print(f"Predicting for {len(upcoming)} upcoming fixtures...")
    upcoming_pred = predict_expected_goals(upcoming, model_home, model_away, X_cols)

    # For each upcoming fixture compute score probabilities and sample scorers
    rows_out = []
    for _, r in upcoming_pred.iterrows():
        home = r["home_team"]
        away = r["away_team"]
        match_id = r["match_id"]
        home_xg = float(r["pred_home_xg"])
        away_xg = float(r["pred_away_xg"])
        probs = scoreline_probabilities(home_xg, away_xg, max_goals=MAX_GOALS_SAMPLING)
        most_likely = probs["most_likely_score"]
        # We'll simulate expected number of goals as rounded expected xg (but use Poisson sampling to get distribution)
        exp_home = home_xg
        exp_away = away_xg
        # For scorer sampling, determine players & weights
        players_home, probs_home = build_team_player_weights(player_stats, home)
        players_away, probs_away = build_team_player_weights(player_stats, away)
        # decide number of goals to sample: use most likely or round(exp)
        # We'll sample n_home by drawing from Poisson(home_xg) once (one scenario) and similarly for away
        n_home = int(poisson.rvs(mu=home_xg, random_state=np.random.RandomState()))
        n_away = int(poisson.rvs(mu=away_xg, random_state=np.random.RandomState()))
        # Cap at MAX_GOALS_SAMPLING
        n_home = min(n_home, MAX_GOALS_SAMPLING)
        n_away = min(n_away, MAX_GOALS_SAMPLING)
        scorers_home = sample_scorers_for_team(n_home, players_home, probs_home) if len(players_home)>0 else []
        scorers_away = sample_scorers_for_team(n_away, players_away, probs_away) if len(players_away)>0 else []

        rows_out.append({
            "match_id": match_id,
            "home_team": home,
            "away_team": away,
            "pred_home_xg": home_xg,
            "pred_away_xg": away_xg,
            "most_likely_score_home": most_likely[0],
            "most_likely_score_away": most_likely[1],
            "p_home_win": probs["p_home_win"],
            "p_draw": probs["p_draw"],
            "p_away_win": probs["p_away_win"],
            "simulated_home_goals": n_home,
            "simulated_away_goals": n_away,
            "predicted_scorers_home": "; ".join(scorers_home),
            "predicted_scorers_away": "; ".join(scorers_away)
        })

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(PREDICTIONS_FILE, index=False)
    print(f"Saved predictions to {PREDICTIONS_FILE}")

    # Print quick summary
    print("\nPredictions summary:")
    for _, row in out_df.iterrows():
        print(f"{row['home_team']} vs {row['away_team']} — expected xG {row['pred_home_xg']:.2f} : {row['pred_away_xg']:.2f} — "
              f"most likely {int(row['most_likely_score_home'])}-{int(row['most_likely_score_away'])}; "
              f"scorers (home): {row['predicted_scorers_home'] or 'N/A'} — scorers (away): {row['predicted_scorers_away'] or 'N/A'}")

if __name__ == "__main__":
    main()
