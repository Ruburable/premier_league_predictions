#!/usr/bin/env python3
"""
predict_scores.py

Reads:
  - data/matches_master.csv
  - data/match_events/*.json

Writes:
  - data/predictions_gameweek.csv

Outputs predictions (expected goals, score probabilities, predicted scorers with probabilities)
using:
  - XGBoost regressors for goals
  - Monte-Carlo simulation for scorers (default N_SIM=2000, adjustable)

Notes:
  - Script is defensive about missing fields and uses fallbacks.
  - Use lineups/injury data later to improve scorer sampling.
"""

from pathlib import Path
import json
import glob
import math
import random
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
from tqdm import tqdm

# ----------------------------
# CONFIG
# ----------------------------
DATA_DIR = Path("data")
MASTER_FILE = DATA_DIR / "matches_master.csv"
EVENTS_GLOB = DATA_DIR / "match_events" / "*.json"
OUTPUT_PREDICTIONS = DATA_DIR / "predictions_gameweek.csv"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Modeling & simulation settings
FORM_WINDOW = 5               # rolling window (matches)
N_MC_SIM = 2000               # Monte Carlo simulations for scorer probabilities
MAX_GOALS_PER_TEAM = 6        # cap per simulation (to limit state space)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# XGBoost parameters
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "objective": "reg:squarederror",
    "random_state": RANDOM_SEED,
    "verbosity": 0,
}

# ----------------------------
# UTIL FUNCTIONS
# ----------------------------

def safe_read_master(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run data downloader first.")
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # unify datetime column name
    for c in ("datetime", "date", "kickoff_time"):
        if c in df.columns:
            df["datetime"] = pd.to_datetime(df[c], errors="coerce")
            break
    # unify goal columns
    if "home_goals" not in df.columns:
        if "home_goals" in df.columns:
            pass
        else:
            # try different names or 'goals' list? if 'goals' present and is stringified list, try parse
            if "goals" in df.columns:
                def parse_goals(x):
                    try:
                        if pd.isna(x):
                            return (None, None)
                        if isinstance(x, str) and x.strip().startswith("["):
                            arr = json.loads(x)
                            return (int(arr[0]) if arr[0] is not None else None, int(arr[1]) if arr[1] is not None else None)
                        if isinstance(x, (list, tuple)):
                            return (int(x[0]) if x[0] is not None else None, int(x[1]) if x[1] is not None else None)
                    except Exception:
                        return (None, None)
                    return (None, None)
                parsed = df["goals"].apply(parse_goals)
                df["home_goals"] = parsed.apply(lambda t: t[0])
                df["away_goals"] = parsed.apply(lambda t: t[1])
    # unify xg columns
    if "home_xg" not in df.columns and "xG" in df.columns:
        # if xG is list-like
        def parse_xg(x):
            try:
                if pd.isna(x):
                    return (None, None)
                if isinstance(x, str) and x.strip().startswith("["):
                    arr = json.loads(x)
                    return (float(arr[0]) if arr[0] is not None else None, float(arr[1]) if arr[1] is not None else None)
                if isinstance(x, (list, tuple)):
                    return (float(x[0]) if x[0] is not None else None, float(x[1]) if x[1] is not None else None)
            except Exception:
                return (None, None)
            return (None, None)
        parsed_xg = df["xG"].apply(parse_xg)
        df["home_xg"] = parsed_xg.apply(lambda t: t[0])
        df["away_xg"] = parsed_xg.apply(lambda t: t[1])
    # ensure numeric types
    for col in ["home_goals", "away_goals", "home_xg", "away_xg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ----------------------------
# PLAYER STATS FROM EVENT JSONS
# ----------------------------

def list_event_files():
    return sorted(glob.glob(str(EVENTS_GLOB)))

def load_event_json(pth):
    try:
        with open(pth, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None

def aggregate_player_stats(events_dir_glob=EVENTS_GLOB):
    """
    Returns DataFrame with per-player aggregated stats:
      - player (name)
      - team (last seen)
      - goals (count)
      - shots (count)
      - xg (sum of shot xG)
      - matches (count distinct matches where player had at least one shot)
      - goals_per_match, xg_per_shot
    """
    files = list_event_files()
    player = {}
    # iterate events
    for f in tqdm(files, desc="Aggregating player events", unit="file"):
        data = load_event_json(f)
        if not data:
            continue
        match_id = data.get("_match_id") or Path(f).stem
        # shots data often under 'shotsData' key or generic keys; handle variants
        shots = data.get("shotsData") or data.get("shots") or []
        if not isinstance(shots, list):
            # sometimes nested structure: try extract lists
            try:
                # flatten if dict with list values
                shots = []
                for v in data.values():
                    if isinstance(v, list):
                        shots.extend(v)
            except Exception:
                shots = []
        seen_in_match = set()
        for s in shots:
            # shot objects vary; try typical fields
            pname = s.get("player") or s.get("player_name") or s.get("s") or s.get("playerId") or None
            if not pname:
                continue
            team = s.get("h_team") or s.get("h") or s.get("team") or s.get("team_title") or s.get("team_name") or None
            # xG fields
            xg_val = None
            for key in ("xG", "xg", "shot_xg"):
                if key in s:
                    try:
                        xg_val = float(s.get(key) or 0.0)
                        break
                    except Exception:
                        xg_val = 0.0
            res = s.get("result") or s.get("type") or ""
            # determine goal
            is_goal = False
            if isinstance(res, str) and "goal" in res.lower():
                is_goal = True
            if s.get("isGoal") or s.get("is_goal"):
                is_goal = True
            # update player record
            rec = player.setdefault(pname, {"team": team, "goals": 0, "shots": 0, "xg": 0.0, "matches": set()})
            if team:
                rec["team"] = team
            rec["shots"] += 1
            rec["xg"] += float(xg_val or 0.0)
            if is_goal:
                rec["goals"] += 1
            rec["matches"].add(match_id)
            seen_in_match.add(pname)
    # convert to DataFrame
    rows = []
    for pname, r in player.items():
        matches_played = len(r["matches"])
        rows.append({
            "player": pname,
            "team": r.get("team"),
            "goals": r.get("goals", 0),
            "shots": r.get("shots", 0),
            "xg": r.get("xg", 0.0),
            "matches": matches_played,
            "goals_per_match": (r["goals"] / matches_played) if matches_played > 0 else 0.0,
            "xg_per_shot": (r["xg"] / r["shots"]) if r["shots"] > 0 else 0.0
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # normalize team names (strip)
    df["team"] = df["team"].astype(str).str.strip()
    return df

# ----------------------------
# TEAM-LEVEL LOGS & ROLLING FEATURES
# ----------------------------

def build_team_logs(master_df):
    """
    Turn match-level rows into team-level logs (one row per team per match).
    Columns: match_id, team, opponent, is_home, goals_for, goals_against, xg_for, xg_against, date
    """
    rows = []
    for _, r in master_df.iterrows():
        home = r.get("home_team")
        away = r.get("away_team")
        mid = r.get("id") or r.get("match_id") or r.get("matchId")
        dt = r.get("datetime") if "datetime" in r else None
        # handle xg fields
        hxg = r.get("home_xg") if "home_xg" in r else r.get("home_xg")
        axg = r.get("away_xg") if "away_xg" in r else r.get("away_xg")
        rows.append({
            "match_id": mid,
            "team": home,
            "opponent": away,
            "is_home": 1,
            "goals_for": r.get("home_goals"),
            "goals_against": r.get("away_goals"),
            "xg_for": hxg,
            "xg_against": axg,
            "date": dt
        })
        rows.append({
            "match_id": mid,
            "team": away,
            "opponent": home,
            "is_home": 0,
            "goals_for": r.get("away_goals"),
            "goals_against": r.get("home_goals"),
            "xg_for": axg,
            "xg_against": hxg,
            "date": dt
        })
    logs = pd.DataFrame(rows)
    if logs.empty:
        return logs
    logs["date"] = pd.to_datetime(logs["date"], errors="coerce")
    logs = logs.sort_values(["team", "date"]).reset_index(drop=True)
    return logs

def compute_rolling_features(team_logs, window=FORM_WINDOW):
    """
    Compute rolling metrics per team (shifted to be pre-match):
      - rolling_points (sum last N matches)
      - rolling_goals_for, rolling_goals_against (mean)
      - rolling_xg_for (mean)
      - weighted_pts (recent match weighted)
    """
    frames = []
    for team, g in team_logs.groupby("team", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        # compute points
        def pts(row):
            if pd.isna(row["goals_for"]) or pd.isna(row["goals_against"]):
                return np.nan
            if row["goals_for"] > row["goals_against"]:
                return 3
            if row["goals_for"] == row["goals_against"]:
                return 1
            return 0
        g["points"] = g.apply(pts, axis=1)
        g["rolling_points"] = g["points"].shift(1).rolling(window, min_periods=1).sum()
        g["rolling_gf"] = g["goals_for"].shift(1).rolling(window, min_periods=1).mean()
        g["rolling_ga"] = g["goals_against"].shift(1).rolling(window, min_periods=1).mean()
        # rolling xg
        g["rolling_xg"] = g["xg_for"].shift(1).rolling(window, min_periods=1).mean()
        # weighted points (newer matches heavier)
        weights = np.arange(1, window + 1)[::-1]  # e.g. [5,4,3,2,1]
        def weighted_pts(series):
            arr = series.shift(1).rolling(window, min_periods=1).apply(
                lambda vals: np.sum(vals * weights[-len(vals):]) / np.sum(weights[-len(vals):]), raw=True
            )
            return arr
        g["weighted_pts"] = weighted_pts(g["points"])
        frames.append(g)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return result

# ----------------------------
# BUILD MATCH-LEVEL FEATURES
# ----------------------------

def build_match_features(master_df, team_rolling):
    """
    For each match produce features:
      - home_rolling_points, away_rolling_points
      - home_rolling_gf, away_rolling_gf
      - home_rolling_xg, away_rolling_xg
      - diff features
    """
    # helper to lookup last rolling row for team before given date
    def last_before(team, date):
        sub = team_rolling[(team_rolling["team"] == team) & (team_rolling["date"] < date)]
        if sub.empty:
            # fallback to most recent overall
            sub = team_rolling[team_rolling["team"] == team]
            if sub.empty:
                return None
        return sub.iloc[-1]

    rows = []
    for _, r in master_df.iterrows():
        date = r.get("datetime")
        home = r.get("home_team")
        away = r.get("away_team")
        if (home is None) or (away is None):
            continue
        home_row = last_before(home, date)
        away_row = last_before(away, date)
        row = {
            "match_id": r.get("id"),
            "datetime": date,
            "season": r.get("season"),
            "event": r.get("event"),
            "home_team": home,
            "away_team": away,
            "home_goals": r.get("home_goals"),
            "away_goals": r.get("away_goals"),
            "home_xg_obs": r.get("home_xg"),
            "away_xg_obs": r.get("away_xg"),
        }
        # fill features from rolling rows or zeros
        def fill_prefix(prefix, src):
            if src is None:
                row[f"{prefix}_rolling_points"] = 0.0
                row[f"{prefix}_rolling_gf"] = 0.0
                row[f"{prefix}_rolling_ga"] = 0.0
                row[f"{prefix}_rolling_xg"] = 0.0
            else:
                row[f"{prefix}_rolling_points"] = float(src.get("rolling_points") or 0.0)
                row[f"{prefix}_rolling_gf"] = float(src.get("rolling_gf") or 0.0)
                row[f"{prefix}_rolling_ga"] = float(src.get("rolling_ga") or 0.0)
                row[f"{prefix}_rolling_xg"] = float(src.get("rolling_xg") or 0.0)
        fill_prefix("home", home_row)
        fill_prefix("away", away_row)
        row["diff_rolling_points"] = row["home_rolling_points"] - row["away_rolling_points"]
        row["diff_rolling_gf"] = row["home_rolling_gf"] - row["away_rolling_gf"]
        row["diff_rolling_xg"] = row["home_rolling_xg"] - row["away_rolling_xg"]
        rows.append(row)
    feats = pd.DataFrame(rows)
    # ensure numeric
    feats = feats.fillna(0.0)
    return feats

# ----------------------------
# MODELING: training/predicting goals
# ----------------------------

def train_goal_models(feat_df, save_models=True):
    labeled = feat_df.dropna(subset=["home_goals", "away_goals"])
    if len(labeled) < 30:
        print("Warning: small training set — predictions may be poor.")

    feature_cols = [
        "home_rolling_points", "away_rolling_points",
        "home_rolling_gf", "away_rolling_gf",
        "home_rolling_xg", "away_rolling_xg",
        "diff_rolling_points", "diff_rolling_gf", "diff_rolling_xg"
    ]
    X = labeled[feature_cols].values
    y_home = labeled["home_goals"].astype(float).values
    y_away = labeled["away_goals"].astype(float).values

    # simple train/test split
    X_train, X_val, yh_train, yh_val = train_test_split(X, y_home, test_size=0.12, random_state=RANDOM_SEED)
    _, _, ya_train, ya_val = train_test_split(X, y_away, test_size=0.12, random_state=RANDOM_SEED)

    model_home = xgb.XGBRegressor(**XGB_PARAMS)
    model_away = xgb.XGBRegressor(**XGB_PARAMS)

    model_home.fit(X_train, yh_train, eval_set=[(X_val, yh_val)], early_stopping_rounds=30, verbose=False)
    model_away.fit(X_train, ya_train, eval_set=[(X_val, ya_val)], early_stopping_rounds=30, verbose=False)

    if save_models:
        joblib.dump(model_home, MODEL_DIR / "xgb_home_goals.joblib")
        joblib.dump(model_away, MODEL_DIR / "xgb_away_goals.joblib")

    return model_home, model_away, feature_cols

def predict_expected_goals(models, feat_rows, feature_cols):
    model_home, model_away = models
    X = feat_rows[feature_cols].values
    pred_h = model_home.predict(X)
    pred_a = model_away.predict(X)
    # clip and floor
    pred_h = np.clip(pred_h, 0.03, 6.0)
    pred_a = np.clip(pred_a, 0.03, 6.0)
    res = feat_rows.copy()
    res["pred_home_xg"] = pred_h
    res["pred_away_xg"] = pred_a
    return res

# ----------------------------
# SCORELINE PROBS & SCORER SAMPLING
# ----------------------------

def scoreline_probs_closed_form(home_xg, away_xg, max_goals=6):
    probs = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            probs[i, j] = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
    p_home = probs[np.triu_indices(max_goals+1, k=1)].sum()
    p_draw = np.sum(np.diag(probs))
    p_away = probs[np.tril_indices(max_goals+1, k=-1)].sum()
    idx = np.unravel_index(np.argmax(probs), probs.shape)
    most_likely = (int(idx[0]), int(idx[1]))
    return {
        "p_home_win": float(p_home),
        "p_draw": float(p_draw),
        "p_away_win": float(p_away),
        "most_likely_score": most_likely,
        "probs_matrix": probs
    }

def build_team_player_weights(player_stats_df, team_name, top_k=10, alpha_goals=1.0, beta_xg=1.2):
    """
    Create probability distribution over players of a given team to be the scorer.
    We use a linear combination of goals_per_match and xg_per_shot (or xg/match)
    """
    if player_stats_df is None or player_stats_df.empty:
        return [], np.array([])
    team_players = player_stats_df[player_stats_df["team"].str.lower() == str(team_name).lower()].copy()
    if team_players.empty:
        # try fuzzy match by substring
        team_players = player_stats_df[player_stats_df["team"].str.lower().str.contains(str(team_name).lower().split()[0])].copy()
    if team_players.empty:
        return [], np.array([])
    # scoring metric
    team_players["score_metric"] = team_players["goals_per_match"].fillna(0.0) * alpha_goals + team_players["xg_per_shot"].fillna(0.0) * beta_xg
    if team_players["score_metric"].sum() <= 0:
        # fallback to raw goals + shots
        team_players["score_metric"] = team_players["goals"].fillna(0.0) + 0.1 * team_players["shots"].fillna(0.0)
    team_players = team_players.sort_values("score_metric", ascending=False).head(top_k)
    players = team_players["player"].tolist()
    weights = team_players["score_metric"].values.astype(float)
    probs = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)
    return players, probs

def monte_carlo_scorers(home_xg, away_xg, home_players, home_probs, away_players, away_probs, n_sim=N_MC_SIM, max_goals=MAX_GOALS_PER_TEAM):
    """
    Monte Carlo: for each sim, draw home_goals ~ Poisson(home_xg), away_goals ~ Poisson(away_xg),
    then for each team sample that many scorers with replacement according to player probs.
    Returns:
      - score_freqs: Counter of score tuples
      - home_scorer_counts: Counter of player-name -> count
      - away_scorer_counts: Counter
    """
    rng = np.random.default_rng(RANDOM_SEED)
    score_counter = Counter()
    home_scorer_counter = Counter()
    away_scorer_counter = Counter()

    # Precompute categorical samplers by using numpy choice with given probs
    hp = np.array(home_probs) if len(home_probs) > 0 else np.array([])
    ap = np.array(away_probs) if len(away_probs) > 0 else np.array([])

    for _ in range(n_sim):
        # sample goals
        gh = int(rng.poisson(home_xg))
        ga = int(rng.poisson(away_xg))
        gh = min(gh, max_goals)
        ga = min(ga, max_goals)
        score_counter[(gh, ga)] += 1
        # sample scorers for home
        if gh > 0 and len(home_players) > 0:
            # vectorized choice
            picks = rng.choice(home_players, size=gh, p=hp)
            for p in picks:
                home_scorer_counter[p] += 1
        if ga > 0 and len(away_players) > 0:
            picks = rng.choice(away_players, size=ga, p=ap)
            for p in picks:
                away_scorer_counter[p] += 1

    # Convert to probabilities
    total = n_sim
    score_probs = {f"{s[0]}-{s[1]}": count / total for s, count in score_counter.items()}
    home_scorer_probs = {p: cnt / total for p, cnt in home_scorer_counter.items()}
    away_scorer_probs = {p: cnt / total for p, cnt in away_scorer_counter.items()}

    # top scorers list
    top_home = sorted(home_scorer_probs.items(), key=lambda x: -x[1])[:6]
    top_away = sorted(away_scorer_probs.items(), key=lambda x: -x[1])[:6]

    # also compute implied probabilities of win/draw/loss from simulation
    p_home_win = sum(v for k, v in score_probs.items() if int(k.split('-')[0]) > int(k.split('-')[1]))
    p_draw = sum(v for k, v in score_probs.items() if int(k.split('-')[0]) == int(k.split('-')[1]))
    p_away_win = sum(v for k, v in score_probs.items() if int(k.split('-')[0]) < int(k.split('-')[1]))

    return {
        "score_probs": score_probs,
        "top_home_scorers": top_home,
        "top_away_scorers": top_away,
        "p_home_win_mc": p_home_win,
        "p_draw_mc": p_draw,
        "p_away_win_mc": p_away_win,
    }

# ----------------------------
# MAIN PIPELINE
# ----------------------------

def main():
    print("Loading master matches...")
    master = safe_read_master(MASTER_FILE)
    # Filter seasons if needed (we will train on all available past matches)
    master = master.sort_values("datetime")
    # Build player stats from event JSONs
    print("Aggregating player stats...")
    player_stats = aggregate_player_stats(EVENTS_GLOB)
    if player_stats.empty:
        print("Warning: No event JSONs found; scorer prediction will fallback to team-level naive allocation.")

    # Build team logs and rolling features
    print("Building team logs and rolling features...")
    team_logs = build_team_logs(master)
    team_rolling = compute_rolling_features(team_logs, window=FORM_WINDOW)
    # Build match features
    print("Building match-level features...")
    feats = build_match_features(master, team_rolling)

    # Train models
    print("Training goal models (XGBoost)...")
    model_home, model_away, feat_cols = train_goal_models(feats, save_models=True)

    # Identify upcoming fixtures: where home_goals or away_goals is NaN and datetime <= now OR event not None
    now = pd.Timestamp.now()
    upcoming = feats[(feats["home_goals"].isna() | feats["away_goals"].isna())].copy()
    # further restrict to matches with datetime not in far future (optional)
    upcoming = upcoming[ (upcoming["datetime"].isna()) | (pd.to_datetime(upcoming["datetime"]) <= now + pd.Timedelta(days=7)) ]

    if upcoming.empty:
        print("No upcoming fixtures found to predict.")
        return

    # Predict expected goals
    print(f"Predicting expected goals for {len(upcoming)} fixtures...")
    pred_df = predict_expected_goals((model_home, model_away), upcoming, feat_cols)

    predictions = []
    print("Running Monte Carlo scorer simulations (this may take a bit)...")
    for _, r in tqdm(pred_df.iterrows(), total=len(pred_df), desc="Predict matches"):
        home = r["home_team"]
        away = r["away_team"]
        mh = float(r["pred_home_xg"])
        ma = float(r["pred_away_xg"])

        # closed-form score probs
        closed = scoreline_probs_closed_form(mh, ma, max_goals=MAX_GOALS_PER_TEAM)
        most_likely = closed["most_likely_score"]

        # build player weight distributions
        home_players, home_probs = build_team_player_weights(player_stats, home, top_k=12)
        away_players, away_probs = build_team_player_weights(player_stats, away, top_k=12)

        # Monte Carlo scorer sampling
        mc = monte_carlo_scorers(mh, ma, home_players, home_probs, away_players, away_probs, n_sim=N_MC_SIM, max_goals=MAX_GOALS_PER_TEAM)

        # Consolidate top scorer strings
        def top_list_to_str(lst):
            return "; ".join([f"{p} ({prob:.3f})" for p, prob in lst])

        predictions.append({
            "match_id": r["match_id"],
            "datetime": r["datetime"],
            "season": r["season"],
            "event": r.get("event"),
            "home_team": home,
            "away_team": away,
            "pred_home_xg": mh,
            "pred_away_xg": ma,
            "most_likely_score_home": most_likely[0],
            "most_likely_score_away": most_likely[1],
            "p_home_win_closed": closed["p_home_win"],
            "p_draw_closed": closed["p_draw"],
            "p_away_win_closed": closed["p_away_win"],
            "p_home_win_mc": mc["p_home_win_mc"],
            "p_draw_mc": mc["p_draw_mc"],
            "p_away_win_mc": mc["p_away_win_mc"],
            "top_home_scorers": json.dumps(mc["top_home_scorers"]),
            "top_away_scorers": json.dumps(mc["top_away_scorers"]),
            "score_probs_sampled": json.dumps(mc["score_probs"])
        })

    out = pd.DataFrame(predictions)
    out.to_csv(OUTPUT_PREDICTIONS, index=False)
    print(f"Predictions saved to {OUTPUT_PREDICTIONS}")

    # Print readable summary
    print("\nPredictions summary:")
    for _, r in out.iterrows():
        print(f"{r['home_team']} vs {r['away_team']} — xG {r['pred_home_xg']:.2f}:{r['pred_away_xg']:.2f} — "
              f"most likely {int(r['most_likely_score_home'])}-{int(r['most_likely_score_away'])}; "
              f"p(H)={r['p_home_win_mc']:.2f} p(D)={r['p_draw_mc']:.2f} p(A)={r['p_away_win_mc']:.2f}")
        # decode top scorers
        try:
            home_top = json.loads(r["top_home_scorers"])
            away_top = json.loads(r["top_away_scorers"])
            if home_top:
                print("  Top home scorers:", ", ".join([f"{p} ({prob:.2%})" for p, prob in home_top]))
            if away_top:
                print("  Top away scorers:", ", ".join([f"{p} ({prob:.2%})" for p, prob in away_top]))
        except Exception:
            pass

if __name__ == "__main__":
    main()
