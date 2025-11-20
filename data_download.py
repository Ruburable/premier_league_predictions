#!/usr/bin/env python3
"""
predict_scores_from_master.py

- Loads data/matches_master.csv and per-match JSONs in data/match_events/
- Builds rolling team features, trains XGBoost home/away goal regressors (version-safe)
- Predicts expected goals for upcoming fixtures and simulates scorelines
- Produces data/predictions_gameweek.csv with expected scores, probabilities and top scorer lists
"""

import json
import glob
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
from tqdm import tqdm

# configuration
DATA_DIR = Path("data")
MASTER_FILE = DATA_DIR / "matches_master.csv"
EVENTS_GLOB = DATA_DIR / "match_events" / "*.json"
OUTPUT_PREDICTIONS = DATA_DIR / "predictions_gameweek.csv"
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FORM_WINDOW = 5
N_MC_SIM = 2000
MAX_GOALS_PER_TEAM = 6
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def safe_fit(model, X_train, y_train, X_val, y_val):
    try:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=30, verbose=False)
    except TypeError:
        print("XGBoost early_stopping_rounds not supported by this version; training without early stopping.")
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def safe_read_master(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the downloader first.")
    df = pd.read_csv(path)
    # normalize datetime
    for c in ("datetime", "date", "kickoff_time"):
        if c in df.columns:
            df["datetime"] = pd.to_datetime(df[c], errors="coerce")
            break
    # parse goals/xG when stored as lists (stringified)
    if "goals" in df.columns and ("home_goals" not in df.columns or "away_goals" not in df.columns):
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
    if "xG" in df.columns and ("home_xg" not in df.columns or "away_xg" not in df.columns):
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
        parsed = df["xG"].apply(parse_xg)
        df["home_xg"] = parsed.apply(lambda t: t[0])
        df["away_xg"] = parsed.apply(lambda t: t[1])
    # ensure numeric columns
    for col in ("home_goals", "away_goals", "home_xg", "away_xg"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def list_event_files():
    return sorted(glob.glob(str(EVENTS_GLOB)))


def load_event_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def aggregate_player_stats(events_glob=EVENTS_GLOB):
    files = list_event_files()
    player = {}
    for f in tqdm(files, desc="Aggregating player events", unit="file"):
        data = load_event_json(f)
        if not data:
            continue
        match_id = data.get("_match_id") or Path(f).stem
        shots = data.get("shotsData") or data.get("shots") or []
        if not isinstance(shots, list):
            # try to extract lists
            shots = []
            for v in data.values():
                if isinstance(v, list):
                    shots.extend(v)
        for s in shots:
            pname = s.get("player") or s.get("player_name") or s.get("s")
            if not pname:
                continue
            team = s.get("h_team") or s.get("h") or s.get("team") or s.get("team_name")
            xg = None
            for k in ("xG", "xg", "shot_xg"):
                if k in s:
                    try:
                        xg = float(s.get(k) or 0.0)
                        break
                    except Exception:
                        xg = 0.0
            res = s.get("result") or s.get("type") or ""
            is_goal = False
            if isinstance(res, str) and "goal" in res.lower():
                is_goal = True
            if s.get("isGoal") or s.get("is_goal"):
                is_goal = True
            rec = player.setdefault(pname, {"team": team, "goals": 0, "shots": 0, "xg": 0.0, "matches": set()})
            if team:
                rec["team"] = team
            rec["shots"] += 1
            rec["xg"] += float(xg or 0.0)
            if is_goal:
                rec["goals"] += 1
            rec["matches"].add(match_id)
    rows = []
    for pname, r in player.items():
        matches = len(r["matches"])
        rows.append({
            "player": pname,
            "team": r.get("team"),
            "goals": r.get("goals", 0),
            "shots": r.get("shots", 0),
            "xg": r.get("xg", 0.0),
            "matches": matches,
            "goals_per_match": (r["goals"] / matches) if matches > 0 else 0.0,
            "xg_per_shot": (r["xg"] / r["shots"]) if r["shots"] > 0 else 0.0
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["team"] = df["team"].astype(str).str.strip()
    return df


def build_team_logs(master_df):
    rows = []
    for _, r in master_df.iterrows():
        home = r.get("home_team")
        away = r.get("away_team")
        mid = r.get("id") or r.get("match_id")
        dt = r.get("datetime")
        hxg = r.get("home_xg")
        axg = r.get("away_xg")
        rows.append({
            "match_id": mid, "team": home, "opponent": away, "is_home": 1,
            "goals_for": r.get("home_goals"), "goals_against": r.get("away_goals"),
            "xg_for": hxg, "xg_against": axg, "date": dt
        })
        rows.append({
            "match_id": mid, "team": away, "opponent": home, "is_home": 0,
            "goals_for": r.get("away_goals"), "goals_against": r.get("home_goals"),
            "xg_for": axg, "xg_against": hxg, "date": dt
        })
    logs = pd.DataFrame(rows)
    if not logs.empty:
        logs["date"] = pd.to_datetime(logs["date"], errors="coerce")
        logs = logs.sort_values(["team", "date"]).reset_index(drop=True)
    return logs


def compute_rolling_features(team_logs, window=FORM_WINDOW):
    frames = []
    for team, g in team_logs.groupby("team", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
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
        g["rolling_xg"] = g["xg_for"].shift(1).rolling(window, min_periods=1).mean()
        frames.append(g)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_match_features(master_df, team_rolling):
    def last_before(team, date):
        sub = team_rolling[(team_rolling["team"] == team) & (team_rolling["date"] < date)]
        if sub.empty:
            sub = team_rolling[team_rolling["team"] == team]
            if sub.empty:
                return None
        return sub.iloc[-1]

    rows = []
    for _, r in master_df.iterrows():
        date = r.get("datetime")
        home = r.get("home_team")
        away = r.get("away_team")
        if pd.isna(home) or pd.isna(away):
            continue
        home_row = last_before(home, date)
        away_row = last_before(away, date)
        row = {
            "match_id": r.get("id"), "datetime": date, "season": r.get("season"), "event": r.get("event"),
            "home_team": home, "away_team": away,
            "home_goals": r.get("home_goals"), "away_goals": r.get("away_goals"),
            "home_xg_obs": r.get("home_xg"), "away_xg_obs": r.get("away_xg")
        }
        def fill(prefix, src):
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
        fill("home", home_row)
        fill("away", away_row)
        row["diff_rolling_points"] = row["home_rolling_points"] - row["away_rolling_points"]
        row["diff_rolling_gf"] = row["home_rolling_gf"] - row["away_rolling_gf"]
        row["diff_rolling_xg"] = row["home_rolling_xg"] - row["away_rolling_xg"]
        rows.append(row)
    feats = pd.DataFrame(rows)
    return feats.fillna(0.0)


def train_goal_models(feat_df, save_models=True):
    labeled = feat_df.dropna(subset=["home_goals", "away_goals"])
    if len(labeled) < 30:
        print("Warning: small training set; model quality may be limited.")
    feature_cols = [
        "home_rolling_points", "away_rolling_points",
        "home_rolling_gf", "away_rolling_gf",
        "home_rolling_xg", "away_rolling_xg",
        "diff_rolling_points", "diff_rolling_gf", "diff_rolling_xg"
    ]
    X = labeled[feature_cols].values
    y_home = labeled["home_goals"].astype(float).values
    y_away = labeled["away_goals"].astype(float).values

    X_train, X_val, y_train_home, y_val_home = train_test_split(X, y_home, test_size=0.12, random_state=RANDOM_SEED)
    _, _, y_train_away, y_val_away = train_test_split(X, y_away, test_size=0.12, random_state=RANDOM_SEED)

    model_home = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=500, max_depth=4, learning_rate=0.03, random_state=RANDOM_SEED)
    model_away = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=500, max_depth=4, learning_rate=0.03, random_state=RANDOM_SEED)

    model_home = safe_fit(model_home, X_train, y_train_home, X_val, y_val_home)
    model_away = safe_fit(model_away, X_train, y_train_away, X_val, y_val_away)

    if save_models:
        joblib.dump(model_home, MODELS_DIR / "xgb_home_goals.joblib")
        joblib.dump(model_away, MODELS_DIR / "xgb_away_goals.joblib")

    return model_home, model_away, feature_cols


def predict_expected_goals(models, feat_rows, feature_cols):
    model_home, model_away = models
    X = feat_rows[feature_cols].values
    pred_h = model_home.predict(X)
    pred_a = model_away.predict(X)
    pred_h = np.clip(pred_h, 0.03, 6.0)
    pred_a = np.clip(pred_a, 0.03, 6.0)
    out = feat_rows.copy()
    out["pred_home_xg"] = pred_h
    out["pred_away_xg"] = pred_a
    return out


def scoreline_probs(home_xg, away_xg, max_goals=MAX_GOALS_PER_TEAM):
    probs = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            probs[i, j] = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
    p_home = probs[np.triu_indices(max_goals + 1, k=1)].sum()
    p_draw = np.sum(np.diag(probs))
    p_away = probs[np.tril_indices(max_goals + 1, k=-1)].sum()
    idx = np.unravel_index(np.argmax(probs), probs.shape)
    most_likely = (int(idx[0]), int(idx[1]))
    return {"p_home_win": float(p_home), "p_draw": float(p_draw), "p_away_win": float(p_away), "most_likely_score": most_likely, "probs_matrix": probs}


def build_team_player_weights(player_stats_df, team_name, top_k=12, alpha_goals=1.0, beta_xg=1.2):
    if player_stats_df is None or player_stats_df.empty:
        return [], np.array([])
    team_players = player_stats_df[player_stats_df["team"].str.lower() == str(team_name).lower()].copy()
    if team_players.empty:
        team_players = player_stats_df[player_stats_df["team"].str.lower().str.contains(str(team_name).lower().split()[0])].copy()
    if team_players.empty:
        return [], np.array([])
    team_players["score_metric"] = team_players["goals_per_match"].fillna(0.0) * alpha_goals + team_players["xg_per_shot"].fillna(0.0) * beta_xg
    if team_players["score_metric"].sum() <= 0:
        team_players["score_metric"] = team_players["goals"].fillna(0.0) + 0.1 * team_players["shots"].fillna(0.0)
    team_players = team_players.sort_values("score_metric", ascending=False).head(top_k)
    players = team_players["player"].tolist()
    weights = team_players["score_metric"].values.astype(float)
    probs = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)
    return players, probs


def monte_carlo_scorers(home_xg, away_xg, home_players, home_probs, away_players, away_probs, n_sim=N_MC_SIM, max_goals=MAX_GOALS_PER_TEAM):
    rng = np.random.default_rng(RANDOM_SEED)
    score_counter = Counter()
    home_scorer_counter = Counter()
    away_scorer_counter = Counter()
    hp = np.array(home_probs) if len(home_probs) > 0 else np.array([])
    ap = np.array(away_probs) if len(away_probs) > 0 else np.array([])
    for _ in range(n_sim):
        gh = int(rng.poisson(home_xg))
        ga = int(rng.poisson(away_xg))
        gh = min(gh, max_goals)
        ga = min(ga, max_goals)
        score_counter[(gh, ga)] += 1
        if gh > 0 and len(home_players) > 0:
            picks = rng.choice(home_players, size=gh, p=hp)
            for p in picks:
                home_scorer_counter[p] += 1
        if ga > 0 and len(away_players) > 0:
            picks = rng.choice(away_players, size=ga, p=ap)
            for p in picks:
                away_scorer_counter[p] += 1
    total = n_sim
    score_probs = {f"{s[0]}-{s[1]}": count / total for s, count in score_counter.items()}
    home_scorer_probs = {p: cnt / total for p, cnt in home_scorer_counter.items()}
    away_scorer_probs = {p: cnt / total for p, cnt in away_scorer_counter.items()}
    top_home = sorted(home_scorer_probs.items(), key=lambda x: -x[1])[:6]
    top_away = sorted(away_scorer_probs.items(), key=lambda x: -x[1])[:6]
    p_home_win = sum(v for k, v in score_probs.items() if int(k.split('-')[0]) > int(k.split('-')[1]))
    p_draw = sum(v for k, v in score_probs.items() if int(k.split('-')[0]) == int(k.split('-')[1]))
    p_away_win = sum(v for k, v in score_probs.items() if int(k.split('-')[0]) < int(k.split('-')[1]))
    return {"score_probs": score_probs, "top_home_scorers": top_home, "top_away_scorers": top_away, "p_home_win_mc": p_home_win, "p_draw_mc": p_draw, "p_away_win_mc": p_away_win}


def main():
    print("Loading master matches...")
    master = safe_read_master(MASTER_FILE)
    print(f"Master rows: {len(master)}")

    print("Aggregating player stats from event JSONs...")
    player_stats = aggregate_player_stats(EVENTS_GLOB)
    if player_stats.empty:
        print("Warning: no event JSONs found; scorer prediction will fallback to team-level allocation.")

    print("Building team logs and rolling features...")
    team_logs = build_team_logs(master)
    team_rolling = compute_rolling_features(team_logs, window=FORM_WINDOW)

    print("Constructing match-level features...")
    feats = build_match_features(master, team_rolling)

    print("Training goal models...")
    model_home, model_away, feature_cols = train_goal_models(feats, save_models=True)

    upcoming = feats[(feats["home_goals"].isna()) | (feats["away_goals"].isna())].copy()
    now = pd.Timestamp.now()
    upcoming = upcoming[(upcoming["datetime"].isna()) | (pd.to_datetime(upcoming["datetime"]) <= now + pd.Timedelta(days=14))]

    if upcoming.empty:
        print("No upcoming fixtures to predict.")
        return

    print(f"Predicting for {len(upcoming)} upcoming fixtures...")
    pred_df = predict_expected_goals((model_home, model_away), upcoming, feature_cols)

    rows_out = []
    for _, r in tqdm(pred_df.iterrows(), total=len(pred_df), desc="Predict matches"):
        home = r["home_team"]
        away = r["away_team"]
        home_xg = float(r["pred_home_xg"])
        away_xg = float(r["pred_away_xg"])

        closed = scoreline_probs(home_xg, away_xg, max_goals=MAX_GOALS_PER_TEAM)
        most_likely = closed["most_likely_score"]

        home_players, home_probs = build_team_player_weights(player_stats, home)
        away_players, away_probs = build_team_player_weights(player_stats, away)

        mc = monte_carlo_scorers(home_xg, away_xg, home_players, home_probs, away_players, away_probs, n_sim=N_MC_SIM, max_goals=MAX_GOALS_PER_TEAM)

        rows_out.append({
            "match_id": r["match_id"],
            "datetime": r["datetime"],
            "season": r["season"],
            "event": r.get("event"),
            "home_team": home,
            "away_team": away,
            "pred_home_xg": home_xg,
            "pred_away_xg": away_xg,
            "most_likely_home": most_likely[0],
            "most_likely_away": most_likely[1],
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

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(OUTPUT_PREDICTIONS, index=False)
    print(f"Predictions saved to {OUTPUT_PREDICTIONS}")

    print("\nSummary:")
    for _, row in out_df.iterrows():
        print(f"{row['home_team']} vs {row['away_team']} — xG {row['pred_home_xg']:.2f}:{row['pred_away_xg']:.2f} — most likely {int(row['most_likely_home'])}-{int(row['most_likely_away'])}; p(H)={row['p_home_win_mc']:.2f} p(D)={row['p_draw_mc']:.2f} p(A)={row['p_away_win_mc']:.2f}")


if __name__ == "__main__":
    main()
