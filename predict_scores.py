#!/usr/bin/env python3
"""
predict_scores_fe_fixed.py

Improved version adapted to the actual structure of your uploaded
output/matches_master.csv. It:
 - parses team columns that are stored as stringified dicts
 - includes rolling feature engineering (goals for/against, points, form)
 - uses available expected-goals (home_xg / away_xg) as features
 - is robust against empty intermediate tables (avoids KeyError: 'team')
 - trains time-aware XGBoost regressors and predicts upcoming fixtures

Inputs:
  - output/matches_master.csv

Outputs:
  - output/predictions_upcoming.csv
"""
import ast
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

MASTER_CSV = Path("output/matches_master.csv")
OUTPUT_PRED = Path("output/predictions_upcoming.csv")
RANDOM_STATE = 42

def load_master(path=MASTER_CSV):
    if not path.exists():
        raise FileNotFoundError(f"Master file not found: {path}")
    df = pd.read_csv(path)
    # parse datetime
    df['datetime'] = pd.to_datetime(df.get('datetime'), errors='coerce')
    # ensure consistent column names (lower)
    df.columns = [c for c in df.columns]
    # reset index into a stable match_id column if not present
    if 'match_id' not in df.columns and 'id' in df.columns:
        df = df.rename(columns={'id': 'match_id'})
    elif 'match_id' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'match_id'})

    # parse team columns (they are string representations of dicts in your file)
    def parse_team_field(v):
        if pd.isna(v):
            return {'id': None, 'title': None, 'short_title': None}
        if isinstance(v, dict):
            return v
        try:
            # safe literal eval
            d = ast.literal_eval(v)
            if isinstance(d, dict):
                return d
            else:
                return {'id': None, 'title': str(v), 'short_title': None}
        except Exception:
            # fallback: return as title string
            return {'id': None, 'title': str(v), 'short_title': None}

    # get team_id and team_name
    for side in ['home', 'away']:
        col = f"{side}_team"
        if col in df.columns:
            parsed = df[col].apply(parse_team_field)
            df[f"{col}_id_parsed"] = parsed.apply(lambda d: d.get('id') if isinstance(d, dict) else None)
            df[f"{col}_name_parsed"] = parsed.apply(lambda d: d.get('title') if isinstance(d, dict) else str(d))
        else:
            raise KeyError(f"Expected column '{col}' in matches master CSV.")

    return df

def split_past_upcoming(master_df):
    now = pd.Timestamp.utcnow()
    cond_past = (master_df['home_goals'].notna() & master_df['away_goals'].notna()) | \
                (master_df['datetime'].notna() & (master_df['datetime'] <= now))
    past_df = master_df[cond_past].copy()
    upcoming_df = master_df[~cond_past].copy()
    print(f"Loaded {len(master_df)} matches total.")
    print(f"Past matches: {len(past_df)}")
    print(f"Upcoming fixtures: {len(upcoming_df)}")
    return past_df, upcoming_df

def build_team_events(past_df):
    """
    Convert match-level rows into team-event-level rows (one row per team per match)
    and include parsed team names/ids + xG where available.
    """
    events = []
    # iterate rows - only finished matches should be in past_df
    for _, row in past_df.iterrows():
        # ensure goals exist
        if pd.isna(row.get('home_goals')) or pd.isna(row.get('away_goals')):
            continue

        mid = row['match_id']
        dt = row['datetime']
        # use parsed team name if available, else fallback to raw column
        home_team = row.get('home_team_name_parsed') or row.get('home_team')
        away_team = row.get('away_team_name_parsed') or row.get('away_team')

        try:
            hg = int(row['home_goals'])
            ag = int(row['away_goals'])
        except Exception:
            # skip if goals not integer-like
            continue

        # xg values (may exist)
        hxg = row.get('home_xg', np.nan)
        axg = row.get('away_xg', np.nan)

        # points
        if hg > ag:
            home_pts, away_pts = 3, 0
        elif hg == ag:
            home_pts, away_pts = 1, 1
        else:
            home_pts, away_pts = 0, 3

        # home event
        events.append({
            'match_id': mid,
            'datetime': dt,
            'team': home_team,
            'team_id_parsed': row.get('home_team_id_parsed'),
            'opponent': away_team,
            'is_home': True,
            'goals_for': hg,
            'goals_against': ag,
            'points': home_pts,
            'xg_for': hxg,
            'xg_against': axg
        })
        # away event
        events.append({
            'match_id': mid,
            'datetime': dt,
            'team': away_team,
            'team_id_parsed': row.get('away_team_id_parsed'),
            'opponent': home_team,
            'is_home': False,
            'goals_for': ag,
            'goals_against': hg,
            'points': away_pts,
            'xg_for': axg,
            'xg_against': hxg
        })

    events_df = pd.DataFrame(events)
    # If events_df is empty, return empty but with expected columns to avoid KeyErrors downstream
    if events_df.empty:
        cols = ['match_id','datetime','team','team_id_parsed','opponent','is_home','goals_for','goals_against','points','xg_for','xg_against']
        events_df = pd.DataFrame(columns=cols)
    # ensure chronological order per team
    if 'team' in events_df.columns and 'datetime' in events_df.columns:
        events_df = events_df.sort_values(['team', 'datetime', 'match_id']).reset_index(drop=True)
    return events_df

def compute_rolling_features(events_df, window=5):
    """
    Compute rolling features per team and shift them by 1 so features represent
    only past matches (exclude the current event).
    """
    df = events_df.copy()
    if df.empty:
        # return empty with expected cols
        return df

    # groupby team sorted by datetime
    df = df.sort_values(['team','datetime']).reset_index(drop=True)

    # rolling means (shifted)
    df['gf_roll_mean'] = df.groupby('team')['goals_for'].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)
    df['ga_roll_mean'] = df.groupby('team')['goals_against'].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)
    df['points_roll_sum'] = df.groupby('team')['points'].rolling(window=window, min_periods=1).sum().reset_index(level=0, drop=True).shift(1)

    # xg rolling features (if xg available)
    if 'xg_for' in df.columns:
        df['xg_roll_mean'] = df.groupby('team')['xg_for'].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)
    else:
        df['xg_roll_mean'] = np.nan

    # days since last match and last goals
    df['last_match_datetime'] = df.groupby('team')['datetime'].shift(1)
    df['days_since_last'] = (df['datetime'] - df['last_match_datetime']).dt.total_seconds() / (24*3600)
    df['days_since_last'] = df['days_since_last'].fillna(999)
    df['last_goals'] = df.groupby('team')['goals_for'].shift(1).fillna(0).astype(int)

    # form per match (normalize by matches played prior)
    df['matches_played_prior'] = df.groupby('team').cumcount()
    # avoid division by zero
    df['form_points_per_match'] = df['points_roll_sum'] / df['matches_played_prior'].replace(0, np.nan)
    df['form_points_per_match'] = df['form_points_per_match'].fillna(0.0)

    # fill NaNs for newly seen teams with sensible defaults (zeros)
    df['gf_roll_mean'] = df['gf_roll_mean'].fillna(0.0)
    df['ga_roll_mean'] = df['ga_roll_mean'].fillna(0.0)
    df['points_roll_sum'] = df['points_roll_sum'].fillna(0.0)
    df['xg_roll_mean'] = df['xg_roll_mean'].fillna(0.0)

    return df

def make_match_level_features(past_matches_df, events_rolled_df):
    """
    For each finished match, attach the prior rolling features for home and away teams.
    """
    if events_rolled_df.empty:
        raise ValueError("No team-event rows available to build features. Check past matches and goals columns.")

    home_feats = events_rolled_df[events_rolled_df['is_home']].set_index('match_id', drop=False)
    away_feats = events_rolled_df[~events_rolled_df['is_home']].set_index('match_id', drop=False)

    feat_cols = ['gf_roll_mean','ga_roll_mean','points_roll_sum','form_points_per_match','days_since_last','last_goals','matches_played_prior','xg_roll_mean']

    home_merge = home_feats[feat_cols].add_suffix('_home')
    away_merge = away_feats[feat_cols].add_suffix('_away')

    matches = past_matches_df.set_index('match_id')
    merged = matches.join(home_merge, how='left').join(away_merge, how='left')

    # home advantage
    merged['home_adv'] = 1

    # fill missing with medians
    for c in merged.columns:
        if c.endswith('_home') or c.endswith('_away'):
            merged[c] = merged[c].fillna(merged[c].median())

    # derived differences
    merged['gf_diff_5'] = merged['gf_roll_mean_home'] - merged['gf_roll_mean_away']
    merged['ga_diff_5'] = merged['ga_roll_mean_home'] - merged['ga_roll_mean_away']
    merged['form_diff'] = merged['form_points_per_match_home'] - merged['form_points_per_match_away']
    merged['xg_diff'] = merged['xg_roll_mean_home'] - merged['xg_roll_mean_away']

    # ensure numeric target
    merged['home_goals'] = merged['home_goals'].astype(float)
    merged['away_goals'] = merged['away_goals'].astype(float)

    return merged.reset_index()

def prepare_upcoming_features(upcoming_df, events_rolled_df):
    """
    For upcoming fixtures, obtain most recent rolling stats per team and merge them.
    """
    # if no prior events exist, create defaults
    if events_rolled_df.empty:
        # create default dataframe with medians = 0
        upcoming = upcoming_df.copy()
        # make basic placeholder cols
        cols = ['gf_roll_mean_home','ga_roll_mean_home','points_roll_sum_home','form_points_per_match_home','days_since_last_home','last_goals_home','matches_played_prior_home','xg_roll_mean_home',
                'gf_roll_mean_away','ga_roll_mean_away','points_roll_sum_away','form_points_per_match_away','days_since_last_away','last_goals_away','matches_played_prior_away','xg_roll_mean_away']
        for c in cols:
            upcoming[c] = 0.0
        upcoming['home_adv'] = 1
        upcoming['gf_diff_5'] = 0.0
        upcoming['ga_diff_5'] = 0.0
        upcoming['form_diff'] = 0.0
        upcoming['xg_diff'] = 0.0
        return upcoming

    latest = events_rolled_df.sort_values('datetime').groupby('team').last()
    feat_cols = ['gf_roll_mean','ga_roll_mean','points_roll_sum','form_points_per_match','days_since_last','last_goals','matches_played_prior','xg_roll_mean']
    latest = latest[feat_cols].rename(columns=lambda x: x + '_last')

    upcoming = upcoming_df.copy()
    # merge home
    upcoming = upcoming.merge(latest.add_suffix('_home'), left_on='home_team_name_parsed', right_on='team_home', how='left')
    # merge away
    upcoming = upcoming.merge(latest.add_suffix('_away'), left_on='away_team_name_parsed', right_on='team_away', how='left')
    # drop helper team columns
    cols_to_drop = [c for c in upcoming.columns if c.startswith('team_')]
    upcoming = upcoming.drop(columns=cols_to_drop, errors='ignore')

    # map last->standard names
    rename_map = {
        'gf_roll_mean_last_home': 'gf_roll_mean_home',
        'ga_roll_mean_last_home': 'ga_roll_mean_home',
        'points_roll_sum_last_home': 'points_roll_sum_home',
        'form_points_per_match_last_home': 'form_points_per_match_home',
        'days_since_last_last_home': 'days_since_last_home',
        'last_goals_last_home': 'last_goals_home',
        'matches_played_prior_last_home': 'matches_played_prior_home',
        'xg_roll_mean_last_home': 'xg_roll_mean_home',

        'gf_roll_mean_last_away': 'gf_roll_mean_away',
        'ga_roll_mean_last_away': 'ga_roll_mean_away',
        'points_roll_sum_last_away': 'points_roll_sum_away',
        'form_points_per_match_last_away': 'form_points_per_match_away',
        'days_since_last_last_away': 'days_since_last_away',
        'last_goals_last_away': 'last_goals_away',
        'matches_played_prior_last_away': 'matches_played_prior_away',
        'xg_roll_mean_last_away': 'xg_roll_mean_away',
    }
    upcoming = upcoming.rename(columns=rename_map)

    # fill missing with medians
    for c in upcoming.columns:
        if c.endswith('_home') or c.endswith('_away'):
            upcoming[c] = upcoming[c].fillna(upcoming[c].median())

    # add interactions
    upcoming['home_adv'] = 1
    upcoming['gf_diff_5'] = upcoming['gf_roll_mean_home'] - upcoming['gf_roll_mean_away']
    upcoming['ga_diff_5'] = upcoming['ga_roll_mean_home'] - upcoming['ga_roll_mean_away']
    upcoming['form_diff'] = upcoming['form_points_per_match_home'] - upcoming['form_points_per_match_away']
    upcoming['xg_diff'] = upcoming['xg_roll_mean_home'] - upcoming['xg_roll_mean_away']

    return upcoming

def train_and_evaluate(X, y):
    """
    Train with TimeSeriesSplit CV to get a CV estimate, then fit final model on all training data.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    cv_rmse = []
    cv_mae = []

    X_np = X.values
    y_np = y.values

    for train_idx, test_idx in tscv.split(X_np):
        X_tr, X_te = X_np[train_idx], X_np[test_idx]
        y_tr, y_te = y_np[train_idx], y_np[test_idx]

        m = XGBRegressor(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            verbosity=0,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], early_stopping_rounds=25, verbose=False)
        preds = m.predict(X_te)
        cv_rmse.append(np.sqrt(mean_squared_error(y_te, preds)))
        cv_mae.append(mean_absolute_error(y_te, preds))

    metrics = {
        'cv_rmse_mean': float(np.mean(cv_rmse)) if cv_rmse else None,
        'cv_mae_mean': float(np.mean(cv_mae)) if cv_mae else None
    }

    # final fit
    final_model = XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    final_model.fit(X, y, verbose=False)
    return final_model, metrics

def predict_top_scorers(upcoming_df, past_df, top_n=3):
    # simple heuristic if scorer columns exist
    top_map = {}
    if 'home_scorers' in past_df.columns and 'away_scorers' in past_df.columns:
        def norm(val):
            if pd.isna(val):
                return []
            if isinstance(val, list):
                return val
            if isinstance(val, str):
                return [s.strip() for s in val.split(',') if s.strip()]
            return []
        counts = {}
        for _, r in past_df.iterrows():
            for side in ['home','away']:
                team = r.get(f"{side}_team_name_parsed")
                sc = norm(r.get(f"{side}_scorers", []))
                if not team:
                    continue
                counts.setdefault(team, {})
                for s in sc:
                    counts[team][s] = counts[team].get(s,0) + 1
        for team, d in counts.items():
            top_map[team] = [s for s,_ in sorted(d.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    # map
    upcoming_df['top_home_scorers'] = upcoming_df['home_team_name_parsed'].map(lambda t: top_map.get(t, []))
    upcoming_df['top_away_scorers'] = upcoming_df['away_team_name_parsed'].map(lambda t: top_map.get(t, []))
    return upcoming_df

def main():
    master_df = load_master()
    past_df, upcoming_df = split_past_upcoming(master_df)

    if len(past_df) < 5:
        print("⚠ Not enough past data. Using heuristics.")
        upcoming_df['pred_home_goals'] = np.random.uniform(0.8,2.2,len(upcoming_df))
        upcoming_df['pred_away_goals'] = np.random.uniform(0.8,2.2,len(upcoming_df))
        upcoming_df['top_home_scorers'] = [[] for _ in range(len(upcoming_df))]
        upcoming_df['top_away_scorers'] = [[] for _ in range(len(upcoming_df))]
        upcoming_df.to_csv(OUTPUT_PRED, index=False)
        print(f"✔ Saved upcoming predictions → {OUTPUT_PRED}")
        return

    print("--- Building team events and rolling features ---")
    events = build_team_events(past_df)
    events_rolled = compute_rolling_features(events, window=5)

    print("--- Building match-level training features ---")
    train_merged = make_match_level_features(past_df, events_rolled)

    feature_cols = [
        'gf_roll_mean_home','ga_roll_mean_home','points_roll_sum_home','form_points_per_match_home',
        'days_since_last_home','last_goals_home','matches_played_prior_home','xg_roll_mean_home',
        'gf_roll_mean_away','ga_roll_mean_away','points_roll_sum_away','form_points_per_match_away',
        'days_since_last_away','last_goals_away','matches_played_prior_away','xg_roll_mean_away',
        'home_adv','gf_diff_5','ga_diff_5','form_diff','xg_diff'
    ]

    # sort by datetime for time-aware CV
    train_merged = train_merged.sort_values('datetime').reset_index(drop=True)

    X_home = train_merged[feature_cols]
    y_home = train_merged['home_goals']
    X_away = train_merged[feature_cols]
    y_away = train_merged['away_goals']

    print("--- Training home goals model ---")
    model_home, metrics_home = train_and_evaluate(X_home, y_home)
    print("Home model CV RMSE:", metrics_home['cv_rmse_mean'], "MAE:", metrics_home['cv_mae_mean'])

    print("--- Training away goals model ---")
    model_away, metrics_away = train_and_evaluate(X_away, y_away)
    print("Away model CV RMSE:", metrics_away['cv_rmse_mean'], "MAE:", metrics_away['cv_mae_mean'])

    print("--- Preparing upcoming fixture features ---")
    upcoming_feats = prepare_upcoming_features(upcoming_df, events_rolled)

    if upcoming_feats.empty:
        print("No upcoming fixtures to predict. Exiting.")
        return

    X_upcoming = upcoming_feats[feature_cols]
    upcoming_feats['pred_home_goals'] = model_home.predict(X_upcoming).clip(0).round(2)
    upcoming_feats['pred_away_goals'] = model_away.predict(X_upcoming).clip(0).round(2)

    upcoming_feats = predict_top_scorers(upcoming_feats, past_df, top_n=3)

    upcoming_feats.to_csv(OUTPUT_PRED, index=False)
    print(f"✔ Saved upcoming predictions → {OUTPUT_PRED}")

if __name__ == "__main__":
    main()