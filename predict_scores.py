import pandas as pd
import numpy as np
import ast
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

MASTER = Path("output/matches_master.csv")
OUT = Path("output/predictions_upcoming.csv")

def load_data():
    df = pd.read_csv(MASTER)

    # Force consistent datetimes (tz-naive)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce").dt.tz_localize(None)

    # Ensure unique match_id
    if "match_id" not in df.columns:
        df["match_id"] = df.index.astype(int)

    # Parse team dicts
    def parse_team(v):
        try:
            d = ast.literal_eval(v)
            return d.get("title", None)
        except:
            return v

    df["home_team_name"] = df["home_team"].apply(parse_team)
    df["away_team_name"] = df["away_team"].apply(parse_team)

    return df

def split_past_upcoming(df):
    now = pd.Timestamp.utcnow().tz_localize(None)
    past = (df["home_goals"].notna() & df["away_goals"].notna()) | \
           ((df["datetime"].notna()) & (df["datetime"] <= now))
    return df[past].copy(), df[~past].copy()

def build_team_events(past):
    rows = []
    for _, r in past.iterrows():
        if pd.isna(r.home_goals) or pd.isna(r.away_goals):
            continue

        # Home event
        rows.append({
            "match_id": r.match_id,
            "datetime": r.datetime,
            "team": r.home_team_name,
            "opp": r.away_team_name,
            "is_home": 1,
            "gf": r.home_goals,
            "ga": r.away_goals
        })
        # Away event
        rows.append({
            "match_id": r.match_id,
            "datetime": r.datetime,
            "team": r.away_team_name,
            "opp": r.home_team_name,
            "is_home": 0,
            "gf": r.away_goals,
            "ga": r.home_goals
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["team", "datetime"]).reset_index(drop=True)

    grp = df.groupby("team")

    # Rolling features (shifted)
    df["gf_roll"] = grp["gf"].rolling(5, min_periods=1).mean().shift(1).reset_index(level=0, drop=True).fillna(0)
    df["ga_roll"] = grp["ga"].rolling(5, min_periods=1).mean().shift(1).reset_index(level=0, drop=True).fillna(0)

    df["pts"] = np.where(df.gf > df.ga, 3, np.where(df.gf == df.ga, 1, 0))
    df["form"] = grp["pts"].rolling(5, min_periods=1).sum().shift(1).reset_index(level=0, drop=True).fillna(0)

    df["days_rest"] = (df["datetime"] - grp["datetime"].shift(1)).dt.days.fillna(15)

    return df

def attach_features(matches, events):
    if events.empty:
        raise ValueError("Events table is empty. Check match_id creation and goal columns.")

    home = events[events.is_home == 1].set_index("match_id")
    away = events[events.is_home == 0].set_index("match_id")

    feats = ["gf_roll","ga_roll","form","days_rest"]

    H = home[feats].add_suffix("_home")
    A = away[feats].add_suffix("_away")

    m = matches.set_index("match_id").join(H).join(A).reset_index()

    # Diffs + home advantage
    m["gf_diff"] = m["gf_roll_home"] - m["gf_roll_away"]
    m["ga_diff"] = m["ga_roll_home"] - m["ga_roll_away"]
    m["form_diff"] = m["form_home"] - m["form_away"]
    m["home_adv"] = 1

    return m.fillna(0)

def train_model(X, y):
    tscv = TimeSeriesSplit(5)
    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    for tr, te in tscv.split(X):
        model.fit(X.iloc[tr], y.iloc[tr],
                  eval_set=[(X.iloc[te], y.iloc[te])],
                  early_stopping_rounds=20,
                  verbose=False)

    model.fit(X, y)
    return model

def main():
    df = load_data()
    past, upcoming = split_past_upcoming(df)

    events = build_team_events(past)
    train = attach_features(past, events)

    cols = [
        'gf_roll_home','ga_roll_home','form_home','days_rest_home',
        'gf_roll_away','ga_roll_away','form_away','days_rest_away',
        'gf_diff','ga_diff','form_diff','home_adv'
    ]

    home_model = train_model(train[cols], train['home_goals'])
    away_model = train_model(train[cols], train['away_goals'])

    up = attach_features(upcoming, events)
    up["pred_home_goals"] = home_model.predict(up[cols]).clip(0)
    up["pred_away_goals"] = away_model.predict(up[cols]).clip(0)

    up.to_csv(OUT, index=False)
    print("Saved predictions to:", OUT)

if __name__ == "__main__":
    main()