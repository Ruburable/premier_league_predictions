import requests
import pandas as pd
import os


OUTPUT_FOLDER = "data"

HISTORICAL_FILE = "historical_raw.csv"
CURRENT_FILE = "current_raw.csv"
SHOTS_FILE = "shots_xg_raw.csv"
ALL_FILE = "all_matches.csv"

HISTORICAL_SEASONS = [
    "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25"
]

CURRENT_SEASON = "2025-26"


# ----------------------------------------------------
# HISTORICAL (OpenFootball)
# ----------------------------------------------------

def download_openfootball(season):
    url = f"https://raw.githubusercontent.com/openfootball/football.json/master/{season}/en.1.json"
    r = requests.get(url)

    if r.status_code != 200:
        print(f"Historical season missing: {season}")
        return pd.DataFrame()

    data = r.json()["matches"]
    df = pd.DataFrame(data)

    # FIX 1 — rename team1/team2 to home_team/away_team
    df.rename(columns={
        "team1": "home_team",
        "team2": "away_team"
    }, inplace=True)

    # FIX 2 — handle missing score field safely
    df["home_goals"] = df["score"].apply(lambda s: s.get("ft", [None, None])[0] if isinstance(s, dict) else None)
    df["away_goals"] = df["score"].apply(lambda s: s.get("ft", [None, None])[1] if isinstance(s, dict) else None)

    # FIX 3 — guarantee date field exists
    if "date" not in df.columns:
        df["date"] = pd.NaT

    df["season"] = season
    return df



def load_historical():
    frames = []
    for s in HISTORICAL_SEASONS:
        print(f"Downloading historical season {s}")
        frames.append(download_openfootball(s))

    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------
# CURRENT SEASON (FPL API)
# ----------------------------------------------------

def get_team_lookup():
    r = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
    teams = r["teams"]
    df = pd.DataFrame(teams)[["id", "name"]]
    df.columns = ["team_id", "team_name"]
    return df


def load_current_season():
    fixtures = requests.get("https://fantasy.premierleague.com/api/fixtures/").json()
    df = pd.DataFrame(fixtures)
    teams = get_team_lookup()

    df = df.merge(teams, left_on="team_h", right_on="team_id", how="left")
    df.rename(columns={"team_name": "home_team"}, inplace=True)
    df = df.merge(teams, left_on="team_a", right_on="team_id", how="left")
    df.rename(columns={"team_name": "away_team"}, inplace=True)

    df["season"] = CURRENT_SEASON
    df["home_goals"] = df["team_h_score"]
    df["away_goals"] = df["team_a_score"]

    return df


# ----------------------------------------------------
# SHOTS / XG (Understat)
# ----------------------------------------------------

def load_understat(season):
    url = f"https://understatapi.onrender.com/league?league=EPL&season={season}"
    r = requests.get(url)

    if r.status_code != 200:
        print(f"Understat missing for {season}")
        return pd.DataFrame()

    matches = r.json()["matches"]
    df = pd.DataFrame(matches)

    df["season"] = season
    return df


def load_understat_all():
    frames = []
    for s in HISTORICAL_SEASONS + [CURRENT_SEASON]:
        print(f"Downloading xG for {s}")
        frames.append(load_understat(s))
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------
# COMBINE DATASETS
# ----------------------------------------------------

def build_master_table(historical, current, shots):
    # Minimal columns for predictions
    base_cols = [
        "season", "date", "home_team", "away_team",
        "home_goals", "away_goals"
    ]

    # Harmonize historical date column
    if "date" not in historical.columns and "matchday" in historical.columns:
        historical["date"] = pd.NaT

    # Create unified base
    all_base = pd.concat([
        historical[["season", "date", "home_team", "away_team", "home_goals", "away_goals"]],
        current[["season", "kickoff_time", "home_team", "away_team", "home_goals", "away_goals"]]
        .rename(columns={"kickoff_time": "date"})
    ], ignore_index=True)

    # Add xG/shots
    shots_subset = shots[
        ["season", "h_team", "a_team", "xG_h", "xG_a", "h_shots", "a_shots"]
    ].rename(columns={
        "h_team": "home_team",
        "a_team": "away_team",
        "xG_h": "home_xg",
        "xG_a": "away_xg",
        "h_shots": "home_shots",
        "a_shots": "away_shots"
    })

    merged = all_base.merge(
        shots_subset,
        on=["season", "home_team", "away_team"],
        how="left"
    )

    return merged


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------

if __name__ == "__main__":

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load components
    historical = load_historical()
    current = load_current_season()
    shots = load_understat_all()

    historical.to_csv(os.path.join(OUTPUT_FOLDER, HISTORICAL_FILE), index=False)
    current.to_csv(os.path.join(OUTPUT_FOLDER, CURRENT_FILE), index=False)
    shots.to_csv(os.path.join(OUTPUT_FOLDER, SHOTS_FILE), index=False)

    # Combine
    master = build_master_table(historical, current, shots)
    master.to_csv(os.path.join(OUTPUT_FOLDER, ALL_FILE), index=False)

    print("All data saved successfully.")
