import requests
import pandas as pd
import os


# =====================================================
# CONFIGURATION
# =====================================================

OUTPUT_FOLDER = "data"

HISTORICAL_FILE = "historical_premier_league.csv"
CURRENT_FILE = "premier_league_2025_26.csv"

# Historical seasons to download
HISTORICAL_SEASONS = [
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25"
]

# Current season
CURRENT_SEASON = "2025-26"

# URLs
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
FPL_TEAMS_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
OPENFOOTBALL_BASE = (
    "https://raw.githubusercontent.com/openfootball/football.json/master/{season}/en.1.json"
)


# =====================================================
# HISTORICAL DATA LOADING
# =====================================================

def download_historical_season(season):
    """Download a single season from OpenFootball JSON."""
    url = OPENFOOTBALL_BASE.format(season=season)
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Skipping season {season}: not available from OpenFootball.")
        return pd.DataFrame()

    data = response.json()
    matches = data.get("matches", [])

    df = pd.DataFrame(matches)
    if df.empty:
        return df

    # Expand score fields
    df["home_goals"] = df["score"].apply(lambda s: s.get("ft", [None, None])[0])
    df["away_goals"] = df["score"].apply(lambda s: s.get("ft", [None, None])[1])

    df["season"] = season
    return df


def load_all_historical():
    """Load all historical seasons into a single DataFrame."""
    frames = []
    for season in HISTORICAL_SEASONS:
        print(f"Downloading historical season {season}...")
        df = download_historical_season(season)
        frames.append(df)

    historical = pd.concat(frames, ignore_index=True)
    return historical


# =====================================================
# CURRENT SEASON FROM FPL
# =====================================================

def get_team_lookup():
    """Returns team ID → name mapping."""
    teams = requests.get(FPL_TEAMS_URL).json()["teams"]
    df = pd.DataFrame(teams)[["id", "name"]]
    df.columns = ["team_id", "team_name"]
    return df


def download_current_season():
    """Download current season fixtures/results from FPL API."""
    fixtures = requests.get(FPL_FIXTURES_URL).json()
    df = pd.DataFrame(fixtures)

    if df.empty:
        return df

    # Add readable score fields
    df["home_goals"] = df["team_h_score"]
    df["away_goals"] = df["team_a_score"]

    # Add team names
    teams = get_team_lookup()
    df = df.merge(teams, left_on="team_h", right_on="team_id", how="left")
    df = df.rename(columns={"team_name": "home_team"}).drop(columns=["team_id"])

    df = df.merge(teams, left_on="team_a", right_on="team_id", how="left")
    df = df.rename(columns={"team_name": "away_team"}).drop(columns=["team_id"])

    df["season"] = CURRENT_SEASON

    # Clean columns
    df = df[
        [
            "id",
            "event",
            "kickoff_time",
            "finished",
            "team_h",
            "home_team",
            "team_a",
            "away_team",
            "home_goals",
            "away_goals",
            "season",
        ]
    ]

    return df


# =====================================================
# FILE MANAGEMENT
# =====================================================

def load_csv(path):
    """Load CSV if it exists."""
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def data_changed(old_df, new_df):
    """Return True if downloaded data contains new info."""
    if old_df.empty:
        return True

    if len(new_df) != len(old_df):
        return True

    merged = old_df.merge(new_df, how="outer", indicator=True)
    if any(merged["_merge"] != "both"):
        return True

    return False


def save_csv(df, path):
    """Save DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


# =====================================================
# MAIN UPDATE ROUTINES
# =====================================================

def update_historical():
    """Downloads historical seasons and saves them separately."""
    path = os.path.join(OUTPUT_FOLDER, HISTORICAL_FILE)
    old = load_csv(path)

    historical = load_all_historical()

    if data_changed(old, historical):
        save_csv(historical, path)
        print("Historical dataset updated.")
    else:
        print("Historical dataset is already up to date.")


def update_current_season():
    """Downloads current 25/26 season and saves separately."""
    path = os.path.join(OUTPUT_FOLDER, CURRENT_FILE)
    old = load_csv(path)

    current = download_current_season()

    if data_changed(old, current):
        save_csv(current, path)
        print("Current season dataset updated.")
    else:
        print("Current season dataset is already up to date.")


# =====================================================
# MAIN ENTRY POINT
# =====================================================

if __name__ == "__main__":
    update_historical()
    update_current_season()
