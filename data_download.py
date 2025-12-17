#!/usr/bin/env python3
"""
Reliable Premier League match-level dataset
Source: FBref via soccerdata (API-safe version)
"""

import os
from pathlib import Path
import pandas as pd
import soccerdata as sd

# ------------------------------------------------------------------
# FORCE LOCAL CACHE + REAL USER AGENT
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "soccerdata_cache"

DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

os.environ["SOCCERDATA_DIR"] = str(CACHE_DIR)
os.environ["SOCCERDATA_USER_AGENT"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
LEAGUE = "ENG-Premier League"
SEASONS = list(range(2018, 2025))
OUTPUT_FILE = DATA_DIR / "matches_master.csv"

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
def load_fbref():
    print("Initializing FBref scraper...")
    print(f"Cache directory: {CACHE_DIR.resolve()}")

    fbref = sd.FBref(
        leagues=LEAGUE,
        seasons=SEASONS,
        data_dir=CACHE_DIR,
        no_cache=False
    )

    print("Downloading match schedule...")
    schedule = fbref.read_schedule()
    if schedule is None or schedule.empty:
        raise RuntimeError("Schedule download failed or returned empty")

    print("Downloading team shooting stats (xG)...")
    shooting = fbref.read_team_match_stats(stat_type="shooting")
    if shooting is None or shooting.empty:
        raise RuntimeError("Shooting stats download failed or returned empty")

    print("Download completed.")
    return schedule.reset_index(), shooting.reset_index()

# ------------------------------------------------------------------
# BUILD MATCH TABLE
# ------------------------------------------------------------------
def build_matches(schedule: pd.DataFrame, shooting: pd.DataFrame) -> pd.DataFrame:
    print("Building match-level dataset...")

    sched = schedule[
        [
            "game_id", "season", "date",
            "home_team", "away_team",
            "home_goals", "away_goals"
        ]
    ].copy()

    shoot = shooting[
        ["game_id", "is_home", "xg"]
    ].copy()

    home_xg = shoot[shoot["is_home"]].set_index("game_id")["xg"]
    away_xg = shoot[~shoot["is_home"]].set_index("game_id")["xg"]

    sched["home_xg"] = sched["game_id"].map(home_xg)
    sched["away_xg"] = sched["game_id"].map(away_xg)

    sched["datetime"] = pd.to_datetime(sched["date"], utc=True)

    for col in ["home_goals", "away_goals", "home_xg", "away_xg"]:
        sched[col] = pd.to_numeric(sched[col], errors="coerce")

    sched = sched.sort_values("datetime").reset_index(drop=True)

    return sched[
        [
            "season", "datetime",
            "home_team", "away_team",
            "home_goals", "away_goals",
            "home_xg", "away_xg"
        ]
    ]

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    print("=== FBREF DATA PIPELINE START ===")

    schedule, shooting = load_fbref()
    matches = build_matches(schedule, shooting)

    if matches.empty:
        raise RuntimeError("Final dataset is empty — aborting")

    matches.to_csv(OUTPUT_FILE, index=False)

    print("=== SUCCESS ===")
    print(f"Saved {len(matches)} matches to:")
    print(OUTPUT_FILE.resolve())
    print(matches.head())

if __name__ == "__main__":
    main()
