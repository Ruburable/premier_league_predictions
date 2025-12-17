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
def build_matches(schedule: pd.DataFrame) -> pd.DataFrame:
    print("Building match-level dataset...")
    print("Schedule columns:", list(schedule.columns))

    df = schedule.copy()

    # ------------------------------------------------------------
    # Parse goals from score (e.g. "2–1")
    # ------------------------------------------------------------
    if "score" not in df.columns:
        raise RuntimeError("FBref schedule missing 'score' column")

    scores = df["score"].astype(str).str.split("–", expand=True)
    df["home_goals"] = pd.to_numeric(scores[0], errors="coerce")
    df["away_goals"] = pd.to_numeric(scores[1], errors="coerce")

    # ------------------------------------------------------------
    # Validate xG columns
    # ------------------------------------------------------------
    if not {"home_xg", "away_xg"}.issubset(df.columns):
        raise RuntimeError(
            "FBref schedule missing home_xg / away_xg columns"
        )

    # ------------------------------------------------------------
    # Datetime
    # ------------------------------------------------------------
    df["datetime"] = pd.to_datetime(
        df["date"],
        errors="coerce",
        utc=True
    )

    # ------------------------------------------------------------
    # Final selection
    # ------------------------------------------------------------
    matches = df[
        [
            "season",
            "datetime",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "home_xg",
            "away_xg",
        ]
    ].sort_values("datetime").reset_index(drop=True)

    return matches

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    print("=== FBREF DATA PIPELINE START ===")

    schedule, shooting = load_fbref()
    matches = build_matches(schedule)

    if matches.empty:
        raise RuntimeError("Final dataset is empty — aborting")

    matches.to_csv(OUTPUT_FILE, index=False)

    print("=== SUCCESS ===")
    print(f"Saved {len(matches)} matches to:")
    print(OUTPUT_FILE.resolve())
    print(matches.head())

if __name__ == "__main__":
    main()