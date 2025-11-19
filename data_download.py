#!/usr/bin/env python3
"""
Unified Premier League data downloader.

Downloads:
- Historical Understat match list (2018–2025)
- Current season (2025/26) match list incrementally
- Match event JSONs (shots, xG, key passes etc.)

Features:
- Skips already-downloaded matches
- Skips matches whose dates are in the future
- Automatically normalizes week/round column
- Saves single master CSV for all seasons combined
"""

import os
import json
import time
import requests
import pandas as pd
from pathlib import Path

# ======================================================================
# CONFIGURATION
# ======================================================================

SEASON_CURRENT = "2025"     # Understat format for 2025/26 season
LEAGUE = "EPL"

BASE_DIR = Path("data")
MATCH_LIST_FILE = BASE_DIR / "matches_all.csv"
EVENTS_DIR = BASE_DIR / "match_events"
HISTORICAL_FILE = BASE_DIR / "historical_2018_2025.csv"

BASE_DIR.mkdir(exist_ok=True)
EVENTS_DIR.mkdir(exist_ok=True)

URL_LEAGUE = "https://understat.com/league/{league}/{season}"
URL_MATCH = "https://understat.com/match/{match_id}"


# ======================================================================
# HELPERS
# ======================================================================

def extract_json_from_understat(html_text):
    """Extract JSON from Understat HTML."""
    import re

    pattern = re.compile(r"JSON\.parse\('([^']+)'\)")
    m = pattern.search(html_text)
    if not m:
        return None

    raw = m.group(1)
    raw = raw.encode("utf-8").decode("unicode_escape")
    return json.loads(raw)


def polite_delay():
    time.sleep(0.7)


def normalize_week_column(df):
    """Ensure df contains 'week'. Convert 'round' -> 'week' when necessary."""
    if "week" in df.columns:
        df["week"] = pd.to_numeric(df["week"], errors="coerce")
        return df

    if "round" in df.columns:
        df = df.rename(columns={"round": "week"})
        df["week"] = pd.to_numeric(df["week"], errors="coerce")
        return df

    raise KeyError("ERROR: Neither 'week' nor 'round' column found in Understat dataset.")


def load_existing_match_list():
    if MATCH_LIST_FILE.exists():
        return pd.read_csv(MATCH_LIST_FILE)
    return pd.DataFrame()


def latest_completed_week(df):
    """Return the highest completed gameweek in the dataset."""
    if df.empty or "goals_h" not in df.columns:
        return 0

    df["datetime"] = pd.to_datetime(df.get("datetime"), errors="coerce")

    completed = df[df["goals_h"].notna() & df["goals_a"].notna()]
    if completed.empty:
        return 0

    return int(completed["week"].max())


# ======================================================================
# HISTORICAL (2018–2025)
# ======================================================================

def download_historical():
    """Download 2018–2025 historical match lists from Understat."""
    if HISTORICAL_FILE.exists():
        print("Historical dataset already present — skipping download.")
        return pd.read_csv(HISTORICAL_FILE)

    seasons = ["2018", "2019", "2020", "2021", "2022", "2023", "2024"]
    all_frames = []

    for season in seasons:
        print(f"Downloading historical season {season}...")

        url = URL_LEAGUE.format(league=LEAGUE, season=season)
        polite_delay()
        r = requests.get(url)
        data = extract_json_from_understat(r.text)

        if not data:
            print(f"Warning: No data found for season {season}")
            continue

        df = pd.DataFrame(data)
        df = normalize_week_column(df)
        df["season"] = season
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

        all_frames.append(df)

    if not all_frames:
        raise ValueError("Historical download failed — no seasons retrieved.")

    hist = pd.concat(all_frames, ignore_index=True)
    hist.to_csv(HISTORICAL_FILE, index=False)
    print(f"Saved → {HISTORICAL_FILE}")

    return hist


# ======================================================================
# CURRENT SEASON MATCH LIST
# ======================================================================

def download_current_match_list():
    print("Downloading current season match list...")
    url = URL_LEAGUE.format(league=LEAGUE, season=SEASON_CURRENT)

    polite_delay()
    r = requests.get(url)

    data = extract_json_from_understat(r.text)
    if not data:
        raise ValueError("Failed to parse current season JSON from Understat")

    df = pd.DataFrame(data)
    df = normalize_week_column(df)

    df["season"] = SEASON_CURRENT
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    return df


# ======================================================================
# MATCH EVENTS
# ======================================================================

def download_events(match_id):
    """Download per-match JSON if not already downloaded."""
    fpath = EVENTS_DIR / f"{match_id}.json"

    if fpath.exists():
        return

    polite_delay()
    url = URL_MATCH.format(match_id=match_id)
    r = requests.get(url)

    data = extract_json_from_understat(r.text)
    if not data:
        print(f"Skipping match {match_id} — no event JSON yet.")
        return

    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved → {fpath}")


# ======================================================================
# INCREMENTAL UPDATE FOR CURRENT SEASON
# ======================================================================

def update_current_season(existing_df):
    print("Updating 2025/26 season...")

    full_current = download_current_match_list()
    today = pd.Timestamp.today().normalize()

    # Real completed matches
    past_matches = full_current[full_current["datetime"] < today]
    real_latest_week = int(past_matches["week"].max())

    # Already downloaded weeks
    existing_latest_week = latest_completed_week(existing_df)

    print(f"Real-world completed GW: {real_latest_week}")
    print(f"Previously downloaded GW: {existing_latest_week}")

    # Missing matches
    missing = past_matches[past_matches["week"] > existing_latest_week]
    print(f"Matches to download this run: {len(missing)}")

    for _, row in missing.iterrows():
        mid = row["id"]
        w = row["week"]
        print(f"Downloading events for Match ID {mid} (GW {w})")
        download_events(mid)

    # Combine / clean match list
    combined = (
        pd.concat([existing_df, full_current], ignore_index=True)
        .drop_duplicates(subset=["id"], keep="last")
        .sort_values("id")
    )

    combined.to_csv(MATCH_LIST_FILE, index=False)
    print(f"Saved updated match list → {MATCH_LIST_FILE}")

    return combined


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=== Premier League Full Data Downloader ===")

    print("Step 1: Downloading historical database...")
    historical = download_historical()

    print("\nStep 2: Loading existing match list...")
    existing = load_existing_match_list()

    print("\nStep 3: Updating current season incrementally...")
    updated = update_current_season(existing)

    print("\n✓ DONE — All data up to date.")


if __name__ == "__main__":
    main()
