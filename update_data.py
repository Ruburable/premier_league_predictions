#!/usr/bin/env python3
"""
update_data.py

Unified data pipeline that:
1. Downloads/updates historical match data from FBref
2. Identifies which fixtures are upcoming based on current date
3. Estimates xG for upcoming matches based on recent form
4. Produces two outputs:
   - data/matches_master.csv (all historical matches with results)
   - data/upcoming_fixtures.csv (upcoming matches with estimated xG)
"""

import os
from pathlib import Path
import pandas as pd
import soccerdata as sd
from datetime import datetime, timezone, timedelta

# ------------------------------------------------------------------
# CONFIGURATION
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

LEAGUE = "ENG-Premier League"
SEASONS = list(range(2018, 2026))  # Adjust end year as needed
HISTORICAL_OUTPUT = DATA_DIR / "matches_master.csv"
UPCOMING_OUTPUT = DATA_DIR / "upcoming_fixtures.csv"


# ------------------------------------------------------------------
# DOWNLOAD DATA FROM FBREF
# ------------------------------------------------------------------
def download_fbref_data():
    """Download schedule and stats from FBref."""
    print("=" * 80)
    print("DOWNLOADING DATA FROM FBREF")
    print("=" * 80)
    print(f"\nLeague: {LEAGUE}")
    print(f"Seasons: {SEASONS[0]} to {SEASONS[-1]}")
    print(f"Cache directory: {CACHE_DIR.resolve()}\n")

    fbref = sd.FBref(
        leagues=LEAGUE,
        seasons=SEASONS,
        data_dir=CACHE_DIR,
        no_cache=False  # Use cache when available
    )

    print("Downloading match schedule...")
    schedule = fbref.read_schedule()
    if schedule is None or schedule.empty:
        raise RuntimeError("Schedule download failed or returned empty")

    print(f"  ✓ Downloaded {len(schedule)} matches")

    return schedule.reset_index()


# ------------------------------------------------------------------
# PROCESS AND SPLIT DATA
# ------------------------------------------------------------------
def process_schedule(schedule: pd.DataFrame):
    """
    Process schedule and split into historical (with results) and upcoming.
    """
    print("\n" + "=" * 80)
    print("PROCESSING SCHEDULE")
    print("=" * 80)

    df = schedule.copy()

    # Parse datetime
    df["datetime"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

    # Parse goals from score (e.g. "2–1")
    if "score" not in df.columns:
        raise RuntimeError("FBref schedule missing 'score' column")

    # Split score into home/away goals
    scores = df["score"].astype(str).str.split("–", expand=True)
    df["home_goals"] = pd.to_numeric(scores[0], errors="coerce")
    df["away_goals"] = pd.to_numeric(scores[1], errors="coerce")

    # Validate xG columns
    if not {"home_xg", "away_xg"}.issubset(df.columns):
        print("⚠️  Warning: Missing xG columns, will be filled with NaN")
        df["home_xg"] = pd.NA
        df["away_xg"] = pd.NA

    # Get current time with a buffer (consider matches in last 6 hours as historical)
    now = pd.Timestamp.now(tz=timezone.utc)
    cutoff = now - timedelta(hours=6)

    print(f"\nCurrent time: {now}")
    print(f"Cutoff time (6h buffer): {cutoff}")

    # CRITICAL: A match is historical ONLY if it has BOTH goals AND is before cutoff
    # A match is upcoming if it has NO goals OR is after cutoff
    has_results = df["home_goals"].notna() & df["away_goals"].notna()
    is_past = df["datetime"] < cutoff

    historical = df[has_results & is_past].copy()
    upcoming = df[~(has_results & is_past)].copy()

    print(f"\n📊 Data split logic:")
    print(f"  Matches with results AND before cutoff: {len(historical)}")
    print(f"  Matches without results OR after cutoff: {len(upcoming)}")

    if not historical.empty:
        latest = historical["datetime"].max()
        print(f"\n  Latest historical match: {latest}")
        latest_match = historical[historical["datetime"] == latest].iloc[0]
        print(
            f"    {latest_match['home_team']} {latest_match['home_goals']:.0f}-{latest_match['away_goals']:.0f} {latest_match['away_team']}")

    if not upcoming.empty:
        next_match = upcoming["datetime"].min()
        print(f"\n  Next upcoming match: {next_match}")
        next_fixture = upcoming[upcoming["datetime"] == next_match].iloc[0]
        print(f"    {next_fixture['home_team']} vs {next_fixture['away_team']}")

        # Show sample of upcoming matches
        print(f"\n  Sample of upcoming matches:")
        for _, match in upcoming.head(5).iterrows():
            date_str = match["datetime"].strftime("%Y-%m-%d %H:%M") if pd.notna(match["datetime"]) else "TBD"
            print(f"    {date_str} | {match['home_team']} vs {match['away_team']}")

    # Sort both
    historical = historical.sort_values("datetime").reset_index(drop=True)
    upcoming = upcoming.sort_values("datetime").reset_index(drop=True)

    return historical, upcoming


# ------------------------------------------------------------------
# CALCULATE TEAM FORM FOR XG ESTIMATION
# ------------------------------------------------------------------
def calculate_recent_form(historical_df: pd.DataFrame, team: str, n_matches: int = 5):
    """
    Calculate average xG from last N matches for a team.
    Returns (avg_xg_for, avg_xg_against)
    """
    # Get team's recent matches (both home and away)
    home_matches = historical_df[historical_df["home_team"] == team].tail(n_matches * 2)
    away_matches = historical_df[historical_df["away_team"] == team].tail(n_matches * 2)

    # Combine and take last N
    all_matches = pd.concat([home_matches, away_matches]).sort_values("datetime").tail(n_matches)

    if all_matches.empty:
        return 1.5, 1.5  # Default values

    # Calculate xG for and against
    xg_for = []
    xg_against = []

    for _, row in all_matches.iterrows():
        # Skip matches without xG data
        if pd.isna(row.get("home_xg")) or pd.isna(row.get("away_xg")):
            continue

        if row["home_team"] == team:
            xg_for.append(float(row["home_xg"]))
            xg_against.append(float(row["away_xg"]))
        else:
            xg_for.append(float(row["away_xg"]))
            xg_against.append(float(row["home_xg"]))

    if not xg_for:
        return 1.5, 1.5

    return (
        sum(xg_for) / len(xg_for),
        sum(xg_against) / len(xg_against)
    )


# ------------------------------------------------------------------
# ESTIMATE XG FOR UPCOMING MATCHES
# ------------------------------------------------------------------
def estimate_xg_for_upcoming(upcoming: pd.DataFrame, historical: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate xG for upcoming matches based on recent team form.
    """
    if upcoming.empty:
        return upcoming

    print("\n" + "=" * 80)
    print("ESTIMATING XG FOR UPCOMING MATCHES")
    print("=" * 80)
    print("Using last 5 matches per team for form calculation\n")

    upcoming = upcoming.copy()

    for idx, row in upcoming.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]

        # Get recent form
        home_xg_for, home_xg_against = calculate_recent_form(historical, home_team)
        away_xg_for, away_xg_against = calculate_recent_form(historical, away_team)

        # Estimate xG (weighted average of offensive form and defensive form allowed)
        # Add home advantage boost (~0.3 xG)
        est_home_xg = (home_xg_for * 0.6 + away_xg_against * 0.4) + 0.3
        est_away_xg = (away_xg_for * 0.6 + home_xg_against * 0.4)

        upcoming.loc[idx, "home_xg"] = est_home_xg
        upcoming.loc[idx, "away_xg"] = est_away_xg

        date_str = row["datetime"].strftime("%Y-%m-%d") if pd.notna(row["datetime"]) else "TBD"
        print(f"  {date_str} | {home_team:25s} vs {away_team:25s} | xG: {est_home_xg:.2f} - {est_away_xg:.2f}")

    return upcoming


# ------------------------------------------------------------------
# SAVE OUTPUTS
# ------------------------------------------------------------------
def save_historical(df: pd.DataFrame):
    """Save historical matches with results."""
    if df.empty:
        print("\n⚠️  No historical data to save")
        return

    output = df[[
        "season",
        "datetime",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "home_xg",
        "away_xg",
    ]].copy()

    output.to_csv(HISTORICAL_OUTPUT, index=False)
    print(f"\n✅ Saved {len(output)} historical matches to:")
    print(f"   {HISTORICAL_OUTPUT.resolve()}")


def save_upcoming(df: pd.DataFrame):
    """Save upcoming fixtures with estimated xG."""
    if df.empty:
        print("\n⚠️  No upcoming fixtures to save")
        print("   This might mean:")
        print("   - The season has ended")
        print("   - All fixtures have been played")
        print("   - There's a break in the schedule")

        # Create empty file so pipeline doesn't break
        pd.DataFrame(columns=["season", "datetime", "home_team", "away_team", "home_xg", "away_xg"]).to_csv(
            UPCOMING_OUTPUT, index=False)
        return

    output = df[[
        "season",
        "datetime",
        "home_team",
        "away_team",
        "home_xg",
        "away_xg"
    ]].copy()

    output.to_csv(UPCOMING_OUTPUT, index=False)
    print(f"\n✅ Saved {len(output)} upcoming fixtures to:")
    print(f"   {UPCOMING_OUTPUT.resolve()}")


# ------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------
def main():
    print("\n" + "=" * 80)
    print("PREMIER LEAGUE DATA UPDATE PIPELINE")
    print("=" * 80)

    try:
        # Step 1: Download data
        schedule = download_fbref_data()

        # Step 2: Process and split
        historical, upcoming = process_schedule(schedule)

        # Step 3: Estimate xG for upcoming matches
        if not upcoming.empty and not historical.empty:
            upcoming = estimate_xg_for_upcoming(upcoming, historical)

        # Step 4: Save outputs
        save_historical(historical)
        save_upcoming(upcoming)

        # Summary
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(f"\n📈 Historical matches: {len(historical)}")
        print(f"🔮 Upcoming fixtures: {len(upcoming)}")

        if not upcoming.empty:
            print(f"\n🎯 Next fixture:")
            next_match = upcoming.iloc[0]
            date_str = next_match["datetime"].strftime("%A, %B %d, %Y at %H:%M UTC")
            print(f"   {next_match['home_team']} vs {next_match['away_team']}")
            print(f"   {date_str}")

        print("\n✨ Next steps:")
        print("   1. Run: python predict_scores.py")
        print("   2. Run: python visualise.py")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())