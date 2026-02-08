#!/usr/bin/env python3
"""
update_data.py - IMPROVED VERSION with better FBref handling

Unified data pipeline that:
1. Downloads/updates historical match data from FBref
2. Identifies which fixtures are upcoming based on current date
3. Estimates xG for upcoming matches based on recent form
4. Produces two outputs:
   - data/matches_master.csv (all historical matches with results)
   - data/upcoming_fixtures.csv (upcoming matches with estimated xG)

IMPROVEMENTS:
- Better rate limiting to avoid 403 errors
- Exponential backoff on retries
- User agent rotation
- Cache-first approach
- Fallback to cached data if scraping fails
"""

import os
from pathlib import Path
import pandas as pd
import soccerdata as sd
from datetime import datetime, timezone, timedelta
import time
import random

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "soccerdata_cache"

DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

os.environ["SOCCERDATA_DIR"] = str(CACHE_DIR)

# More realistic user agents - rotate to avoid detection
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# Select random user agent
os.environ["SOCCERDATA_USER_AGENT"] = random.choice(USER_AGENTS)

# Rate limiting configuration
RATE_LIMIT_DELAY = 5  # seconds between requests (increased for safety)
MAX_RETRIES = 3
RETRY_BASE_DELAY = 10  # base seconds for exponential backoff

LEAGUE = "ENG-Premier League"


# Determine current season dynamically
def get_current_season():
    """
    Determine current Premier League season based on date.
    Season starts in August, so:
    - Jan-July: previous year's season (e.g., Jan 2025 = 2024 season)
    - Aug-Dec: current year's season (e.g., Aug 2024 = 2024 season)
    """
    from datetime import datetime
    now = datetime.now()
    if now.month >= 8:  # August or later
        return now.year
    else:  # January to July
        return now.year - 1


CURRENT_SEASON = get_current_season()
# Download data from multiple seasons for better training
SEASONS = list(range(2018, CURRENT_SEASON + 1))

HISTORICAL_OUTPUT = DATA_DIR / "matches_master.csv"
UPCOMING_OUTPUT = DATA_DIR / "upcoming_fixtures.csv"


# ------------------------------------------------------------------
# IMPROVED DOWNLOAD WITH BETTER ERROR HANDLING
# ------------------------------------------------------------------
def download_fbref_data_with_retry():
    """Download schedule from FBref with improved error handling and rate limiting."""
    print("=" * 80)
    print("DOWNLOADING DATA FROM FBREF")
    print("=" * 80)
    print(f"\nLeague: {LEAGUE}")
    print(f"Current Season: {CURRENT_SEASON}/{str(CURRENT_SEASON + 1)[-2:]}")
    print(f"Downloading Seasons: {SEASONS[0]} to {SEASONS[-1]}")
    print(f"Cache directory: {CACHE_DIR.resolve()}\n")

    print("⏱️  Using rate limiting to avoid 403 errors...")
    print(f"   Delay between requests: {RATE_LIMIT_DELAY}s")
    print(f"   Max retries: {MAX_RETRIES}")
    print("")

    for attempt in range(MAX_RETRIES):
        try:
            print(f"📥 Attempt {attempt + 1}/{MAX_RETRIES}...")

            # Add delay before request (except first attempt)
            if attempt > 0:
                delay = RETRY_BASE_DELAY * (2 ** attempt)  # exponential backoff
                print(f"   ⏳ Waiting {delay}s before retry...")
                time.sleep(delay)

            # Create FBref scraper with no_cache=False to prefer cached data
            fbref = sd.FBref(
                leagues=LEAGUE,
                seasons=SEASONS,
                data_dir=CACHE_DIR,
                no_cache=False  # Use cache when available to reduce requests
            )

            print("   Downloading match schedule...")
            schedule = fbref.read_schedule()

            if schedule is None or schedule.empty:
                print("   ⚠️  Downloaded schedule is empty")
                continue

            print(f"   ✅ Successfully downloaded {len(schedule)} matches")
            return schedule.reset_index()

        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ Error: {error_msg}")

            # Check if it's a 403 error
            if "403" in error_msg or "Forbidden" in error_msg:
                print("\n   🚫 FBref is blocking requests (403 Forbidden)")
                print("   💡 This usually means:")
                print("      • Too many requests in short time")
                print("      • Need to increase delay between requests")
                print("      • Consider using cached data")

            if attempt < MAX_RETRIES - 1:
                print(f"\n   🔄 Will retry in a moment...")
            else:
                print(f"\n   ❌ All {MAX_RETRIES} attempts failed")

                # Try to use cached data as fallback
                print("\n   🔍 Attempting to use cached data...")
                return try_load_cached_schedule()

    # If all retries failed
    return try_load_cached_schedule()


def try_load_cached_schedule():
    """Try to load schedule from cache files as a fallback."""
    print("\n" + "=" * 80)
    print("FALLBACK: LOADING FROM CACHE")
    print("=" * 80)

    cache_files = list(CACHE_DIR.rglob("*.csv")) + list(CACHE_DIR.rglob("*.html"))

    if not cache_files:
        print("❌ No cache files found")
        raise RuntimeError(
            "Could not download from FBref and no cache available.\n"
            "Solutions:\n"
            "1. Wait a few hours and try again (FBref may have rate limited you)\n"
            "2. Use a VPN to change your IP address\n"
            "3. Manually download data from FBref and place in cache directory\n"
            "4. Use alternative data source (see documentation)"
        )

    print(f"\n✓ Found {len(cache_files)} cached files")
    print(f"  Latest: {max(cache_files, key=lambda p: p.stat().st_mtime)}")
    print("\n⚠️  Using cached data - predictions may not include latest matches")

    # Try to reconstruct schedule from cache
    try:
        fbref = sd.FBref(
            leagues=LEAGUE,
            seasons=SEASONS,
            data_dir=CACHE_DIR,
            no_cache=False
        )

        # This will use only cached data
        schedule = fbref.read_schedule()

        if schedule is not None and not schedule.empty:
            print(f"✅ Loaded {len(schedule)} matches from cache")
            return schedule.reset_index()

    except Exception as e:
        print(f"❌ Could not load from cache: {e}")

    raise RuntimeError("Could not download or load from cache")


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
        # Step 1: Download data with improved error handling
        schedule = download_fbref_data_with_retry()

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

        print("\n💡 Troubleshooting:")
        print("   1. FBref is blocking requests - wait a few hours")
        print("   2. Try using a VPN to change your IP address")
        print("   3. Check if cached data exists in:", CACHE_DIR)
        print("   4. Consider manually downloading data")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())