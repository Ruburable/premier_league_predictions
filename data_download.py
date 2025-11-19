#!/usr/bin/env python3
"""
data_download_with_gameweek_mapping.py

- Scrapes Understat per-season league pages
- Downloads per-match Understat event JSONs (resumable)
- Fetches FPL fixtures & team names and uses them to assign 'event' (gameweek)
- Normalizes multiple Understat schemas (2025 schema with ['h','a','goals','xG'] lists and older schemas)
- Skips future matches and already-downloaded event files
- Produces data/matches_master.csv with gameweek assignments
"""

import re
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# -----------------------
# Config
# -----------------------
OUTPUT_DIR = Path("data")
EVENTS_DIR = OUTPUT_DIR / "match_events"
OUTPUT_DIR.mkdir(exist_ok=True)
EVENTS_DIR.mkdir(exist_ok=True)

UNDERSTAT_LEAGUE_URL = "https://understat.com/league/EPL/{year}"
UNDERSTAT_MATCH_URL = "https://understat.com/match/{match_id}"

FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

# Seasons to fetch from Understat (first-year string used in URLs)
HISTORICAL_SEASONS = ["2018", "2019", "2020", "2021", "2022", "2023", "2024"]
CURRENT_SEASON_YEAR = "2025"  # corresponds to 2025/26

# polite delay between requests
DELAY = 1.2

# maximum time tolerance when matching kickoff datetimes (hours)
MATCH_TIME_TOLERANCE_HOURS = 12

# output master file
MASTER_CSV = OUTPUT_DIR / "matches_master.csv"

# -----------------------
# Helpers
# -----------------------

def polite_sleep():
    time.sleep(DELAY)

def safe_get(url, session=None, timeout=25):
    s = session or requests
    for attempt in range(3):
        try:
            r = s.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                polite_sleep()
                return r.text
            else:
                print(f"Warning: {url} returned status {r.status_code}")
        except Exception as e:
            print(f"Request error for {url}: {e}")
        time.sleep(1 + attempt * 2)
    return None

def extract_json_from_understat_html(html_text):
    """Find and decode JSON.parse('...') payloads on Understat pages."""
    if not html_text:
        return None
    # Look for JSON.parse('....') pattern
    m = re.search(r"JSON\.parse\('(?P<json>.+?)'\)", html_text, flags=re.DOTALL)
    if not m:
        # fallback: look for "matches": [ ... ] pattern
        m2 = re.search(r"\"matches\"\s*:\s*(\[[\s\S]+?\])\s*,\s*\"teams\"", html_text, flags=re.DOTALL)
        if m2:
            raw = m2.group(1)
            try:
                return json.loads(raw)
            except Exception:
                return None
        return None
    raw = m.group("json")
    try:
        decoded = raw.encode("utf-8").decode("unicode_escape")
    except Exception:
        decoded = raw
    try:
        return json.loads(decoded)
    except Exception:
        # try minor fixes
        try:
            fixed = decoded.replace("\\'", "'")
            return json.loads(fixed)
        except Exception:
            return None

def parse_understat_league(year):
    """Return list-of-dicts for matches from Understat league page for given year (e.g., 2025)."""
    url = UNDERSTAT_LEAGUE_URL.format(year=year)
    print(f"Fetching Understat league page: {url}")
    html = safe_get(url)
    data = extract_json_from_understat_html(html)
    if data is None:
        print(f"Failed to parse Understat league page for {year}.")
        return []
    # data often is list-of-dates each containing 'matches' or directly list of matches
    matches = []
    if isinstance(data, list):
        # could be list of matches or list of date-groups
        if data and isinstance(data[0], dict) and "matches" in data[0]:
            for d in data:
                matches.extend(d.get("matches", []))
        else:
            matches = data
    elif isinstance(data, dict):
        # sometimes dict with 'matches'
        if "matches" in data:
            matches = data["matches"]
        else:
            # unknown dict shape
            matches = []
    return matches

def normalize_understat_match(record, year):
    """
    Normalize a raw Understat match record to consistent fields:
    - id
    - datetime (as pandas.Timestamp)
    - season (e.g., '2025-26')
    - home_team, away_team
    - home_goals, away_goals
    - home_xg, away_xg (floats, or NaN)
    - source = 'understat'
    """
    out = {}
    # id
    out['id'] = record.get('id') or record.get('match_id') or record.get('matchId') or None

    # many schemas:
    # - new (2025): 'h' (home team), 'a' (away team), 'goals' == [h,a], 'xG' == [h_xg, a_xg], 'datetime'
    # - older: 'h_team', 'a_team', 'h_goals', 'a_goals', 'hxG', 'axG', 'date'
    # - some: 'h_team'/'a_team' with 'goals' nested
    # home/away names
    home = record.get('h') or record.get('h_team') or record.get('home') or record.get('hTeam') or record.get('home_team')
    away = record.get('a') or record.get('a_team') or record.get('away') or record.get('aTeam') or record.get('away_team')
    out['home_team'] = home
    out['away_team'] = away

    # datetime
    dt = record.get('datetime') or record.get('date') or record.get('match_date')
    # Understat often sets timezone; parse with pandas
    try:
        out['datetime'] = pd.to_datetime(dt)
    except Exception:
        out['datetime'] = pd.NaT

    # goals: different shapes
    home_goals = None
    away_goals = None
    if 'goals' in record and isinstance(record['goals'], (list, tuple)) and len(record['goals']) >= 2:
        home_goals = record['goals'][0]
        away_goals = record['goals'][1]
    else:
        # try explicit fields
        for k in ['h_goals', 'a_goals', 'h_goals_ft', 'a_goals_ft', 'goals_h', 'goals_a']:
            if k in record:
                # attempt to map based on name
                if 'h' in k or k.startswith('home') or k.startswith('goals_h'):
                    home_goals = record.get(k)
                if 'a' in k or k.startswith('away') or k.startswith('goals_a'):
                    away_goals = record.get(k)
    # normalize numeric
    try:
        out['home_goals'] = int(home_goals) if home_goals is not None and str(home_goals).strip() != '' else None
    except Exception:
        out['home_goals'] = None
    try:
        out['away_goals'] = int(away_goals) if away_goals is not None and str(away_goals).strip() != '' else None
    except Exception:
        out['away_goals'] = None

    # xG: record['xG'] may be list [home_xg, away_xg] or fields 'hxG'/'axG' or 'xG'
    home_xg = None
    away_xg = None
    if 'xG' in record and isinstance(record['xG'], (list, tuple)) and len(record['xG']) >= 2:
        home_xg = record['xG'][0]
        away_xg = record['xG'][1]
    else:
        # try hxG/axG fields
        for k in ['hxG', 'home_xg', 'home_xG', 'h_xg', 'home_xG_ft']:
            if k in record:
                try:
                    home_xg = float(record.get(k))
                except Exception:
                    pass
        for k in ['axG', 'away_xg', 'away_xG', 'a_xg', 'away_xG_ft']:
            if k in record:
                try:
                    away_xg = float(record.get(k))
                except Exception:
                    pass
    # coerce to floats or None
    try:
        out['home_xg'] = float(home_xg) if home_xg is not None and str(home_xg) != '' else None
    except Exception:
        out['home_xg'] = None
    try:
        out['away_xg'] = float(away_xg) if away_xg is not None and str(away_xg) != '' else None
    except Exception:
        out['away_xg'] = None

    out['season'] = f"{year}-{int(year)+1}" if year.isdigit() else year
    out['source'] = 'understat'
    return out

# -----------------------
# FPL helpers (for gameweek mapping)
# -----------------------

def load_fpl_fixtures_and_teams():
    """
    Returns:
      fixtures_df: DataFrame with columns including 'id','team_h','team_a','event','kickoff_time'
      teams_df: DataFrame mapping fpl team id -> team name ('id','name')
    """
    print("Fetching FPL bootstrap (teams) and fixtures...")
    r = safe_get(FPL_BOOTSTRAP_URL)
    if r is None:
        raise RuntimeError("Could not fetch FPL bootstrap data.")
    boot = json.loads(r)
    teams = boot.get('teams', [])
    teams_df = pd.DataFrame(teams)[['id', 'name']]
    teams_df.columns = ['team_id', 'team_name']

    r2 = safe_get(FPL_FIXTURES_URL)
    if r2 is None:
        raise RuntimeError("Could not fetch FPL fixtures.")
    fixtures = json.loads(r2)
    fixtures_df = pd.DataFrame(fixtures)
    # convert kickoff_time to datetime
    if 'kickoff_time' in fixtures_df.columns:
        fixtures_df['kickoff_time'] = pd.to_datetime(fixtures_df['kickoff_time'], errors='coerce')
    return fixtures_df, teams_df

def normalize_name(name):
    """Lowercase, remove punctuation and common variations to help matching."""
    if name is None:
        return ''
    s = str(name).lower()
    # remove punctuation, accents
    s = re.sub(r"[^a-z0-9]", "", s)
    # map common variations
    s = s.replace('manutd', 'manutd')  # example
    return s

def build_fpl_name_lookup(teams_df):
    """
    Return dict mapping normalized team_name -> team_id and original name.
    Some names may be ambiguous; we keep normalized->list mapping.
    """
    lookup = {}
    for _, r in teams_df.iterrows():
        nid = int(r['team_id'])
        name = r['team_name']
        norm = normalize_name(name)
        lookup.setdefault(norm, []).append({'team_id': nid, 'team_name': name})
    return lookup

# -----------------------
# Matching Understat match -> FPL fixture
# -----------------------

def find_fpl_event_for_match(under_row, fixtures_df, teams_df):
    """
    Try to find FPL fixture (and its 'event' / gameweek) matching the Understat row.
    Matching logic:
    - Normalize home/away team names for both sides and compare
    - Match kickoff times within MATCH_TIME_TOLERANCE_HOURS
    - If found, return event (int) else None
    """
    # prepare normalized names
    home_us = normalize_name(under_row['home_team'])
    away_us = normalize_name(under_row['away_team'])
    # build mapping of FPL fixture home/away names normalized
    # we need team_id->team_name mapping
    # create a temporary merged fixtures DataFrame with team names
    fpl = fixtures_df.copy()
    fpl = fpl.merge(teams_df, left_on='team_h', right_on='team_id', how='left').rename(columns={'team_name':'team_h_name'}).drop(columns=['team_id'])
    fpl = fpl.merge(teams_df, left_on='team_a', right_on='team_id', how='left').rename(columns={'team_name':'team_a_name'}).drop(columns=['team_id'])
    # add normalized names
    fpl['team_h_norm'] = fpl['team_h_name'].apply(normalize_name)
    fpl['team_a_norm'] = fpl['team_a_name'].apply(normalize_name)
    # filter by name pairs
    candidates = fpl[(fpl['team_h_norm'] == home_us) & (fpl['team_a_norm'] == away_us)].copy()
    if candidates.empty:
        # try swapped names (sometimes understat uses different order)
        candidates = fpl[(fpl['team_h_norm'] == away_us) & (fpl['team_a_norm'] == home_us)].copy()
        # If swapped, we will ignore (shouldn't happen) but continue
    # filter by datetime tolerance
    under_dt = under_row.get('datetime')
    if pd.isna(under_dt):
        # if no datetime, fallback to name-only match and return event if unique
        if len(candidates) == 1:
            ev = candidates.iloc[0].get('event')
            return int(ev) if not pd.isna(ev) else None
        return None
    tol = pd.Timedelta(hours=MATCH_TIME_TOLERANCE_HOURS)
    # ensure fixtures have kickoff_time
    if 'kickoff_time' not in candidates.columns:
        candidates['kickoff_time'] = pd.NaT
    # select candidates within time tolerance
    candidates['kickoff_time'] = pd.to_datetime(candidates['kickoff_time'], errors='coerce')
    # compute time difference
    candidates['dt_diff'] = (candidates['kickoff_time'] - under_dt).abs()
    cand_ok = candidates[candidates['dt_diff'] <= tol]
    if not cand_ok.empty:
        # choose the candidate with smallest dt_diff
        chosen = cand_ok.sort_values('dt_diff').iloc[0]
        ev = chosen.get('event')
        return int(ev) if not pd.isna(ev) else None
    # as fallback, if candidates exist but none within tolerance, return None
    return None

# -----------------------
# Main orchestration
# -----------------------

def load_existing_master():
    if MASTER_CSV.exists():
        return pd.read_csv(MASTER_CSV, parse_dates=['datetime'], dayfirst=False)
    return pd.DataFrame()

def save_master(df):
    df = df.sort_values(['season', 'datetime']).reset_index(drop=True)
    df.to_csv(MASTER_CSV, index=False)
    print(f"Saved master CSV: {MASTER_CSV}")

def download_and_build_master():
    # load FPL fixtures & teams to map gameweeks
    fixtures_df, teams_df = load_fpl_fixtures_and_teams()

    # collect all Understat seasons (historical + current)
    all_raw_matches = []

    seasons = HISTORICAL_SEASONS + [CURRENT_SEASON_YEAR]
    for year in seasons:
        print(f"Processing Understat season page for year {year}...")
        raw_matches = parse_understat_league(year)
        for rec in raw_matches:
            norm = normalize_understat_match(rec, year)
            # only keep matches with datetime or that have been played (isResult True)
            all_raw_matches.append(norm)

    master_df = pd.DataFrame(all_raw_matches)
    # ensure datetime is parsed
    if 'datetime' in master_df.columns:
        master_df['datetime'] = pd.to_datetime(master_df['datetime'], errors='coerce')

    # attach FPL event where possible
    print("Mapping Understat matches to FPL fixtures (gameweeks)...")
    mapped_events = []
    for _, r in tqdm(master_df.iterrows(), total=len(master_df)):
        row = r.to_dict()
        try:
            ev = find_fpl_event_for_match(row, fixtures_df, teams_df)
        except Exception:
            ev = None
        mapped_events.append(ev)
    master_df['event'] = mapped_events

    # filter out future matches (datetime > now)
    now = pd.Timestamp.now()
    # keep matches with datetime <= now OR isResult True
    cond_past = (master_df['datetime'].notna() & (master_df['datetime'] <= now)) | (master_df.get('home_goals').notna() & master_df.get('away_goals').notna())
    master_df = master_df[cond_past].copy()

    # ensure consistent columns
    # ensure home_goals/away_goals numeric
    master_df['home_goals'] = pd.to_numeric(master_df.get('home_goals'), errors='coerce')
    master_df['away_goals'] = pd.to_numeric(master_df.get('away_goals'), errors='coerce')

    # Save master
    save_master(master_df)

    # Download per-match events for matches that have an 'id' and are past and not already downloaded
    print("Downloading per-match event JSONs for missing matches...")
    for _, r in tqdm(master_df.iterrows(), total=len(master_df)):
        mid = r.get('id')
        dt = r.get('datetime')
        if pd.isna(mid):
            continue
        # skip future by dt (already filtered but check)
        if pd.notna(dt) and pd.to_datetime(dt) > now:
            continue
        fpath = EVENTS_DIR / f"{mid}.json"
        if fpath.exists():
            continue
        # attempt to download match page
        print(f"Downloading match events for id {mid} ...")
        html = safe_get(UNDERSTAT_MATCH_URL.format(match_id=mid))
        jsdata = extract_json_from_understat_html(html)
        if jsdata:
            # save json (raw)
            try:
                with open(fpath, "w", encoding="utf-8") as fh:
                    json.dump(jsdata, fh, ensure_ascii=False, indent=2)
                polite_sleep()
            except Exception as e:
                print(f"Failed to save events for {mid}: {e}")
        else:
            print(f"No event JSON found for match {mid}; skipping (will retry next run).")

    return master_df

# -----------------------
# Entrypoint
# -----------------------

if __name__ == "__main__":
    print("=== Starting Understat + FPL mapping downloader ===")
    master = download_and_build_master()
    print("Done. Master rows:", len(master))
