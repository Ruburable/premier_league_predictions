#!/usr/bin/env python3
"""
data_download.py

Downloads historical matches from Understat and upcoming fixtures from FPL API.
Produces output/matches_master.csv with all matches.
"""

import re
import os
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import numpy as np

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
EVENTS_DIR = OUTPUT_DIR / "match_events"
EVENTS_DIR.mkdir(exist_ok=True)

UNDERSTAT_LEAGUE_URL = "https://understat.com/league/EPL/{year}"
UNDERSTAT_MATCH_URL = "https://understat.com/match/{match_id}"

FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

HISTORICAL_SEASONS = ["2018", "2019", "2020", "2021", "2022", "2023", "2024"]
CURRENT_SEASON_YEAR = "2025"

DELAY = 1.2
MATCH_TIME_TOLERANCE_HOURS = 12
MASTER_CSV = OUTPUT_DIR / "matches_master.csv"


def polite_sleep():
    time.sleep(DELAY)


def safe_get_text(url, session=None, timeout=25, params=None, headers=None):
    s = session or requests
    for attempt in range(3):
        try:
            r = s.get(url, timeout=timeout, params=params, headers=headers or {"User-Agent": "Mozilla/5.0"})
            polite_sleep()
            return r.text if r.status_code == 200 else None
        except Exception:
            time.sleep(1 + attempt * 2)
    return None


def safe_get_json(url, session=None, timeout=25, params=None, headers=None):
    s = session or requests
    for attempt in range(3):
        try:
            r = s.get(url, timeout=timeout, params=params, headers=headers or {"User-Agent": "Mozilla/5.0"})
            polite_sleep()
            if r.status_code == 200:
                return r.json()
            return None
        except Exception:
            time.sleep(1 + attempt * 2)
    return None


def extract_json_from_understat_html(html_text):
    if not html_text:
        return None

    m = re.search(r"JSON\.parse\('(?P<json>.+?)'\)", html_text, flags=re.DOTALL)
    if not m:
        m2 = re.search(r"\"matches\"\s*:\s*(\[[\s\S]+?\])\s*,\s*\"teams\"", html_text, flags=re.DOTALL)
        if m2:
            try:
                return json.loads(m2.group(1))
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
        try:
            return json.loads(decoded.replace("\\'", "'"))
        except Exception:
            return None


def parse_understat_league(year):
    url = UNDERSTAT_LEAGUE_URL.format(year=year)
    html = safe_get_text(url)
    data = extract_json_from_understat_html(html)
    if data is None:
        return []

    matches = []
    if isinstance(data, list):
        if data and isinstance(data[0], dict) and "matches" in data[0]:
            for d in data:
                matches.extend(d.get("matches", []))
        else:
            matches = data
    elif isinstance(data, dict) and "matches" in data:
        matches = data["matches"]

    return matches


def normalize_understat_match(record, year):
    out = {}
    out['id'] = record.get('id') or record.get('match_id') or record.get('matchId') or None

    home = record.get('h') or record.get('h_team') or record.get('home') or record.get('hTeam') or record.get(
        'home_team')
    away = record.get('a') or record.get('a_team') or record.get('away') or record.get('aTeam') or record.get(
        'away_team')
    out['home_team'] = str(home) if home is not None else None
    out['away_team'] = str(away) if away is not None else None

    dt = record.get('datetime') or record.get('date') or record.get('match_date')
    try:
        out['datetime'] = pd.to_datetime(dt)
    except Exception:
        out['datetime'] = pd.NaT

    home_goals = away_goals = None
    if 'goals' in record and isinstance(record['goals'], (list, tuple)) and len(record['goals']) >= 2:
        home_goals = record['goals'][0]
        away_goals = record['goals'][1]
    else:
        for k in ['h_goals', 'a_goals', 'h_goals_ft', 'a_goals_ft', 'goals_h', 'goals_a', 'home_goals', 'away_goals']:
            if k in record:
                if ('h' in k) or k.startswith('home'):
                    home_goals = record.get(k)
                if ('a' in k) or k.startswith('away'):
                    away_goals = record.get(k)

    try:
        out['home_goals'] = int(home_goals) if home_goals is not None and str(home_goals).strip() != '' else None
    except Exception:
        out['home_goals'] = None
    try:
        out['away_goals'] = int(away_goals) if away_goals is not None and str(away_goals).strip() != '' else None
    except Exception:
        out['away_goals'] = None

    home_xg = away_xg = None
    xg_data = None
    for key in ('xG', 'xg', 'hxG', 'home_xg', 'home_xG', 'h_xg', 'home_xG_ft'):
        if key in record:
            xg_data = record.get(key)
            break

    if isinstance(xg_data, (list, tuple)) and len(xg_data) >= 2:
        try:
            home_xg = float(xg_data[0])
            away_xg = float(xg_data[1])
        except Exception:
            home_xg = away_xg = None
    elif isinstance(xg_data, dict):
        home_xg = xg_data.get('h') or xg_data.get('home') or xg_data.get('home_xg')
        away_xg = xg_data.get('a') or xg_data.get('away') or xg_data.get('away_xg')
    else:
        try:
            home_xg = float(record.get('home_xg')) if record.get('home_xg') is not None else None
        except Exception:
            home_xg = None
        try:
            away_xg = float(record.get('away_xg')) if record.get('away_xg') is not None else None
        except Exception:
            away_xg = None

    try:
        out['home_xg'] = float(home_xg) if home_xg is not None and str(home_xg) != '' else None
    except Exception:
        out['home_xg'] = None
    try:
        out['away_xg'] = float(away_xg) if away_xg is not None and str(away_xg) != '' else None
    except Exception:
        out['away_xg'] = None

    out['season'] = f"{year}-{int(year) + 1}" if str(year).isdigit() else year
    out['source'] = 'understat'
    return out


def download_match_events(match_id):
    fpath = EVENTS_DIR / f"{match_id}.json"
    if fpath.exists():
        return

    url = UNDERSTAT_MATCH_URL.format(match_id=match_id)
    html = safe_get_text(url)
    data = extract_json_from_understat_html(html)
    if data:
        try:
            with open(fpath, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
            polite_sleep()
        except Exception:
            pass


def load_fpl_fixtures_and_teams():
    boot = safe_get_json(FPL_BOOTSTRAP_URL)
    if boot is None:
        print("Warning: Failed to load FPL bootstrap data")
        return pd.DataFrame(), pd.DataFrame()

    teams = boot.get('teams', [])
    teams_df = pd.DataFrame(teams)
    if not teams_df.empty and 'id' in teams_df.columns and 'name' in teams_df.columns:
        teams_df = teams_df[['id', 'name']].copy()
        teams_df.columns = ['team_id', 'team_name']
    else:
        teams_df = pd.DataFrame(columns=['team_id', 'team_name'])

    fixtures = safe_get_json(FPL_FIXTURES_URL)
    if fixtures is None:
        print("Warning: Failed to load FPL fixtures")
        return pd.DataFrame(), teams_df

    fixtures_df = pd.DataFrame(fixtures)
    if not fixtures_df.empty and 'kickoff_time' in fixtures_df.columns:
        fixtures_df['kickoff_time'] = pd.to_datetime(fixtures_df['kickoff_time'], errors='coerce')

    return fixtures_df, teams_df


def normalize_name(name):
    if name is None:
        return ''
    s = str(name).lower()
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def find_fpl_event_for_match(under_row, fixtures_df, teams_df):
    home_us = normalize_name(under_row.get('home_team'))
    away_us = normalize_name(under_row.get('away_team'))

    if fixtures_df.empty or teams_df.empty:
        return None

    fpl = fixtures_df.copy()

    if 'team_h' in fpl.columns and pd.api.types.is_numeric_dtype(fpl['team_h']):
        fpl = fpl.merge(teams_df, left_on='team_h', right_on='team_id', how='left')
        fpl = fpl.rename(columns={'team_name': 'team_h_name'})
        fpl = fpl.drop(columns=['team_id'], errors='ignore')

    if 'team_a' in fpl.columns and pd.api.types.is_numeric_dtype(fpl['team_a']):
        fpl = fpl.merge(teams_df, left_on='team_a', right_on='team_id', how='left')
        fpl = fpl.rename(columns={'team_name': 'team_a_name'})
        fpl = fpl.drop(columns=['team_id'], errors='ignore')

    if 'team_h_name' not in fpl.columns and 'team_h' in fpl.columns:
        fpl['team_h_name'] = fpl['team_h'].astype(str)
    if 'team_a_name' not in fpl.columns and 'team_a' in fpl.columns:
        fpl['team_a_name'] = fpl['team_a'].astype(str)

    fpl['team_h_norm'] = fpl['team_h_name'].astype(str).apply(normalize_name)
    fpl['team_a_norm'] = fpl['team_a_name'].astype(str).apply(normalize_name)

    candidates = fpl[(fpl['team_h_norm'] == home_us) & (fpl['team_a_norm'] == away_us)].copy()
    if candidates.empty:
        candidates = fpl[(fpl['team_h_norm'] == away_us) & (fpl['team_a_norm'] == home_us)].copy()

    if candidates.empty:
        return None

    under_dt = under_row.get('datetime')
    if pd.isna(under_dt):
        if len(candidates) == 1:
            ev = candidates.iloc[0].get('event')
            return int(ev) if not pd.isna(ev) else None
        return None

    tol = pd.Timedelta(hours=MATCH_TIME_TOLERANCE_HOURS)
    candidates['kickoff_time'] = pd.to_datetime(candidates.get('kickoff_time'), errors='coerce')
    candidates['dt_diff'] = (candidates['kickoff_time'] - under_dt).abs()
    cand_ok = candidates[candidates['dt_diff'] <= tol]

    if not cand_ok.empty:
        chosen = cand_ok.sort_values('dt_diff').iloc[0]
        ev = chosen.get('event')
        return int(ev) if not pd.isna(ev) else None

    return None


def extract_name(x):
    if isinstance(x, dict):
        return str(x.get('name') or x.get('team_name') or x.get('short_name') or '')
    return str(x) if x is not None else ''


def get_next_gameweek(fixtures_df):
    """Find the next upcoming gameweek number"""
    if fixtures_df is None or fixtures_df.empty:
        return None

    df = fixtures_df.copy()
    if 'kickoff_time' in df.columns:
        df['kickoff_time'] = pd.to_datetime(df['kickoff_time'], errors='coerce')

    now = pd.Timestamp.utcnow()
    future = df[df['kickoff_time'].notna() & (df['kickoff_time'] > now)]

    if future.empty:
        return None

    next_event = future['event'].min()
    return int(next_event) if not pd.isna(next_event) else None


def build_upcoming_from_fpl(fixtures_df, teams_df):
    if fixtures_df is None or fixtures_df.empty:
        return pd.DataFrame()

    next_gw = get_next_gameweek(fixtures_df)
    if next_gw is None:
        print("Could not determine next gameweek")
        return pd.DataFrame()

    print(f"Next gameweek: {next_gw}")

    df = fixtures_df[fixtures_df['event'] == next_gw].copy()

    if df.empty:
        return pd.DataFrame()

    if 'kickoff_time' in df.columns:
        df['datetime'] = pd.to_datetime(df['kickoff_time'], errors='coerce', utc=True)
    else:
        df['datetime'] = pd.NaT

    if 'team_h' in df.columns and pd.api.types.is_numeric_dtype(df['team_h']):
        df = df.merge(teams_df, left_on='team_h', right_on='team_id', how='left')
        df = df.rename(columns={'team_name': 'team_h_name'})
        df = df.drop(columns=['team_id'], errors='ignore')

    if 'team_a' in df.columns and pd.api.types.is_numeric_dtype(df['team_a']):
        df = df.merge(teams_df, left_on='team_a', right_on='team_id', how='left')
        df = df.rename(columns={'team_name': 'team_a_name'})
        df = df.drop(columns=['team_id'], errors='ignore')

    if 'team_h_name' not in df.columns and 'team_h' in df.columns:
        df['team_h_name'] = df['team_h'].astype(str)
    if 'team_a_name' not in df.columns and 'team_a' in df.columns:
        df['team_a_name'] = df['team_a'].astype(str)

    df['team_h_name'] = df['team_h_name'].apply(extract_name)
    df['team_a_name'] = df['team_a_name'].apply(extract_name)

    out = pd.DataFrame({
        'season': f"{CURRENT_SEASON_YEAR}-{int(CURRENT_SEASON_YEAR) + 1}",
        'event': next_gw,
        'datetime': pd.to_datetime(df['datetime'], errors='coerce'),
        'home_team': df['team_h_name'].astype(str),
        'away_team': df['team_a_name'].astype(str),
        'home_goals': np.nan,
        'away_goals': np.nan,
        'home_xg': np.nan,
        'away_xg': np.nan
    })

    return out.reset_index(drop=True)


def download_and_build_master():
    fixtures_df, teams_df = load_fpl_fixtures_and_teams()

    all_raw_matches = []
    seasons = HISTORICAL_SEASONS + [CURRENT_SEASON_YEAR]

    print("Downloading historical matches from Understat...")
    for year in seasons:
        raw_matches = parse_understat_league(year)
        for rec in raw_matches:
            norm = normalize_understat_match(rec, year)
            all_raw_matches.append(norm)

    master_df = pd.DataFrame(all_raw_matches)
    if 'datetime' in master_df.columns:
        master_df['datetime'] = pd.to_datetime(master_df['datetime'], errors='coerce')

    master_df['home_team'] = master_df['home_team'].astype(str)
    master_df['away_team'] = master_df['away_team'].astype(str)

    print("Mapping FPL gameweeks to historical matches...")
    mapped_events = []
    for _, r in tqdm(master_df.iterrows(), total=len(master_df)):
        row = r.to_dict()
        try:
            ev = find_fpl_event_for_match(row, fixtures_df, teams_df)
        except Exception:
            ev = None
        mapped_events.append(ev)
    master_df['event'] = mapped_events

    now = pd.Timestamp.now()
    cond_past = (master_df['datetime'].notna() & (master_df['datetime'] <= now)) | \
                (master_df.get('home_goals').notna() & master_df.get('away_goals').notna())
    master_df = master_df[cond_past].copy()

    master_df['home_goals'] = pd.to_numeric(master_df.get('home_goals'), errors='coerce')
    master_df['away_goals'] = pd.to_numeric(master_df.get('away_goals'), errors='coerce')

    print("Downloading match events...")
    for _, r in tqdm(master_df.iterrows(), total=len(master_df)):
        mid = r.get('id')
        dt = r.get('datetime')
        if pd.isna(mid):
            continue
        if pd.notna(dt) and pd.to_datetime(dt) > now:
            continue
        download_match_events(mid)

    print("Fetching upcoming gameweek fixtures from FPL...")
    future_df = build_upcoming_from_fpl(fixtures_df, teams_df)

    if future_df is None or future_df.empty:
        print("No upcoming fixtures for next gameweek.")
        combined = master_df
    else:
        print(f"Found {len(future_df)} upcoming fixtures in next gameweek")

        future_df['home_team'] = future_df['home_team'].astype(str)
        future_df['away_team'] = future_df['away_team'].astype(str)

        combined = pd.concat([master_df, future_df], ignore_index=True, sort=False)

        combined['home_team'] = combined['home_team'].astype(str)
        combined['away_team'] = combined['away_team'].astype(str)
        combined['datetime'] = pd.to_datetime(combined['datetime'], errors='coerce')

        combined = combined.drop_duplicates(
            subset=["datetime", "home_team", "away_team"],
            keep="first"
        ).sort_values("datetime").reset_index(drop=True)

    combined.to_csv(MASTER_CSV, index=False)
    print(f"Saved to {MASTER_CSV}")
    return combined


if __name__ == "__main__":
    master = download_and_build_master()
    print(f"Done. Total matches in master: {len(master)}")