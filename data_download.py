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

# -----------------------
# Config
# -----------------------
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
EVENTS_DIR = OUTPUT_DIR / "match_events"
EVENTS_DIR.mkdir(exist_ok=True)

UNDERSTAT_LEAGUE_URL = "https://understat.com/league/EPL/{year}"
UNDERSTAT_MATCH_URL = "https://understat.com/match/{match_id}"

FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

PULSELIVE_FIXTURES_URL = "https://footballapi.pulselive.com/football/fixtures"

HISTORICAL_SEASONS = ["2018", "2019", "2020", "2021", "2022", "2023", "2024"]
CURRENT_SEASON_YEAR = "2025"

DELAY = 1.2
MATCH_TIME_TOLERANCE_HOURS = 12
MASTER_CSV = OUTPUT_DIR / "matches_master.csv"

# -----------------------
# Helpers
# -----------------------
def polite_sleep():
    time.sleep(DELAY)

def safe_get_text(url, session=None, timeout=25, params=None, headers=None):
    s = session or requests
    for attempt in range(3):
        try:
            r = s.get(url, timeout=timeout, params=params, headers=headers or {"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                polite_sleep()
                return r.text
            else:
                # return the response text for logging/diagnostics on non-200
                polite_sleep()
                return r.text
        except Exception:
            time.sleep(1 + attempt * 2)
    return None

def safe_get_json(url, session=None, timeout=25, params=None, headers=None):
    s = session or requests
    for attempt in range(3):
        try:
            r = s.get(url, timeout=timeout, params=params, headers=headers or {"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                polite_sleep()
                try:
                    return r.json()
                except Exception:
                    return None
            else:
                polite_sleep()
                try:
                    return r.json()
                except Exception:
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
        try:
            fixed = decoded.replace("\\'", "'")
            return json.loads(fixed)
        except Exception:
            return None

# -----------------------
# Understat scraping
# -----------------------
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
    home = record.get('h') or record.get('h_team') or record.get('home') or record.get('hTeam') or record.get('home_team')
    away = record.get('a') or record.get('a_team') or record.get('away') or record.get('aTeam') or record.get('away_team')
    out['home_team'] = home
    out['away_team'] = away
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
        for k in ['h_goals', 'a_goals', 'h_goals_ft', 'a_goals_ft', 'goals_h', 'goals_a']:
            if k in record:
                if 'h' in k:
                    home_goals = record.get(k)
                if 'a' in k:
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
    if 'xG' in record and isinstance(record['xG'], (list, tuple)) and len(record['xG']) >= 2:
        home_xg = record['xG'][0]
        away_xg = record['xG'][1]
    else:
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

# -----------------------
# FPL helpers (for mapping)
# -----------------------
def load_fpl_fixtures_and_teams():
    r = safe_get_json(FPL_BOOTSTRAP_URL)
    if r is None:
        return pd.DataFrame(), pd.DataFrame()
    teams = r.get('teams', [])
    teams_df = pd.DataFrame(teams)[['id', 'name']]
    teams_df.columns = ['team_id', 'team_name']
    fixtures = safe_get_json(FPL_FIXTURES_URL) or []
    fixtures_df = pd.DataFrame(fixtures)
    if 'kickoff_time' in fixtures_df.columns:
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
    fpl = fpl.merge(teams_df, left_on='team_h', right_on='team_id', how='left').rename(columns={'name':'team_h_name', 'team_name':'team_h_name'}).drop(columns=['team_id'], errors='ignore')
    fpl = fpl.merge(teams_df, left_on='team_a', right_on='team_id', how='left').rename(columns={'name':'team_a_name', 'team_name':'team_a_name'}).drop(columns=['team_id'], errors='ignore')
    # handle both possible column names
    if 'team_h_name' not in fpl.columns and 'team_h' in fpl.columns:
        fpl['team_h_name'] = fpl['team_h']
    if 'team_a_name' not in fpl.columns and 'team_a' in fpl.columns:
        fpl['team_a_name'] = fpl['team_a']
    fpl['team_h_norm'] = fpl['team_h_name'].astype(str).apply(normalize_name)
    fpl['team_a_norm'] = fpl['team_a_name'].astype(str).apply(normalize_name)
    candidates = fpl[(fpl['team_h_norm'] == home_us) & (fpl['team_a_norm'] == away_us)].copy()
    if candidates.empty:
        candidates = fpl[(fpl['team_h_norm'] == away_us) & (fpl['team_a_norm'] == home_us)].copy()
    under_dt = under_row.get('datetime')
    if candidates.empty:
        return None
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

# -----------------------
# Pulselive (official PL) fixtures for upcoming matches
# -----------------------
def download_pulselive_fixtures():
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Origin": "https://www.premierleague.com",
        "Referer": "https://www.premierleague.com",
        "x-requested-with": "XMLHttpRequest"
    }
    params = {
        "comps": "1",     # competition: Premier League
        "pageSize": "1000"
    }
    j = safe_get_json(PULSELIVE_FIXTURES_URL, params=params, headers=headers)
    if not j:
        return None
    items = j.get('content') or j.get('fixtures') or j.get('data') or []
    rows = []
    for it in items:
        kickoff = it.get('kickoff')
        if not kickoff:
            kickoff = it.get('kickoffDate') or it.get('kickoff_time') or it.get('kickoffDateTime')
        home = None
        away = None
        # home/away team structure
        ht = it.get('homeTeam') or it.get('h') or it.get('home')
        at = it.get('awayTeam') or it.get('a') or it.get('away')
        if isinstance(ht, dict):
            home = ht.get('teamName') or ht.get('name') or ht.get('shortName')
        else:
            home = ht
        if isinstance(at, dict):
            away = at.get('teamName') or at.get('name') or at.get('shortName')
        else:
            away = at
        matchweek = it.get('matchweek') or it.get('round') or it.get('event') or None
        try:
            kickoff_dt = pd.to_datetime(kickoff)
        except Exception:
            kickoff_dt = pd.NaT
        rows.append({
            "date": kickoff_dt,
            "home_team": home,
            "away_team": away,
            "week": matchweek,
            "source": "pulselive"
        })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["home_team", "away_team"])
    return df

# -----------------------
# Fallback: FBref fixtures (only used if pulselive fails)
# -----------------------
FBREF_FIXTURE_URL = "https://fbref.com/en/comps/9/schedule/Premier-League-Fixtures"

def clean_team_name(name):
    if name is None:
        return None
    return (
        name.replace("Manchester Utd", "Manchester United")
            .replace("Nott'ham Forest", "Nottingham Forest")
            .replace("West Ham", "West Ham United")
            .replace("Newcastle Utd", "Newcastle United")
            .replace("Brighton", "Brighton and Hove Albion")
            .replace("Wolves", "Wolverhampton Wanderers")
            .strip()
    )

def download_fbref_fixtures():
    text = safe_get_text(FBREF_FIXTURE_URL)
    if not text:
        return None
    soup = BeautifulSoup(text, "html.parser")
    table = soup.find("table")
    if table is None:
        return None
    df = pd.read_html(str(table))[0]
    df = df.rename(columns={"Wk": "week", "Date": "date", "Home": "home_team", "Away": "away_team", "Score": "score"})
    df = df[["week", "date", "home_team", "away_team", "score"]]
    df["home_team"] = df["home_team"].apply(clean_team_name)
    df["away_team"] = df["away_team"].apply(clean_team_name)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_goals"] = np.nan
    df["away_goals"] = np.nan
    played_mask = df["score"].notna()
    def split_score(s):
        try:
            h, a = s.split("–")
            return int(h), int(a)
        except Exception:
            return None, None
    df.loc[played_mask, ["home_goals", "away_goals"]] = df.loc[played_mask, "score"].apply(lambda s: pd.Series(split_score(s)))
    df = df.drop(columns=["score"])
    df = df.dropna(subset=["home_team", "away_team"])
    return df

# -----------------------
# Main orchestration
# -----------------------
def download_and_build_master():
    fixtures_df, teams_df = load_fpl_fixtures_and_teams()
    all_raw_matches = []
    for year in HISTORICAL_SEASONS + [CURRENT_SEASON_YEAR]:
        raw_matches = parse_understat_league(year)
        for rec in raw_matches:
            norm = normalize_understat_match(rec, year)
            all_raw_matches.append(norm)
    master_df = pd.DataFrame(all_raw_matches)
    if 'datetime' in master_df.columns:
        master_df['datetime'] = pd.to_datetime(master_df['datetime'], errors='coerce')
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
    cond_past = (master_df['datetime'].notna() & (master_df['datetime'] <= now)) | (master_df.get('home_goals').notna() & master_df.get('away_goals').notna())
    master_df = master_df[cond_past].copy()
    master_df['home_goals'] = pd.to_numeric(master_df.get('home_goals'), errors='coerce')
    master_df['away_goals'] = pd.to_numeric(master_df.get('away_goals'), errors='coerce')
    # download per-match events for missing matches
    for _, r in tqdm(master_df.iterrows(), total=len(master_df)):
        mid = r.get('id')
        dt = r.get('datetime')
        if pd.isna(mid):
            continue
        if pd.notna(dt) and pd.to_datetime(dt) > now:
            continue
        fpath = EVENTS_DIR / f"{mid}.json"
        if fpath.exists():
            continue
        download_match_events(mid)
    # fetch upcoming fixtures from pulselive (preferred) then fallback to fbref
    pulselive_df = download_pulselive_fixtures()
    if pulselive_df is None:
        fbref_df = download_fbref_fixtures()
        future_df = fbref_df
    else:
        future_df = pulselive_df
    if future_df is None or future_df.empty:
        print("No upcoming fixtures retrieved from pulselive or fbref.")
        combined = master_df
    else:
        # harmonize columns to match master_df
        future_df = future_df.rename(columns={"date": "datetime", "week": "event"})
        future_df['season'] = f"{CURRENT_SEASON_YEAR}-{int(CURRENT_SEASON_YEAR)+1}"
        # ensure goal columns exist
        if 'home_goals' not in future_df.columns:
            future_df['home_goals'] = np.nan
        if 'away_goals' not in future_df.columns:
            future_df['away_goals'] = np.nan
        # ensure home_xg/away_xg present
        if 'home_xg' not in future_df.columns:
            future_df['home_xg'] = np.nan
        if 'away_xg' not in future_df.columns:
            future_df['away_xg'] = np.nan
        future_df = future_df[['season', 'event', 'datetime', 'home_team', 'away_team', 'home_goals', 'away_goals', 'home_xg', 'away_xg']]
        # merge: append future fixtures to master_df and drop duplicates by date+teams
        combined = pd.concat([master_df, future_df], ignore_index=True, sort=False)
        combined = combined.drop_duplicates(subset=["datetime", "home_team", "away_team"], keep="first").sort_values("datetime").reset_index(drop=True)
    combined.to_csv(MASTER_CSV, index=False)
    return combined

# -----------------------
# Entrypoint
# -----------------------
if __name__ == "__main__":
    master = download_and_build_master()
    print("Done. Total matches in master:", len(master))
