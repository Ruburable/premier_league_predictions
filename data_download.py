#!/usr/bin/env python3
"""
data_download.py

Purpose:
  - Full-season Understat scraping (Option B: per-match event logs + match aggregates)
  - Resumable: saves per-match JSON to disk so interrupted runs can resume without re-downloading everything
  - Polite: rate limiting + retries
  - Outputs:
      data/understat_matches_{season}.csv   -- match-level table from league page
      data/understat_matches_all.csv        -- concatenation across seasons
      data/understat_match_events/{match_id}.json  -- raw per-match JSON data files for events/shots/etc.

Notes:
  - Understat site layout may change; the script uses robust regex parsing of embedded JSON.
  - This script uses only public web pages (no API key).
  - Use responsibly and do not hammer the site.
"""

import os
import re
import time
import json
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# -----------------------
# Configuration
# -----------------------

OUTPUT_DIR = Path("data")
MATCH_EVENTS_DIR = OUTPUT_DIR / "understat_match_events"
MATCH_SUMMARY_FILENAME = OUTPUT_DIR / "understat_matches_all.csv"  # final combined table

# Per-season file pattern (match-level summary per season)
PER_SEASON_MATCHES_FMT = OUTPUT_DIR / "understat_matches_{season}.csv"

# User agent header (polite)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DataScraper/1.0; +https://example.com/)"
}

# Politeness / retry config
REQUEST_DELAY_SECONDS = 1.5  # baseline delay between requests (increase if you want)
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # multiply delay on retry

# Seasons to scrape -- format "YYYY-YY"
SEASONS = [
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26"
]

# Understat league URL template: note it expects first year (e.g. 2024 for 2024-25)
UNDERSTAT_LEAGUE_URL = "https://understat.com/league/EPL/{year}"
UNDERSTAT_MATCH_URL = "https://understat.com/match/{match_id}"

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# -----------------------
# Utility helpers
# -----------------------

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MATCH_EVENTS_DIR.mkdir(parents=True, exist_ok=True)


def safe_request(url: str, session: requests.Session, max_retries: int = MAX_RETRIES) -> Optional[str]:
    """
    Perform GET with retries and backoff. Returns response.text or None on failure.
    """
    delay = REQUEST_DELAY_SECONDS
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=25)
            if resp.status_code == 200:
                time.sleep(REQUEST_DELAY_SECONDS)  # baseline polite delay after successful request
                return resp.text
            else:
                logging.warning("Non-200 status %s for %s (attempt %d)", resp.status_code, url, attempt)
        except requests.RequestException as e:
            logging.warning("Request error for %s on attempt %d: %s", url, attempt, e)

        # backoff and retry
        time.sleep(delay)
        delay *= RETRY_BACKOFF
    logging.error("Failed to fetch %s after %d attempts", url, max_retries)
    return None


def extract_json_from_script_block(js_text: str, variable_name_patterns: List[str]) -> Optional[str]:
    """
    Search js_text for JSON.parse('...') blocks assigned to one of variable_name_patterns.
    Returns the unescaped JSON string if found, else None.

    variable_name_patterns example: ["matchesData", "datesData"]
    """
    for var in variable_name_patterns:
        # pattern: var matchesData = JSON.parse('...'); or matchesData = JSON.parse('...'); sometimes double quotes appear
        pattern = re.compile(rf"{re.escape(var)}\s*=\s*JSON\.parse\('(?P<json>.+?)'\)", flags=re.DOTALL)
        m = pattern.search(js_text)
        if m:
            raw = m.group("json")
            # unescape JavaScript-encoded string into valid JSON
            try:
                decoded = raw.encode("utf-8").decode("unicode_escape")
            except Exception:
                decoded = raw
            return decoded

    # fallback: some pages embed JSON directly as e.g. "matches": [...]
    # try to find "matches": [ ... ] block
    m2 = re.search(r"\"matches\"\s*:\s*(\[\s*\{.+?\}\s*\])\s*,\s*\"teams\"", js_text, flags=re.DOTALL)
    if m2:
        return m2.group(1)

    return None


# -----------------------
# Understat league scraper
# -----------------------

def scrape_understat_league(season: str, session: requests.Session) -> Optional[pd.DataFrame]:
    """
    Scrape the Understat league page for a season and return the matchesData table as a DataFrame.
    season: "YYYY-YY" (e.g. "2024-25") -> year used in URL is first part (YYYY)
    """
    year = int(season.split("-")[0])
    url = UNDERSTAT_LEAGUE_URL.format(year=year)
    logging.info("Fetching Understat league page for season %s -> %s", season, url)
    html = safe_request(url, session)
    if html is None:
        logging.error("Failed to fetch league page for %s", season)
        return None

    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.find_all("script")
    json_text = None
    for script in scripts:
        if script.string and "matchesData" in script.string:
            json_text = extract_json_from_script_block(script.string, ["matchesData", "datesData", "matches"])
            if json_text:
                break

    if not json_text:
        # sometimes the JS block is minified and not easily readable; try entire page
        json_text = extract_json_from_script_block(html, ["matchesData", "datesData", "matches"])

    if not json_text:
        logging.error("Could not locate matches JSON for season %s", season)
        return None

    # parse JSON into Python objects
    try:
        raw = json.loads(json_text)
    except Exception as e:
        logging.exception("Failed to parse matches JSON for %s: %s", season, e)
        return None

    # raw is usually a list of match dicts nested in dates or directly
    # Normalize into a flat list of dicts
    matches_list = []
    if isinstance(raw, dict):
        # If parsed as object containing 'matches' key
        if "matches" in raw:
            matches_list = raw["matches"]
        else:
            # some structure unexpected
            logging.warning("Parsed JSON is dict without 'matches' key; trying to extract arrays")
            # try to find lists inside
            for v in raw.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    matches_list = v
                    break
    elif isinstance(raw, list):
        # could be list of date sections where each has 'matches'
        # detect inner structure
        if raw and isinstance(raw[0], dict) and "matches" in raw[0]:
            for day in raw:
                if isinstance(day, dict) and "matches" in day:
                    matches_list.extend(day["matches"])
        else:
            matches_list = raw
    else:
        logging.error("Unexpected JSON shape for season %s", season)
        return None

    # Convert to DataFrame
    df = pd.DataFrame(matches_list)

    # Normalize column names known in Understat's matches data
    # Understat field names vary; attempt to standardize common ones we want
    rename_map = {}
    # common keys in Understat matches data: id, date, h_team, a_team, h_goals, a_goals, h_shot, a_shot, xG, xGA, ...
    # map likely candidates to consistent names
    for cand in [
        ("h_team", "home_team"),
        ("a_team", "away_team"),
        ("h_goals", "home_goals"),
        ("a_goals", "away_goals"),
        ("h_shot", "home_shots"),
        ("a_shot", "away_shots"),
        ("hxG", "home_xg"),
        ("axG", "away_xg"),
        ("xG", "home_xg"),    # sometimes xG refers to home xG (rare)
        ("xGA", "away_xg"),
        ("id", "match_id"),
        ("date", "date")
    ]:
        if cand[0] in df.columns and cand[1] not in df.columns:
            rename_map[cand[0]] = cand[1]

    if rename_map:
        df = df.rename(columns=rename_map)

    # Additional normalization: ensure date is parsed to timestamp if present
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception:
            pass

    # Add season tag and original source indicator
    df["season"] = season
    df["source"] = "understat_league"

    logging.info("Extracted %d matches for season %s from Understat league page", len(df), season)
    return df


# -----------------------
# Understat per-match scraper (events + shots)
# -----------------------

def scrape_understat_match(match_id: str, session: requests.Session) -> Optional[Dict[str, Any]]:
    """
    Scrape the full match page for a given understat match_id and extract any embedded JSON blocks.
    Returns a dict containing raw parsed JSON fragments (shots, events, teams, etc.) or None on failure.
    The match page typically contains:
      - 'shotsData' JSON (list of shot objects with xG, player, minute, result, etc.)
      - possibly 'h' / 'a' or other objects
    """
    url = UNDERSTAT_MATCH_URL.format(match_id=match_id)
    html = safe_request(url, session)
    if html is None:
        return None

    # Parse scripts and search for JSON.parse('...') occurrences or variable assignments
    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.find_all("script")
    extracted = {}

    for script in scripts:
        text = script.string
        if not text:
            continue
        # common variable names used in Understat match pages:
        #   shotsData
        #   h, a (home/away dictionaries)
        #   additionalData
        # We'll attempt to extract these blocks.

        # 1) shotsData = JSON.parse('....');
        m = re.search(r"shotsData\s*=\s*JSON\.parse\('(?P<json>.+?)'\)", text, flags=re.DOTALL)
        if m:
            raw = m.group("json")
            try:
                decoded = raw.encode("utf-8").decode("unicode_escape")
                shots = json.loads(decoded)
                extracted["shotsData"] = shots
            except Exception:
                # try safe replacements
                try:
                    decoded = raw.replace("\\'", "'").encode("utf-8").decode("unicode_escape")
                    shots = json.loads(decoded)
                    extracted["shotsData"] = shots
                except Exception as e:
                    logging.warning("Failed to parse shotsData for match %s: %s", match_id, e)

        # 2) look for 'h' and 'a' objects assigned as JSON.parse('...') or as JS objects
        # pattern: var h = JSON.parse('...'); or h = JSON.parse('...');
        for team_var in ["h", "a", "home", "away"]:
            pattern = re.compile(rf"{team_var}\s*=\s*JSON\.parse\('(?P<json>.+?)'\)", flags=re.DOTALL)
            mm = pattern.search(text)
            if mm:
                raw = mm.group("json")
                try:
                    decoded = raw.encode("utf-8").decode("unicode_escape")
                    data = json.loads(decoded)
                    extracted[team_var] = data
                except Exception:
                    try:
                        decoded = raw.replace("\\'", "'").encode("utf-8").decode("unicode_escape")
                        data = json.loads(decoded)
                        extracted[team_var] = data
                    except Exception as e:
                        logging.warning("Failed to parse %s for match %s: %s", team_var, match_id, e)

        # 3) fallback: if script contains JSON-like arrays/objects we can try to find them by keys
        # find any JSON.parse occurrence and try to decode
        generic_matches = re.findall(r"JSON\.parse\('(.+?)'\)", text, flags=re.DOTALL)
        for raw in generic_matches:
            try:
                decoded = raw.encode("utf-8").decode("unicode_escape")
                parsed = json.loads(decoded)
                # heuristics: if parsed is list of dicts and has 'xG' or 'player' -> it's shots
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    keys = set(parsed[0].keys())
                    if {"xG", "player", "minute"} & keys:
                        extracted.setdefault("shotsData", parsed)
            except Exception:
                continue

    # If nothing extracted, attempt a looser parse: look for '"shots": [ ... ]' block
    if not extracted:
        text_all = soup.get_text()
        m2 = re.search(r"\"shots\"\s*:\s*(\[[\s\S]+?\])\s*,\s*\"something\"", text_all)
        if m2:
            try:
                parsed = json.loads(m2.group(1))
                extracted["shotsData"] = parsed
            except Exception:
                pass

    if not extracted:
        logging.warning("No JSON blocks extracted from match page %s", match_id)
        return None

    # attach metadata: match_id and timestamp
    extracted["_match_id"] = match_id
    extracted["_url"] = url
    extracted["_fetched_at"] = pd.Timestamp.utcnow().isoformat()

    return extracted


# -----------------------
# Orchestration: league -> per-match
# -----------------------

def season_to_year(season: str) -> int:
    """Return first-year integer used by Understat URL for given season string like '2024-25' -> 2024"""
    return int(season.split("-")[0])


def download_season_understat(season: str, session: requests.Session, resume: bool = True) -> Optional[pd.DataFrame]:
    """
    For a single season:
      - scrape league page, save per-season matches CSV
      - for every match row (match_id present), scrape match page and save per-match JSON to disk
    Returns: DataFrame of the league matches (pandas) or None on failure
    """
    logging.info("Starting download for season %s", season)
    df_matches = scrape_understat_league(season, session)
    if df_matches is None:
        logging.error("Failed to extract league matches for %s", season)
        return None

    # Save per-season matches table
    season_file = PER_SEASON_MATCHES_FMT.with_name(PER_SEASON_MATCHES_FMT.name.format(season=season))
    try:
        df_matches.to_csv(season_file, index=False)
        logging.info("Saved season match table: %s", season_file)
    except Exception as e:
        logging.exception("Failed saving season CSV: %s", e)

    # Determine match ids from DataFrame
    # Understat sometimes uses 'id' or 'match_id' key in league matches JSON
    match_id_col = None
    for candidate in ("id", "match_id", "id_match"):
        if candidate in df_matches.columns:
            match_id_col = candidate
            break

    # Some league JSON does not surface internal id; if not present, try to construct match_id from date+teams
    if match_id_col is None:
        logging.warning("No explicit match id column found in league matches for %s. We'll try to build keys from team+date.", season)

    # iterate matches and fetch per-match event JSON
    for _, row in tqdm(df_matches.iterrows(), total=len(df_matches), desc=f"Matches {season}", unit="match"):
        # determine an identifier to save file under
        if match_id_col:
            mid = str(row[match_id_col])
        else:
            # fallback: safe unique key combining date + home + away
            home = str(row.get("home_team") or row.get("h_team") or row.get("hTeam") or "")
            away = str(row.get("away_team") or row.get("a_team") or row.get("aTeam") or "")
            date = str(row.get("date") or row.get("match_date") or "")
            mid = f"{season}_{home}_vs_{away}_{date}".replace(" ", "_").replace(":", "-")

        out_path = MATCH_EVENTS_DIR / f"{mid}.json"
        if resume and out_path.exists():
            # already fetched
            continue

        # If we have a direct Understat match id numeric string, use match URL
        # match URL expects Understat internal id; league JSON often includes it as 'id'
        if match_id_col:
            match_url_id = row[match_id_col]
            # build match page URL and scrape
            try:
                extracted = scrape_understat_match(match_url_id, session)
            except Exception as e:
                logging.exception("Error scraping match %s: %s", match_url_id, e)
                extracted = None
        else:
            # Without explicit match id, we can't hit match page reliably; skip saving per-match events
            logging.debug("No id for match; skipping per-match page fetch for %s", mid)
            extracted = None

        # Save extracted JSON if any
        if extracted:
            try:
                with open(out_path, "w", encoding="utf-8") as fh:
                    json.dump(extracted, fh, ensure_ascii=False, indent=2)
            except Exception as e:
                logging.exception("Failed to save match JSON for %s: %s", mid, e)
        else:
            # create a small marker file to indicate attempted (to avoid repeated attempts)
            try:
                with open(out_path.with_suffix(".failed"), "w", encoding="utf-8") as fh:
                    fh.write(json.dumps({"attempted": True, "timestamp": pd.Timestamp.utcnow().isoformat()}))
            except Exception:
                pass

    return df_matches


def merge_all_seasons(seasons: List[str]) -> None:
    """
    After seasons are downloaded individually, combine all per-season CSVs into a single file.
    """
    frames = []
    for s in seasons:
        path = PER_SEASON_MATCHES_FMT.with_name(PER_SEASON_MATCHES_FMT.name.format(season=s))
        if path.exists():
            try:
                df = pd.read_csv(path)
                df["season"] = s
                frames.append(df)
            except Exception as e:
                logging.exception("Failed reading season file %s: %s", path, e)
        else:
            logging.warning("Season file missing: %s", path)

    if frames:
        all_df = pd.concat(frames, ignore_index=True, sort=False)
        # Attempt to normalize common columns
        # Try to rename Understat-like columns into consistent names
        col_renames = {}
        if "h_team" in all_df.columns and "home_team" not in all_df.columns:
            col_renames["h_team"] = "home_team"
        if "a_team" in all_df.columns and "away_team" not in all_df.columns:
            col_renames["a_team"] = "away_team"
        if "h_shot" in all_df.columns and "home_shots" not in all_df.columns:
            col_renames["h_shot"] = "home_shots"
        if "a_shot" in all_df.columns and "away_shots" not in all_df.columns:
            col_renames["a_shot"] = "away_shots"
        if col_renames:
            all_df = all_df.rename(columns=col_renames)

        try:
            all_df.to_csv(MATCH_SUMMARY_FILENAME, index=False)
            logging.info("Wrote combined matches file: %s", MATCH_SUMMARY_FILENAME)
        except Exception as e:
            logging.exception("Failed to write combined matches CSV: %s", e)
    else:
        logging.error("No per-season match files to merge.")


# -----------------------
# Main entry-point
# -----------------------

def main():
    ensure_dirs()
    s = requests.Session()

    seasons_to_process = SEASONS.copy()

    for season in seasons_to_process:
        try:
            _ = download_season_understat(season, s, resume=True)
        except Exception as e:
            logging.exception("Error processing season %s: %s", season, e)

    # After all seasons processed, merge
    merge_all_seasons(seasons_to_process)
    logging.info("All done. Data saved in %s", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
