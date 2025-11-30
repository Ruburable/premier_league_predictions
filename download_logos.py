#!/usr/bin/env python3
"""
download_logos_final.py — ensures all missing Premier League badges are downloaded
Tries: Official → GitHub fallback → Wikipedia (with headers)
"""

import requests
from pathlib import Path
import time
import urllib.parse

LOGO_DIR = Path("output/logos")
LOGO_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://resources.premierleague.com/premierleague/badges/"
FALLBACK_BASE = "https://raw.githubusercontent.com/luukhopman/football-logos/main/logos/England/"

CLUBS = {
    "Arsenal": "t3.svg",
    "Aston Villa": "t7.svg",
    "Bournemouth": "t90.svg",
    "Brentford": "t108.svg",
    "Brighton & Hove Albion": "t36.svg",
    "Burnley": "t95.svg",
    "Chelsea": "t8.svg",
    "Crystal Palace": "t31.svg",
    "Everton": "t11.svg",
    "Fulham": "t54.svg",
    "Liverpool": "t14.svg",
    "Luton Town": "t188.svg",
    "Manchester City": "t43.svg",
    "Manchester United": "t1.svg",
    "Newcastle United": "t4.svg",
    "Nottingham Forest": "t137.svg",
    "Tottenham Hotspur": "t6.svg",
    "West Ham United": "t21.svg",
    "Wolves": "t39.svg",
    "Sunderland": None,
    "Leeds United": None,
}

WIKI_LOGOS = {
    "Burnley": "https://upload.wikimedia.org/wikipedia/en/0/02/Burnley_FC_badge.svg",
    "Luton Town": "https://upload.wikimedia.org/wikipedia/en/5/53/Luton_Town_FC_crest.svg",
    "Nottingham Forest": "https://upload.wikimedia.org/wikipedia/en/8/8c/Nottingham_Forest_FC_logo.svg",
    "Sunderland": "https://upload.wikimedia.org/wikipedia/en/6/60/Sunderland_AFC_logo.svg",
    "Leeds United": "https://upload.wikimedia.org/wikipedia/en/0/09/Leeds_United_Logo.svg",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def try_download(url, out_path, headers=None):
    try:
        resp = requests.get(url, timeout=15, headers=headers)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        return True
    except Exception:
        return False

def slugify(name: str) -> str:
    return name.replace(" ", "%20")

def download_logo(club, badge_id):
    out = LOGO_DIR / f"{club}.svg"

    # 1. Official
    if badge_id:
        url = BASE_URL + badge_id
        if try_download(url, out):
            print(f"✅ Official badge downloaded for {club}")
            return

    # 2. GitHub fallback
    slug = slugify(club)
    url_fb = FALLBACK_BASE + f"{slug}.svg"
    if try_download(url_fb, out):
        print(f"✅ GitHub fallback badge downloaded for {club}")
        return

    # 3. Wikipedia fallback
    wiki_url = WIKI_LOGOS.get(club)
    if wiki_url:
        if try_download(wiki_url, out, headers=HEADERS):
            print(f"✅ Wikipedia badge downloaded for {club}")
            return

    print(f"❗ Failed to download badge for {club}")

def main():
    for club, bid in CLUBS.items():
        download_logo(club, bid)
        time.sleep(0.2)

if __name__ == "__main__":
    main()
