#!/usr/bin/env python3
"""
download_logos.py
Downloads Premier League club SVG badges into output/logos/<Club>.svg
"""

import requests
from pathlib import Path
import time

LOGO_DIR = Path("output/logos")
LOGO_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://resources.premierleague.com/premierleague/badges/"

# Mapping of club names to Premier League badge IDs
CLUB_BADGES = {
    "Arsenal": "t3.svg",
    "Aston Villa": "t7.svg",
    "Bournemouth": "t90.svg",
    "Brentford": "t108.svg",
    "Brighton": "t36.svg",
    "Burnley": "t95.svg",
    "Chelsea": "t8.svg",
    "Crystal Palace": "t31.svg",
    "Everton": "t11.svg",
    "Fulham": "t54.svg",
    "Liverpool": "t14.svg",
    "Luton": "t188.svg",
    "Manchester City": "t43.svg",
    "Manchester United": "t1.svg",
    "Newcastle": "t4.svg",
    "Nottingham Forest": "t137.svg",
    "Tottenham": "t6.svg",
    "West Ham": "t21.svg",
    "Wolves": "t39.svg",
    # Add more if needed
}


def download_logo(club, badge_id):
    url = BASE_URL + badge_id
    outfile = LOGO_DIR / f"{club}.svg"

    try:
        print(f"Downloading: {club} ...")
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        outfile.write_bytes(resp.content)
        print(f"Saved → {outfile}")
    except Exception as e:
        print(f"FAILED: {club} → {e}")


def main():
    for club, badge in CLUB_BADGES.items():
        download_logo(club, badge)
        time.sleep(0.25)  # polite delay


if __name__ == "__main__":
    main()
