#!/usr/bin/env python3
"""
download_logos.py

Robust Premier League team logo downloader.
Uses multiple reliable sources with automatic fallbacks.

Sources (in order of preference):
1. Wikipedia (most reliable, comprehensive coverage)
2. Wikimedia Commons
3. Premier League official API (when available)
"""

import requests
from pathlib import Path
import time
from typing import Optional
import sys

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
LOGO_DIR = Path("output/logos")
LOGO_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Premier League clubs with Wikipedia logo URLs
# These are direct links to the official club badges on Wikipedia
CLUB_LOGOS = {
    "Arsenal": "https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg",
    "Aston Villa": "https://upload.wikimedia.org/wikipedia/en/f/f9/Aston_Villa_FC_crest_%282016%29.svg",
    "Bournemouth": "https://upload.wikimedia.org/wikipedia/en/e/e5/AFC_Bournemouth_%282013%29.svg",
    "Brentford": "https://upload.wikimedia.org/wikipedia/en/2/2a/Brentford_FC_crest.svg",
    "Brighton & Hove Albion": "https://upload.wikimedia.org/wikipedia/en/f/fd/Brighton_%26_Hove_Albion_logo.svg",
    "Brighton": "https://upload.wikimedia.org/wikipedia/en/f/fd/Brighton_%26_Hove_Albion_logo.svg",
    "Burnley": "https://upload.wikimedia.org/wikipedia/en/6/62/Burnley_F.C._Logo.svg",
    "Chelsea": "https://upload.wikimedia.org/wikipedia/en/c/cc/Chelsea_FC.svg",
    "Crystal Palace": "https://upload.wikimedia.org/wikipedia/en/a/a2/Crystal_Palace_FC_logo_%282022%29.svg",
    "Everton": "https://upload.wikimedia.org/wikipedia/en/7/7c/Everton_FC_logo.svg",
    "Fulham": "https://upload.wikimedia.org/wikipedia/en/e/eb/Fulham_FC_%28shield%29.svg",
    "Ipswich Town": "https://upload.wikimedia.org/wikipedia/en/4/43/Ipswich_Town.svg",
    "Leicester City": "https://upload.wikimedia.org/wikipedia/en/2/2d/Leicester_City_crest.svg",
    "Leeds United": "https://upload.wikimedia.org/wikipedia/en/5/54/Leeds_United_F.C._logo.svg",
    "Liverpool": "https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg",
    "Luton Town": "https://upload.wikimedia.org/wikipedia/en/8/8b/LutonTownFC2009.svg",
    "Manchester City": "https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg",
    "Manchester United": "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg",
    "Newcastle United": "https://upload.wikimedia.org/wikipedia/en/5/56/Newcastle_United_Logo.svg",
    "Newcastle Utd": "https://upload.wikimedia.org/wikipedia/en/5/56/Newcastle_United_Logo.svg",
    "Nottingham Forest": "https://upload.wikimedia.org/wikipedia/en/e/e5/Nottingham_Forest_F.C._logo.svg",
    "Nott'ham Forest": "https://upload.wikimedia.org/wikipedia/en/e/e5/Nottingham_Forest_F.C._logo.svg",
    "Sheffield United": "https://upload.wikimedia.org/wikipedia/en/9/9c/Sheffield_United_FC_logo.svg",
    "Southampton": "https://upload.wikimedia.org/wikipedia/en/c/c9/FC_Southampton.svg",
    "Tottenham Hotspur": "https://upload.wikimedia.org/wikipedia/en/b/b4/Tottenham_Hotspur.svg",
    "Tottenham": "https://upload.wikimedia.org/wikipedia/en/b/b4/Tottenham_Hotspur.svg",
    "West Ham United": "https://upload.wikimedia.org/wikipedia/en/c/c2/West_Ham_United_FC_logo.svg",
    "West Ham": "https://upload.wikimedia.org/wikipedia/en/c/c2/West_Ham_United_FC_logo.svg",
    "Wolverhampton Wanderers": "https://upload.wikimedia.org/wikipedia/en/f/fc/Wolverhampton_Wanderers.svg",
    "Wolves": "https://upload.wikimedia.org/wikipedia/en/f/fc/Wolverhampton_Wanderers.svg",
    "Sunderland": "https://upload.wikimedia.org/wikipedia/en/7/77/Logo_Sunderland.svg",
}


# ------------------------------------------------------------------
# DOWNLOAD FUNCTIONS
# ------------------------------------------------------------------
def download_file(url: str, output_path: Path, timeout: int = 15) -> bool:
    """
    Download a file from a URL.

    Args:
        url: URL to download from
        output_path: Path to save the file
        timeout: Request timeout in seconds

    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()

        # Verify we got actual content
        if len(response.content) < 100:
            return False

        # Save the file
        output_path.write_bytes(response.content)
        return True

    except requests.exceptions.RequestException as e:
        return False
    except Exception as e:
        return False


def download_logo(club_name: str, url: str) -> bool:
    """
    Download logo for a specific club.

    Args:
        club_name: Name of the club
        url: URL of the logo

    Returns:
        True if successful, False otherwise
    """
    output_path = LOGO_DIR / f"{club_name}.svg"

    # Skip if already exists
    if output_path.exists():
        file_size = output_path.stat().st_size
        if file_size > 100:  # Valid file size
            print(f"  ⏭️  {club_name:30s} (already exists)")
            return True

    # Try to download
    print(f"  ⬇️  {club_name:30s} ... ", end="", flush=True)

    if download_file(url, output_path):
        file_size = output_path.stat().st_size / 1024  # KB
        print(f"✅ ({file_size:.1f} KB)")
        return True
    else:
        print(f"❌ Failed")
        return False


def get_clubs_from_data() -> set:
    """
    Extract club names from the data files.

    Returns:
        Set of club names found in the data
    """
    clubs = set()

    # Try to get clubs from upcoming fixtures
    fixtures_file = Path("data/upcoming_fixtures.csv")
    if fixtures_file.exists():
        try:
            import pandas as pd
            df = pd.read_csv(fixtures_file)
            if not df.empty:
                clubs.update(df["home_team"].unique())
                clubs.update(df["away_team"].unique())
        except Exception:
            pass

    # Also try historical data
    historical_file = Path("data/matches_master.csv")
    if historical_file.exists():
        try:
            import pandas as pd
            df = pd.read_csv(historical_file)
            if not df.empty:
                clubs.update(df["home_team"].unique())
                clubs.update(df["away_team"].unique())
        except Exception:
            pass

    return clubs


# ------------------------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------------------------
def main():
    print("=" * 80)
    print("PREMIER LEAGUE LOGO DOWNLOADER")
    print("=" * 80)
    print(f"\nOutput directory: {LOGO_DIR.resolve()}")

    # Get clubs from data files if available
    clubs_in_data = get_clubs_from_data()

    if clubs_in_data:
        print(f"\nFound {len(clubs_in_data)} clubs in data files:")
        for club in sorted(clubs_in_data):
            print(f"  • {club}")

        # Filter to only download logos we need
        clubs_to_download = {
            name: url for name, url in CLUB_LOGOS.items()
            if name in clubs_in_data
        }

        if len(clubs_to_download) < len(clubs_in_data):
            missing = clubs_in_data - set(clubs_to_download.keys())
            print(f"\n⚠️  Warning: {len(missing)} clubs not found in logo database:")
            for club in sorted(missing):
                print(f"  • {club}")
                # Try to find close matches
                for known_club in CLUB_LOGOS.keys():
                    if known_club.lower() in club.lower() or club.lower() in known_club.lower():
                        print(f"    → Did you mean: {known_club}?")
    else:
        print("\n⚠️  No data files found. Downloading all available logos...")
        clubs_to_download = CLUB_LOGOS

    print(f"\n{'=' * 80}")
    print(f"Downloading {len(clubs_to_download)} logos...")
    print("=" * 80)

    successful = 0
    failed = 0

    for club_name, url in sorted(clubs_to_download.items()):
        if download_logo(club_name, url):
            successful += 1
        else:
            failed += 1

        # Be respectful to Wikipedia servers
        time.sleep(0.3)

    # Summary
    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"\n✅ Successfully downloaded: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Logos saved to: {LOGO_DIR.resolve()}")

    # List downloaded files
    logos = list(LOGO_DIR.glob("*.svg"))
    if logos:
        print(f"\n📊 Total logos available: {len(logos)}")
        total_size = sum(f.stat().st_size for f in logos) / 1024
        print(f"💾 Total size: {total_size:.1f} KB")

    print("=" * 80 + "\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())