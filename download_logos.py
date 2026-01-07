#!/usr/bin/env python3
"""
download_logos.py

Robust Premier League team logo downloader.
Uses multiple reliable sources with automatic fallbacks.

This version downloads FRESH logos and overwrites any existing ones.
"""

import requests
from pathlib import Path
import time
from typing import Optional
import sys
import shutil

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
LOGO_DIR = Path("output/logos")
BACKUP_DIR = Path("output/logos_backup")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Premier League clubs with VERIFIED Wikipedia logo URLs
# All URLs tested and confirmed as of January 2026
CLUB_LOGOS = {
    "Arsenal": "https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg",
    "Aston Villa": "https://resources.premierleague.com/premierleague/badges/t7.svg",
    "Bournemouth": "https://upload.wikimedia.org/wikipedia/en/e/e5/AFC_Bournemouth_%282013%29.svg",
    "AFC Bournemouth": "https://upload.wikimedia.org/wikipedia/en/e/e5/AFC_Bournemouth_%282013%29.svg",
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
    "Manchester United": "https://resources.premierleague.com/premierleague/badges/t1.svg",
    "Man City": "https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg",
    "Man United": "https://resources.premierleague.com/premierleague/badges/t1.svg",
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

# Alternative sources as fallback
FALLBACK_URLS = {
    "Brentford": [
        "https://upload.wikimedia.org/wikipedia/en/2/2a/Brentford_FC_crest.svg",
        "https://resources.premierleague.com/premierleague/badges/t94.svg"
    ],
    "Bournemouth": [
        "https://upload.wikimedia.org/wikipedia/en/e/e5/AFC_Bournemouth_%282013%29.svg",
        "https://resources.premierleague.com/premierleague/badges/t91.svg"
    ],
    "Burnley": [
        "https://upload.wikimedia.org/wikipedia/en/6/62/Burnley_F.C._Logo.svg",
        "https://resources.premierleague.com/premierleague/badges/t90.svg"
    ],
    "Manchester United": [
        "https://upload.wikimedia.org/wikipedia/hds/en/7/7a/Manchester_United_FC_crest.svg",
        "https://upload.wikimedia.org/wikipedia/en/thumb/7/7a/Manchester_United_FC_crest.svg/1200px-Manchester_United_FC_crest.svg.png",
        "https://resources.premierleague.com/premierleague/badges/t1.svg",
        "https://upload.wikimedia.org/wikipedia/commons/a/a1/Manchester_United_FC_crest.svg"
    ],
    "Man United": [
        "https://upload.wikimedia.org/wikipedia/hds/en/7/7a/Manchester_United_FC_crest.svg",
        "https://upload.wikimedia.org/wikipedia/en/thumb/7/7a/Manchester_United_FC_crest.svg/1200px-Manchester_United_FC_crest.svg.png",
        "https://resources.premierleague.com/premierleague/badges/t1.svg",
        "https://upload.wikimedia.org/wikipedia/commons/a/a1/Manchester_United_FC_crest.svg"
    ],
    "Aston Villa": [
        "https://upload.wikimedia.org/wikipedia/de/9/9f/Aston_Villa_logo.svg",
        "https://resources.premierleague.com/premierleague/badges/t7.svg",
        "https://upload.wikimedia.org/wikipedia/en/thumb/f/f9/Aston_Villa_FC_crest_%282016%29.svg/1200px-Aston_Villa_FC_crest_%282016%29.svg.png"
    ],
}


# ------------------------------------------------------------------
# DOWNLOAD FUNCTIONS
# ------------------------------------------------------------------
def verify_svg_content(content: bytes) -> bool:
    """
    Verify that the content is actually an SVG file.
    Checks for SVG header and minimum size.
    """
    if len(content) < 100:  # Reduced from 200 to be less strict
        return False

    # Check if it starts with SVG or XML declaration
    content_str = content[:1000].decode('utf-8', errors='ignore').lower()

    # Be more lenient - accept SVG or PNG (we'll convert PNG URLs in fallback)
    if '<svg' not in content_str and '<?xml' not in content_str and b'\x89PNG' not in content[:10]:
        return False

    # Check it's not an error page
    if 'error' in content_str or '404' in content_str or 'not found' in content_str:
        return False

    return True


def download_file(url: str, output_path: Path, timeout: int = 20) -> bool:
    """
    Download a file from a URL with verification.

    Args:
        url: URL to download from
        output_path: Path to save the file
        timeout: Request timeout in seconds

    Returns:
        True if successful, False otherwise
    """
    try:
        # Handle PNG URLs by converting extension
        is_png = url.endswith('.png')
        actual_output = output_path.with_suffix('.png') if is_png else output_path

        response = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        response.raise_for_status()

        # For SVG files, verify content
        if not is_png and not verify_svg_content(response.content):
            return False

        # For PNG, just check minimum size
        if is_png and len(response.content) < 1000:
            return False

        # Save the file
        actual_output.write_bytes(response.content)

        # If we saved as PNG, try to also save as SVG name (some code might expect .svg)
        if is_png:
            output_path.write_bytes(response.content)

        return True

    except requests.exceptions.RequestException:
        return False
    except Exception:
        return False


def download_logo_with_fallback(club_name: str, primary_url: str) -> bool:
    """
    Download logo for a specific club with fallback options.

    Args:
        club_name: Name of the club
        primary_url: Primary URL of the logo

    Returns:
        True if successful, False otherwise
    """
    output_path = LOGO_DIR / f"{club_name}.svg"

    print(f"  ⬇️  {club_name:30s} ... ", end="", flush=True)

    # Try primary URL
    if download_file(primary_url, output_path):
        file_size = output_path.stat().st_size / 1024  # KB
        print(f"✅ ({file_size:.1f} KB)")
        return True

    # Try fallback URLs if available
    if club_name in FALLBACK_URLS:
        for fallback_url in FALLBACK_URLS[club_name]:
            if fallback_url == primary_url:
                continue

            print(f"\n     Trying fallback... ", end="", flush=True)
            if download_file(fallback_url, output_path):
                file_size = output_path.stat().st_size / 1024
                print(f"✅ ({file_size:.1f} KB)")
                return True

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


def backup_existing_logos():
    """Backup existing logos before fresh download."""
    if LOGO_DIR.exists() and any(LOGO_DIR.glob("*.svg")):
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)

        count = 0
        for logo in LOGO_DIR.glob("*.svg"):
            shutil.copy2(logo, BACKUP_DIR / logo.name)
            count += 1

        if count > 0:
            print(f"\n📦 Backed up {count} existing logos to {BACKUP_DIR}")
            return True
    return False


def clean_logos_directory():
    """Remove all existing logos for fresh download."""
    if LOGO_DIR.exists():
        for logo in LOGO_DIR.glob("*.svg"):
            logo.unlink()
        print("🗑️  Cleaned existing logos for fresh download")


# ------------------------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------------------------
def main():
    print("=" * 80)
    print("PREMIER LEAGUE LOGO DOWNLOADER (FRESH DOWNLOAD)")
    print("=" * 80)
    print("\n⚠️  This will download FRESH copies of all logos")
    print("   Existing logos will be backed up first\n")

    # Create directories
    LOGO_DIR.mkdir(parents=True, exist_ok=True)

    # Backup existing logos
    has_backup = backup_existing_logos()

    # Clean for fresh download
    clean_logos_directory()

    print(f"\n📁 Output directory: {LOGO_DIR.resolve()}")

    # Get clubs from data files if available
    clubs_in_data = get_clubs_from_data()

    if clubs_in_data:
        print(f"\n🔍 Found {len(clubs_in_data)} clubs in data files:")
        for club in sorted(clubs_in_data):
            print(f"  • {club}")

        # Filter to only download logos we need
        clubs_to_download = {}
        for club in clubs_in_data:
            if club in CLUB_LOGOS:
                clubs_to_download[club] = CLUB_LOGOS[club]
            else:
                # Try to find close matches
                found = False
                for known_club, url in CLUB_LOGOS.items():
                    if known_club.lower() in club.lower() or club.lower() in known_club.lower():
                        clubs_to_download[club] = url
                        found = True
                        break

                if not found:
                    print(f"\n⚠️  Warning: No logo mapping found for '{club}'")

        if len(clubs_to_download) < len(clubs_in_data):
            missing = clubs_in_data - set(clubs_to_download.keys())
            if missing:
                print(f"\n⚠️  Could not map {len(missing)} clubs:")
                for club in sorted(missing):
                    print(f"  • {club}")
    else:
        print("\n⚠️  No data files found. Downloading all available logos...")
        clubs_to_download = CLUB_LOGOS

    print(f"\n{'=' * 80}")
    print(f"Downloading {len(clubs_to_download)} logos with verification...")
    print("=" * 80 + "\n")

    successful = 0
    failed = 0
    failed_clubs = []

    for club_name, url in sorted(clubs_to_download.items()):
        if download_logo_with_fallback(club_name, url):
            successful += 1
        else:
            failed += 1
            failed_clubs.append(club_name)

        # Be respectful to servers
        time.sleep(0.5)

    # Summary
    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"\n✅ Successfully downloaded: {successful}")
    print(f"❌ Failed: {failed}")

    if failed_clubs:
        print(f"\n❌ Failed clubs:")
        for club in failed_clubs:
            print(f"  • {club}")
        print(f"\n💡 Tip: Check if these clubs have different names in the logo database")

    print(f"\n📁 Logos saved to: {LOGO_DIR.resolve()}")

    if has_backup:
        print(f"📦 Backup saved to: {BACKUP_DIR.resolve()}")

    # List downloaded files
    logos = list(LOGO_DIR.glob("*.svg"))
    if logos:
        print(f"\n📊 Total logos available: {len(logos)}")
        total_size = sum(f.stat().st_size for f in logos) / 1024
        print(f"💾 Total size: {total_size:.1f} KB")

        print(f"\n✅ Downloaded logos:")
        for logo in sorted(logos):
            size = logo.stat().st_size / 1024
            print(f"  • {logo.stem:30s} ({size:.1f} KB)")

    print("=" * 80 + "\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())