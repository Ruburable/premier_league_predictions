import os
import pandas as pd
import soccerdata as sd
from pathlib import Path

# --- IMPORTANT: set UA BEFORE soccerdata ---
os.environ["SOCCERDATA_USER_AGENT"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)

LEAGUE = "ENG-Premier League"
SEASONS = list(range(2018, 2025))
DATA_DIR = Path("data")
OUTPUT_FILE = DATA_DIR / "matches_master.csv"

DATA_DIR.mkdir(exist_ok=True)

def load_fbref():
    fbref = sd.FBref(
        leagues=LEAGUE,
        seasons=SEASONS,
        data_dir=DATA_DIR,
        no_cache=False,
        throttle=3.0
    )

    print("Loading match schedule...")
    schedule = fbref.read_schedule()

    print("Loading team shooting stats...")
    shooting = fbref.read_team_match_stats(stat_type="shooting")

    return schedule.reset_index(), shooting.reset_index()
