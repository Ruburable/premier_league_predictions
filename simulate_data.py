import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = "output"
EVENTS_DIR = "events"

NUM_PAST_MATCHES = 380        # one full PL season
NUM_TEAMS = 20
TEAMS = [
    "Arsenal","Aston Villa","Bournemouth","Brentford","Brighton","Burnley","Chelsea",
    "Crystal Palace","Everton","Fulham","Liverpool","Luton","Man City","Man United",
    "Newcastle","Nottingham Forest","Sheffield United","Tottenham","West Ham","Wolves"
]

# ============================================================
# UTILS
# ============================================================

def random_xg():
    """Return a random xG-like value."""
    return round(np.random.uniform(0.1, 2.5), 2)

def random_score():
    return np.random.poisson(1.4)

def random_player():
    return f"Player_{random.randint(1, 300)}"

# ============================================================
# 1. Simulate MASTER MATCH DATA
# ============================================================

def simulate_master_matches():
    rows = []
    start_date = datetime(2024, 8, 10)

    for i in range(NUM_PAST_MATCHES):
        home, away = random.sample(TEAMS, 2)

        date = start_date + timedelta(days=i // 10 * 7)
        home_goals = random_score()
        away_goals = random_score()

        rows.append({
            "match_id": i + 1,
            "season": "2024/25",
            "date": date.strftime("%Y-%m-%d"),
            "home_team": home,
            "away_team": away,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "home_xg": random_xg(),
            "away_xg": random_xg(),
            "week": (i // 10) + 1
        })

    df = pd.DataFrame(rows)
    return df


# ============================================================
# 2. Simulate PLAYER MATCH STATS
# ============================================================

def simulate_player_stats():
    rows = []
    player_pool = [f"Player_{i}" for i in range(1, 301)]

    for p in player_pool:
        rows.append({
            "player": p,
            "team": random.choice(TEAMS),
            "minutes": random.randint(300, 2700),
            "goals": random.randint(0, 18),
            "xG": round(np.random.uniform(0, 12), 2),
            "shots": random.randint(3, 60),
        })

    return pd.DataFrame(rows)


# ============================================================
# 3. Simulate EVENT JSON FILES
# ============================================================

def simulate_event_jsons(matches_df):
    if not os.path.exists(EVENTS_DIR):
        os.makedirs(EVENTS_DIR)

    for _, row in matches_df.iterrows():
        match_id = row["match_id"]
        events = []
        total_goals = row["home_goals"] + row["away_goals"]

        for _ in range(total_goals):
            events.append({
                "event_type": "goal",
                "player": random_player(),
                "team": random.choice([row["home_team"], row["away_team"]]),
                "minute": random.randint(1, 90)
            })

        with open(f"{EVENTS_DIR}/{match_id}.json", "w") as f:
            json.dump(events, f, indent=2)


# ============================================================
# 4. Simulate UPCOMING FIXTURES
# ============================================================

def simulate_fixtures():
    fixtures = []
    today = datetime.today()

    for i in range(10):
        home, away = random.sample(TEAMS, 2)
        fixtures.append({
            "date": (today + timedelta(days=i+1)).strftime("%Y-%m-%d"),
            "home_team": home,
            "away_team": away,
            "week": 39
        })

    return pd.DataFrame(fixtures)


# ============================================================
# MAIN
# ============================================================

def main():
    print("Simulating data...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Master dataset
    master = simulate_master_matches()
    master.to_csv(f"{OUTPUT_DIR}/matches_master.csv", index=False)
    print(f"✔ Saved master dataset → {OUTPUT_DIR}/matches_master.csv")

    # Player data
    players = simulate_player_stats()
    players.to_csv(f"{OUTPUT_DIR}/players_master.csv", index=False)
    print(f"✔ Saved player stats → {OUTPUT_DIR}/players_master.csv")

    # Events
    simulate_event_jsons(master)
    print(f"✔ Created {len(master)} event JSON files → {EVENTS_DIR}/")

    # Fixtures
    fixtures = simulate_fixtures()
    fixtures.to_csv(f"{OUTPUT_DIR}/upcoming_fixtures.csv", index=False)
    print(f"✔ Saved simulated fixtures → {OUTPUT_DIR}/upcoming_fixtures.csv")

    print("\nAll simulated datasets created successfully!")


if __name__ == "__main__":
    main()
