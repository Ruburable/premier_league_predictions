#!/usr/bin/env python3
"""
predict_final_table.py

Projects final Premier League table using:
1. Actual results from completed matches
2. Most likely outcomes from predicted matches
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
HISTORICAL_DATA = Path("data/matches_master.csv")
PRED_HISTORICAL = Path("output/predictions_historical.csv")
PRED_UPCOMING = Path("output/predictions_upcoming.csv")
OUTPUT_TABLE = Path("output/projected_table.csv")


def get_current_season():
    """Determine current season code (e.g., 2526 for 2025/26)."""
    now = datetime.now()
    if now.month >= 8:
        season_start = now.year
    else:
        season_start = now.year - 1
    season_end = season_start + 1
    season_code = int(f"{str(season_start)[-2:]}{str(season_end)[-2:]}")
    return season_code, f"{season_start}/{str(season_end)[-2:]}"


CURRENT_SEASON, SEASON_DISPLAY = get_current_season()


# ------------------------------------------------------------------
# TABLE CALCULATION
# ------------------------------------------------------------------
def process_match_result(table, home_team, away_team, home_goals, away_goals):
    """Update table based on a match result."""
    # Initialize teams if not in table
    for team in [home_team, away_team]:
        if team not in table:
            table[team] = {
                'played': 0,
                'won': 0,
                'drawn': 0,
                'lost': 0,
                'goals_for': 0,
                'goals_against': 0,
                'goal_difference': 0,
                'points': 0
            }

    # Update stats
    table[home_team]['played'] += 1
    table[away_team]['played'] += 1

    table[home_team]['goals_for'] += home_goals
    table[home_team]['goals_against'] += away_goals
    table[away_team]['goals_for'] += away_goals
    table[away_team]['goals_against'] += home_goals

    # Determine result
    if home_goals > away_goals:
        # Home win
        table[home_team]['won'] += 1
        table[home_team]['points'] += 3
        table[away_team]['lost'] += 1
    elif home_goals < away_goals:
        # Away win
        table[away_team]['won'] += 1
        table[away_team]['points'] += 3
        table[home_team]['lost'] += 1
    else:
        # Draw
        table[home_team]['drawn'] += 1
        table[away_team]['drawn'] += 1
        table[home_team]['points'] += 1
        table[away_team]['points'] += 1

    # Update goal difference
    table[home_team]['goal_difference'] = table[home_team]['goals_for'] - table[home_team]['goals_against']
    table[away_team]['goal_difference'] = table[away_team]['goals_for'] - table[away_team]['goals_against']

    return table


def predict_most_likely_result(pred_home, pred_away, prob_home, prob_draw, prob_away):
    """
    Determine most likely result based on probabilities.
    Returns (home_goals, away_goals) as integers.
    """
    # Find most likely outcome
    max_prob = max(prob_home, prob_draw, prob_away)

    if max_prob == prob_home:
        # Home win - use predicted scores rounded
        return round(pred_home), round(pred_away) if round(pred_away) < round(pred_home) else round(pred_home) - 1
    elif max_prob == prob_away:
        # Away win
        return round(pred_home) if round(pred_home) < round(pred_away) else round(pred_away) - 1, round(pred_away)
    else:
        # Draw
        score = round((pred_home + pred_away) / 2)
        return score, score


def build_current_table():
    """Build table from actual results so far this season."""
    print("=" * 80)
    print("BUILDING CURRENT TABLE FROM ACTUAL RESULTS")
    print("=" * 80)

    # Load historical data
    if not HISTORICAL_DATA.exists():
        print("No historical data found")
        return {}

    df = pd.read_csv(HISTORICAL_DATA, parse_dates=["datetime"])

    # Filter for current season
    current_season_df = df[df["season"] == CURRENT_SEASON].copy()

    if current_season_df.empty:
        print(f"No matches found for season {CURRENT_SEASON}")
        return {}

    print(f"\n✓ Found {len(current_season_df)} completed matches")

    # Build table
    table = {}
    for _, row in current_season_df.iterrows():
        table = process_match_result(
            table,
            row['home_team'],
            row['away_team'],
            int(row['home_goals']),
            int(row['away_goals'])
        )

    return table


def project_from_predictions(table):
    """Add predicted results to table."""
    print("\n" + "=" * 80)
    print("PROJECTING FROM UPCOMING PREDICTIONS")
    print("=" * 80)

    if not PRED_UPCOMING.exists():
        print("No predictions found")
        return table

    df = pd.read_csv(PRED_UPCOMING)

    if df.empty:
        print("No upcoming matches to predict")
        return table

    print(f"\n✓ Processing {len(df)} predicted matches")

    predicted_results = []

    for _, row in df.iterrows():
        # Get most likely result
        home_goals, away_goals = predict_most_likely_result(
            row['pred_home_goals'],
            row['pred_away_goals'],
            row['prob_home'],
            row['prob_draw'],
            row['prob_away']
        )

        predicted_results.append({
            'home': row['home_team'],
            'away': row['away_team'],
            'score': f"{home_goals}-{away_goals}"
        })

        # Update table
        table = process_match_result(
            table,
            row['home_team'],
            row['away_team'],
            home_goals,
            away_goals
        )

    # Show sample predictions
    print("\nSample predicted results:")
    for result in predicted_results[:5]:
        print(f"  {result['home']:25s} {result['score']:5s} {result['away']}")

    if len(predicted_results) > 5:
        print(f"  ... and {len(predicted_results) - 5} more")

    return table


def format_table(table):
    """Convert table dict to sorted DataFrame."""
    rows = []
    for team, stats in table.items():
        rows.append({
            'Team': team,
            'P': stats['played'],
            'W': stats['won'],
            'D': stats['drawn'],
            'L': stats['lost'],
            'GF': stats['goals_for'],
            'GA': stats['goals_against'],
            'GD': stats['goal_difference'],
            'Pts': stats['points']
        })

    df = pd.DataFrame(rows)

    # Sort by points (desc), then GD (desc), then GF (desc)
    df = df.sort_values(['Pts', 'GD', 'GF'], ascending=[False, False, False])
    df.insert(0, 'Pos', range(1, len(df) + 1))

    return df.reset_index(drop=True)


def print_table(df):
    """Print formatted table."""
    print("\n" + "=" * 80)
    print(f"PROJECTED FINAL TABLE - {SEASON_DISPLAY}")
    print("=" * 80)
    print()

    # Header
    print(f"{'Pos':<4} {'Team':<25} {'P':<4} {'W':<4} {'D':<4} {'L':<4} {'GF':<5} {'GA':<5} {'GD':<6} {'Pts':<5}")
    print("-" * 80)

    # Rows with formatting
    for _, row in df.iterrows():
        pos = row['Pos']

        # Position indicators
        if pos <= 4:
            indicator = "CL"  # Champions League
        elif pos == 5:
            indicator = "EL"  # Europa League
        elif pos >= 18:
            indicator = "RL"  # Relegation
        else:
            indicator = "  "

        print(f"{indicator} {pos:<2} {row['Team']:<25} {row['P']:<4} {row['W']:<4} {row['D']:<4} {row['L']:<4} "
              f"{row['GF']:<5} {row['GA']:<5} {row['GD']:<+6} {row['Pts']:<5}")

    print()
    print("CL = Champions League  |  EL = Europa League  |  RL = Relegation")
    print("=" * 80)


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    print("\n" + "=" * 80)
    print("SEASON TABLE PROJECTION")
    print("=" * 80)
    print(f"\nSeason: {SEASON_DISPLAY}")
    print(f"Method: Actual results + Most likely predicted outcomes")

    # Build table from actual results
    table = build_current_table()

    if not table:
        print("\n No data available to build table")
        return 1

    # Project from predictions
    table = project_from_predictions(table)

    # Format and display
    df_table = format_table(table)
    print_table(df_table)

    # Save to CSV
    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    df_table.to_csv(OUTPUT_TABLE, index=False)

    print(f"\n Table saved to: {OUTPUT_TABLE.resolve()}")

    # Show top and bottom
    print("\nTop 4 (Champions League):")
    for _, row in df_table.head(4).iterrows():
        print(f"  {row['Pos']}. {row['Team']} - {row['Pts']} pts")

    print("\nBottom 3 (Relegation):")
    for _, row in df_table.tail(3).iterrows():
        print(f"  {row['Pos']}. {row['Team']} - {row['Pts']} pts")

    print("\n" + "=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())