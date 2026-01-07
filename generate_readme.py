#!/usr/bin/env python3
"""
generate_readme.py

Generates a README.md file with:
- Project introduction
- Upcoming gameweek predictions
- Instructions for use
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sys

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
PREDICTIONS_FILE = Path("output/predictions_upcoming.csv")
README_FILE = Path("README.md")


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------
def group_by_gameweek(df: pd.DataFrame) -> dict:
    """
    Group matches by gameweek based on dates.
    Matches within 4 days of each other are considered same gameweek.
    """
    if df.empty:
        return {}

    df = df.sort_values("datetime").reset_index(drop=True)

    gameweeks = {}
    current_gw = 1
    current_gw_start = df.iloc[0]["datetime"]

    for idx, row in df.iterrows():
        match_date = row["datetime"]

        # If more than 4 days from current gameweek start, start new gameweek
        if (match_date - current_gw_start).days > 4:
            current_gw += 1
            current_gw_start = match_date

        if current_gw not in gameweeks:
            gameweeks[current_gw] = []

        gameweeks[current_gw].append(row)

    return gameweeks


def format_match_prediction(row) -> str:
    """Format a single match prediction for markdown."""
    home = row["home_team"]
    away = row["away_team"]
    pred_home = row["pred_home_goals"]
    pred_away = row["pred_away_goals"]
    prob_home = row["prob_home"] * 100
    prob_draw = row["prob_draw"] * 100
    prob_away = row["prob_away"] * 100

    date_str = row["datetime"].strftime("%a %d %b, %H:%M")

    return f"""
**{home}** vs **{away}**  
📅 {date_str} | 🎯 Prediction: **{pred_home:.1f} - {pred_away:.1f}**  
📊 Win Probabilities: Home {prob_home:.0f}% | Draw {prob_draw:.0f}% | Away {prob_away:.0f}%
"""


def get_winner_prediction(row) -> str:
    """Get the predicted winner."""
    prob_home = row["prob_home"]
    prob_draw = row["prob_draw"]
    prob_away = row["prob_away"]

    max_prob = max(prob_home, prob_draw, prob_away)

    if max_prob == prob_home:
        return f"**{row['home_team']}** to win"
    elif max_prob == prob_away:
        return f"**{row['away_team']}** to win"
    else:
        return "**Draw** predicted"


# ------------------------------------------------------------------
# README GENERATION
# ------------------------------------------------------------------
def generate_readme():
    """Generate the README.md file."""

    # Check if predictions file exists
    if not PREDICTIONS_FILE.exists():
        print(f"❌ Error: Predictions file not found at {PREDICTIONS_FILE}")
        print("Please run the pipeline first: python run_all.py")
        return 1

    # Load predictions
    df = pd.read_csv(PREDICTIONS_FILE, parse_dates=["datetime"])

    if df.empty:
        print("⚠️  Warning: No upcoming fixtures found")
        # Generate minimal README
        readme_content = generate_empty_readme()
        README_FILE.write_text(readme_content, encoding="utf-8")
        print(f"✅ Generated README.md (no fixtures available)")
        return 0

    print(f"📊 Loaded {len(df)} predictions")

    # Group by gameweek
    gameweeks = group_by_gameweek(df)
    print(f"📅 Found {len(gameweeks)} gameweek(s)")

    # Generate README content
    readme_content = generate_full_readme(gameweeks)

    # Write to file
    README_FILE.write_text(readme_content, encoding="utf-8")

    print(f"✅ Generated {README_FILE}")
    print(f"   Next gameweek has {len(gameweeks.get(1, []))} matches")

    return 0


def generate_empty_readme() -> str:
    """Generate README when no fixtures are available."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""# ⚽ Premier League Match Predictor

AI-powered predictions for Premier League matches using XGBoost and historical xG data.

## 📊 Current Status

**No upcoming fixtures available**

This could mean:
- The season has ended
- There's a break in fixtures
- The data needs to be updated

Run `python run_all.py` to refresh the data.

---

## 🚀 Quick Start

1. **Update data and generate predictions:**
   ```bash
   python run_all.py
   ```

2. **View the interactive dashboard:**
   ```bash
   open output/predictions_dashboard.html
   ```

3. **Download team logos (optional):**
   ```bash
   python download_logos.py
   ```

## 📁 Project Structure

- `update_data.py` - Downloads match data from FBref
- `predict_scores.py` - Trains model and generates predictions
- `visualise.py` - Creates HTML dashboard
- `run_all.py` - Runs the complete pipeline

## 🤖 How It Works

1. **Data Collection**: Scrapes historical Premier League data from FBref
2. **Feature Engineering**: Uses xG (expected goals) and team form
3. **Model Training**: XGBoost regressor trained on historical matches
4. **Prediction**: Estimates scores and win probabilities for upcoming fixtures

## 📈 Model Features

- Home/Away xG (expected goals)
- xG differential
- Home advantage factor
- Recent team form (last 5 matches)

---

*Last updated: {now}*
*Data source: FBref via soccerdata*
"""


def generate_full_readme(gameweeks: dict) -> str:
    """Generate full README with predictions."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Get next gameweek
    next_gw = min(gameweeks.keys())
    next_gw_matches = gameweeks[next_gw]

    # Header
    content = f"""# ⚽ Premier League Match Predictor

AI-powered predictions for Premier League matches using XGBoost and historical xG data.

## 🔮 Next Gameweek Predictions

### Gameweek {next_gw}

"""

    # Add each match from next gameweek
    for match in next_gw_matches:
        content += format_match_prediction(match)
        content += "\n"

    # Quick predictions summary
    content += "\n### Quick Picks\n\n"
    for match in next_gw_matches[:5]:  # Top 5 matches
        winner = get_winner_prediction(match)
        content += f"- {winner}\n"

    # All gameweeks summary
    if len(gameweeks) > 1:
        content += f"\n\n## 📅 All Upcoming Gameweeks\n\n"
        content += f"Total fixtures predicted: **{sum(len(matches) for matches in gameweeks.values())}**\n\n"

        for gw_num in sorted(gameweeks.keys()):
            matches = gameweeks[gw_num]
            first_match = matches[0]["datetime"].strftime("%d %b")
            last_match = matches[-1]["datetime"].strftime("%d %b")
            content += f"- **Gameweek {gw_num}**: {len(matches)} matches ({first_match} - {last_match})\n"

    # Footer
    content += f"""

---

## 🚀 Quick Start

1. **Update data and generate predictions:**
   ```bash
   python run_all.py
   ```

2. **View the interactive dashboard:**
   ```bash
   open output/predictions_dashboard.html
   ```

3. **Download team logos (optional):**
   ```bash
   python download_logos.py
   ```

## 📁 Project Structure

```
├── update_data.py           # Downloads and processes match data
├── predict_scores.py        # ML model training and prediction
├── visualise.py             # Generates HTML dashboard
├── download_logos.py        # Downloads team badges
├── run_all.py              # Complete pipeline runner
├── generate_readme.py      # Updates this README
├── data/
│   ├── matches_master.csv  # Historical matches
│   └── upcoming_fixtures.csv # Upcoming fixtures
└── output/
    ├── predictions_upcoming.csv
    ├── predictions_dashboard.html
    └── logos/               # Team badges
```

## 🤖 How It Works

1. **Data Collection**: Scrapes historical Premier League data from FBref using soccerdata
2. **Feature Engineering**: 
   - xG (expected goals) metrics
   - Team form (last 5 matches)
   - Home advantage factor
   - xG differential
3. **Model Training**: XGBoost regressor trained on {len(pd.read_csv('data/matches_master.csv') if Path('data/matches_master.csv').exists() else [])} historical matches
4. **Prediction**: Generates score predictions and win probabilities

## 📈 Model Performance

The model uses the following features:
- `home_xg` - Home team's expected goals
- `away_xg` - Away team's expected goals  
- `xg_diff` - Difference in xG
- `home_adv` - Home advantage factor

Trained on multiple seasons of Premier League data for robust predictions.

## 🎯 Understanding the Predictions

- **Score Prediction**: Expected goals for each team
- **Win Probabilities**: Likelihood of home win / draw / away win
- **Form-Based xG**: Uses recent performance (last 5 matches) to estimate team strength

## 📊 Data Sources

- **Match Data**: [FBref](https://fbref.com/) (via [soccerdata](https://github.com/probberechts/soccerdata))
- **Team Logos**: Wikipedia/Wikimedia Commons

## 🔄 Updating Predictions

The pipeline automatically:
1. Downloads latest match results
2. Identifies upcoming fixtures
3. Trains model on all historical data
4. Generates predictions for future matches

Run `python run_all.py` to refresh everything!

---

*Last updated: {now}*  
*Predictions generated using machine learning on historical xG data*  
*For entertainment purposes - always gamble responsibly* 🎲
"""

    return content


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    print("=" * 80)
    print("README GENERATOR")
    print("=" * 80)
    print()

    return generate_readme()


if __name__ == "__main__":
    sys.exit(main())