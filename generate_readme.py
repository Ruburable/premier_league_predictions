#!/usr/bin/env python3
"""
generate_readme.py

Generates a professional README.md file with:
- Project overview
- Model performance statistics
- Upcoming gameweek predictions
- Technical documentation
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
PREDICTIONS_UPCOMING = Path("output/predictions_upcoming.csv")
PREDICTIONS_HISTORICAL = Path("output/predictions_historical.csv")
README_FILE = Path("README.md")


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------
def get_current_season():
    """Determine current Premier League season."""
    now = datetime.now()
    if now.month >= 8:
        season_year = now.year
    else:
        season_year = now.year - 1
    return season_year, f"{season_year}/{str(season_year + 1)[-2:]}"


def group_by_gameweek(df: pd.DataFrame) -> dict:
    """Group matches by gameweek based on dates."""
    if df.empty:
        return {}

    df = df.sort_values("datetime").reset_index(drop=True)

    gameweeks = {}
    current_gw = 1
    current_gw_start = df.iloc[0]["datetime"]

    for idx, row in df.iterrows():
        match_date = row["datetime"]

        if (match_date - current_gw_start).days > 4:
            current_gw += 1
            current_gw_start = match_date

        if current_gw not in gameweeks:
            gameweeks[current_gw] = []

        gameweeks[current_gw].append(row)

    return gameweeks


def calculate_performance_metrics(df: pd.DataFrame) -> dict:
    """Calculate model performance metrics."""
    if df.empty:
        return None

    mae_home = abs(df["home_goals"] - df["pred_home_goals"]).mean()
    mae_away = abs(df["away_goals"] - df["pred_away_goals"]).mean()
    avg_mae = (mae_home + mae_away) / 2

    # Accuracy within 1 goal
    accurate = (
            (abs(df["home_goals"] - df["pred_home_goals"]) <= 1.0) &
            (abs(df["away_goals"] - df["pred_away_goals"]) <= 1.0)
    ).sum()
    accuracy_pct = (accurate / len(df) * 100) if len(df) > 0 else 0

    # Perfect predictions
    perfect = (
            (df["home_goals"] == df["pred_home_goals"].round()) &
            (df["away_goals"] == df["pred_away_goals"].round())
    ).sum()
    perfect_pct = (perfect / len(df) * 100) if len(df) > 0 else 0

    return {
        "total_matches": len(df),
        "mae_home": mae_home,
        "mae_away": mae_away,
        "avg_mae": avg_mae,
        "accuracy_pct": accuracy_pct,
        "perfect_predictions": perfect,
        "perfect_pct": perfect_pct,
    }


# ------------------------------------------------------------------
# README GENERATORS
# ------------------------------------------------------------------
def generate_empty_readme() -> str:
    """Generate README when no fixtures are available."""
    season_year, season_display = get_current_season()
    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    return f"""# Premier League Match Predictor

Machine learning system for predicting Premier League match outcomes using XGBoost and expected goals (xG) data.

**Season:** {season_display}  
**Status:** No upcoming fixtures available

---

## Quick Start

Update data and generate predictions:

```bash
python run_all.py
```

View the interactive dashboard:

```bash
open output/predictions_dashboard.html
```

---

## Project Structure

```
├── update_data.py              # Data collection and preprocessing
├── predict_scores_enhanced.py  # Model training and prediction
├── visualise.py                # Dashboard generation
├── download_logos.py           # Team logo downloader
├── run_all.py                  # Complete pipeline runner
├── data/
│   ├── matches_master.csv      # Historical match data
│   └── upcoming_fixtures.csv   # Upcoming fixtures
└── output/
    ├── predictions_upcoming.csv
    ├── predictions_historical.csv
    └── predictions_dashboard.html
```

---

## Methodology

### Data Collection
- Source: FBref via soccerdata library
- Historical data: 2018-present
- Features: xG metrics, team form, home advantage

### Model Architecture
- Algorithm: XGBoost Regressor
- Target variables: Home goals, Away goals
- Training data: All historical Premier League matches
- Validation: In-sample predictions for current season

### Features
- `home_xg`: Home team expected goals
- `away_xg`: Away team expected goals
- `xg_diff`: Expected goal differential
- `home_adv`: Home advantage factor

---

## Technical Details

**Requirements:**
- Python 3.8+
- pandas, numpy, scikit-learn, xgboost
- soccerdata (FBref scraper)
- requests (for logo downloads)

**Model Parameters:**
- n_estimators: 300
- max_depth: 4
- learning_rate: 0.05
- subsample: 0.9
- colsample_bytree: 0.9

---

*Last updated: {now}*  
*Data provided by FBref | For informational purposes only*
"""


def generate_full_readme(upcoming_df: pd.DataFrame, historical_df: pd.DataFrame) -> str:
    """Generate complete README with predictions and statistics."""
    season_year, season_display = get_current_season()
    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    # Calculate performance metrics
    metrics = calculate_performance_metrics(historical_df) if not historical_df.empty else None

    # Get next gameweek
    gameweeks = group_by_gameweek(upcoming_df) if not upcoming_df.empty else {}
    next_gw = min(gameweeks.keys()) if gameweeks else None

    # Build README
    content = f"""# Premier League Match Predictor

Machine learning system for predicting Premier League match outcomes using XGBoost and expected goals (xG) data.

**Season:** {season_display}
"""

    # Performance metrics
    if metrics:
        content += f"""
---

## Model Performance

Current season ({season_display}) statistics:

| Metric | Value |
|--------|-------|
| Matches Analyzed | {metrics['total_matches']} |
| Average Error (MAE) | ±{metrics['avg_mae']:.3f} goals |
| Home Goals MAE | ±{metrics['mae_home']:.3f} |
| Away Goals MAE | ±{metrics['mae_away']:.3f} |
| Accuracy (±1 goal) | {metrics['accuracy_pct']:.1f}% |
| Perfect Predictions | {metrics['perfect_predictions']} ({metrics['perfect_pct']:.1f}%) |
"""

    # Next gameweek predictions
    if next_gw and next_gw in gameweeks:
        matches = gameweeks[next_gw]
        content += f"""
---

## Next Gameweek Predictions

**Gameweek {next_gw}** ({len(matches)} matches)

| Home | Prediction | Away | Win Probability |
|------|------------|------|----------------|
"""

        for match in matches[:10]:  # Show up to 10 matches
            home = match["home_team"]
            away = match["away_team"]
            pred_home = match["pred_home_goals"]
            pred_away = match["pred_away_goals"]
            prob_home = match["prob_home"] * 100
            prob_draw = match["prob_draw"] * 100
            prob_away = match["prob_away"] * 100

            # Determine favorite
            if prob_home > prob_draw and prob_home > prob_away:
                favorite = f"**{home}** {prob_home:.0f}%"
            elif prob_away > prob_draw and prob_away > prob_home:
                favorite = f"**{away}** {prob_away:.0f}%"
            else:
                favorite = f"Draw {prob_draw:.0f}%"

            content += f"| {home} | {pred_home:.1f} - {pred_away:.1f} | {away} | {favorite} |\n"

        if len(matches) > 10:
            content += f"\n*... and {len(matches) - 10} more fixtures*\n"

    # All upcoming gameweeks summary
    if len(gameweeks) > 1:
        content += f"""
---

## Upcoming Fixtures Summary

Total fixtures predicted: **{len(upcoming_df)}**

| Gameweek | Matches | Date Range |
|----------|---------|------------|
"""
        for gw_num in sorted(gameweeks.keys()):
            matches = gameweeks[gw_num]
            first = matches[0]["datetime"].strftime("%d %b")
            last = matches[-1]["datetime"].strftime("%d %b")
            content += f"| {gw_num} | {len(matches)} | {first} - {last} |\n"

    # Project documentation
    content += """
---

## Quick Start

Run the complete prediction pipeline:

```bash
python run_all.py
```

This will:
1. Download latest match data from FBref
2. Train the prediction model
3. Generate predictions for upcoming fixtures
4. Create interactive HTML dashboard

View the dashboard:

```bash
open output/predictions_dashboard.html
```

Download team logos (optional):

```bash
python download_logos.py
```

---

## Project Structure

```
├── update_data.py              # Data collection and preprocessing
├── predict_scores_enhanced.py  # Model training and prediction
├── visualise.py                # Dashboard generation
├── download_logos.py           # Team logo downloader
├── run_all.py                  # Complete pipeline runner
├── generate_readme.py          # README generator
├── data/
│   ├── matches_master.csv      # Historical match data (2018-present)
│   └── upcoming_fixtures.csv   # Upcoming fixtures with estimated xG
└── output/
    ├── predictions_upcoming.csv     # Upcoming match predictions
    ├── predictions_historical.csv   # Current season predictions
    └── predictions_dashboard.html   # Interactive visualization
```

---

## Methodology

### Data Collection

**Source:** FBref via [soccerdata](https://github.com/probberechts/soccerdata) library

**Historical Data:**
- Seasons: 2018-present
- Metrics: Goals, xG (expected goals), match outcomes
- Automatic updates: Runs on each prediction cycle

**Upcoming Fixtures:**
- Extracted from FBref schedule
- xG estimated from recent team form (last 5 matches)
- Home advantage factor included (+0.3 xG)

### Model Architecture

**Algorithm:** XGBoost Regressor (Gradient Boosting)

**Target Variables:**
- Home team goals
- Away team goals

**Training Strategy:**
- Train on all historical data
- Generate in-sample predictions for current season
- Apply to upcoming fixtures

### Feature Engineering

| Feature | Description |
|---------|-------------|
| `home_xg` | Home team's expected goals (estimated or actual) |
| `away_xg` | Away team's expected goals (estimated or actual) |
| `xg_diff` | Difference in expected goals (home - away) |
| `home_adv` | Home advantage constant (1.0 for all matches) |

### Win Probability Calculation

Probabilities derived from predicted score differential:
- Large differential → High win probability
- Small differential → Higher draw probability
- Margin of 0.5 goals used as threshold

---

## Model Configuration

```python
XGBRegressor(
    n_estimators=300,       # Number of boosting rounds
    max_depth=4,            # Maximum tree depth
    learning_rate=0.05,     # Step size shrinkage
    subsample=0.9,          # Training data sampling
    colsample_bytree=0.9,   # Feature sampling
    objective='reg:squarederror',
    random_state=42
)
```

---

## Requirements

**Python 3.8+**

**Core Dependencies:**
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
soccerdata>=1.5.0
requests>=2.28.0
```

**Install:**
```bash
pip install pandas numpy scikit-learn xgboost soccerdata requests
```

---

## Output Files

### predictions_upcoming.csv
Predictions for upcoming fixtures

| Column | Description |
|--------|-------------|
| datetime | Match date and time |
| home_team | Home team name |
| away_team | Away team name |
| pred_home_goals | Predicted home goals |
| pred_away_goals | Predicted away goals |
| prob_home | Home win probability |
| prob_draw | Draw probability |
| prob_away | Away win probability |

### predictions_historical.csv
Current season predictions vs actual results

Additional columns:
- `home_goals`: Actual home goals
- `away_goals`: Actual away goals
- `home_xg`: Actual home xG
- `away_xg`: Actual away xG

---

## Dashboard Features

The interactive HTML dashboard includes:

**Past Performance**
- Current season matches with actual vs predicted scores
- Color-coded accuracy indicators
- Actual xG comparison

**Next Gameweek**
- Highlighted upcoming matches
- Win probabilities
- Predicted scores

**Future Predictions**
- All upcoming gameweeks
- Organized chronologically
- Team logos and match details

---

## Data Sources

**Match Data:** [FBref](https://fbref.com/)  
**Team Logos:** [Wikipedia](https://en.wikipedia.org/) / [Wikimedia Commons](https://commons.wikimedia.org/)  
**xG Calculations:** Based on FBref's expected goals model

---

## Limitations and Disclaimers

- Predictions are for informational and entertainment purposes only
- Model accuracy varies based on data quality and team form
- xG estimates for upcoming matches based on recent performance
- Does not account for: injuries, suspensions, motivation, weather
- Past performance does not guarantee future results

---

## License

MIT License - See LICENSE file for details

---

*Last updated: {now}*  
*Generated automatically by the prediction pipeline*  
*For questions or issues, please open a GitHub issue*
"""

    return content


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    print("=" * 80)
    print("README GENERATOR")
    print("=" * 80)

    # Check for prediction files
    has_upcoming = PREDICTIONS_UPCOMING.exists()
    has_historical = PREDICTIONS_HISTORICAL.exists()

    if not has_upcoming and not has_historical:
        print("\n⚠️  No prediction files found")
        print("Generating minimal README...")
        content = generate_empty_readme()
    else:
        print(f"\n✓ Found prediction files")

        # Load data
        upcoming_df = pd.read_csv(PREDICTIONS_UPCOMING, parse_dates=["datetime"]) if has_upcoming else pd.DataFrame()
        historical_df = pd.read_csv(PREDICTIONS_HISTORICAL,
                                    parse_dates=["datetime"]) if has_historical else pd.DataFrame()

        print(f"  - Upcoming predictions: {len(upcoming_df)}")
        print(f"  - Historical predictions: {len(historical_df)}")

        # Generate README
        content = generate_full_readme(upcoming_df, historical_df)

    # Write README
    README_FILE.write_text(content, encoding="utf-8")

    season_year, season_display = get_current_season()
    print(f"\n✅ Generated README.md")
    print(f"   Season: {season_display}")
    print(f"   Path: {README_FILE.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())