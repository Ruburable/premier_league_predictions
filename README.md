# Premier League Match Predictor

Machine learning system for predicting Premier League match outcomes using XGBoost and expected goals (xG) data.

**Season:** 2025/26

---

## Model Performance

Current season (2025/26) statistics:

| Metric | Value |
|--------|-------|
| Matches Analyzed | 210 |
| Average Error (MAE) | ±0.726 goals |
| Home Goals MAE | ±0.750 |
| Away Goals MAE | ±0.701 |
| Accuracy (±1 goal) | 55.7% |
| Perfect Predictions | 43 (20.5%) |

---

## Next Gameweek Predictions

**Gameweek 1** (10 matches)

| Home | Prediction | Away | Win Probability |
|------|------------|------|----------------|
| Chelsea | 2.0 - 3.3 | Brentford | **Brentford** 76% |
| Leeds United | 2.0 - 1.4 | Fulham | **Leeds United** 61% |
| Liverpool | 1.6 - 0.8 | Burnley | **Liverpool** 66% |
| Manchester Utd | 1.8 - 1.3 | Manchester City | Draw 35% |
| Nott'ham Forest | 1.3 - 1.6 | Arsenal | Draw 35% |
| Sunderland | 1.5 - 1.5 | Crystal Palace | Draw 35% |
| Tottenham | 1.6 - 1.0 | West Ham | **Tottenham** 62% |
| Aston Villa | 2.2 - 1.3 | Everton | **Aston Villa** 69% |
| Wolves | 1.4 - 1.7 | Newcastle Utd | Draw 35% |
| Brighton | 1.9 - 1.6 | Bournemouth | Draw 35% |

---

## Upcoming Fixtures Summary

Total fixtures predicted: **170**

| Gameweek | Matches | Date Range |
|----------|---------|------------|
| 1 | 10 | 17 Jan - 19 Jan |
| 2 | 10 | 24 Jan - 26 Jan |
| 3 | 10 | 31 Jan - 02 Feb |
| 4 | 14 | 06 Feb - 10 Feb |
| 5 | 6 | 11 Feb - 12 Feb |
| 6 | 10 | 21 Feb - 23 Feb |
| 7 | 10 | 27 Feb - 01 Mar |
| 8 | 10 | 04 Mar - 04 Mar |
| 9 | 10 | 14 Mar - 14 Mar |
| 10 | 10 | 21 Mar - 21 Mar |
| 11 | 10 | 11 Apr - 11 Apr |
| 12 | 10 | 18 Apr - 18 Apr |
| 13 | 10 | 25 Apr - 25 Apr |
| 14 | 10 | 02 May - 02 May |
| 15 | 10 | 09 May - 09 May |
| 16 | 10 | 17 May - 17 May |
| 17 | 10 | 24 May - 24 May |

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

*Last updated: {now}*  
*Generated automatically by the prediction pipeline*
