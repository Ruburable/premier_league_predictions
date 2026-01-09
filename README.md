# Premier League Match Predictor

Machine learning system for predicting Premier League match outcomes using XGBoost and expected goals (xG) data.

**Season:** 2025/26

---

## Next Gameweek Predictions

**Gameweek 1** (9 matches)

| Home | Prediction | Away | Win Probability |
|------|------------|------|----------------|
| Bournemouth | 2.1 - 1.4 | Tottenham | **Bournemouth** 62% |
| Brentford | 2.2 - 1.2 | Sunderland | **Brentford** 70% |
| Burnley | 1.1 - 1.6 | Manchester Utd | Draw 35% |
| Crystal Palace | 1.9 - 2.1 | Aston Villa | Draw 35% |
| Everton | 1.5 - 1.4 | Wolves | Draw 35% |
| Fulham | 1.8 - 1.8 | Chelsea | Draw 35% |
| Manchester City | 2.0 - 1.4 | Brighton | **Manchester City** 62% |
| Newcastle Utd | 1.8 - 1.4 | Leeds United | Draw 35% |
| Arsenal | 2.1 - 1.4 | Liverpool | **Arsenal** 62% |

---

## Upcoming Fixtures Summary

Total fixtures predicted: **179**

| Gameweek | Matches | Date Range |
|----------|---------|------------|
| 1 | 9 | 07 Jan - 08 Jan |
| 2 | 10 | 17 Jan - 19 Jan |
| 3 | 10 | 24 Jan - 26 Jan |
| 4 | 10 | 31 Jan - 02 Feb |
| 5 | 14 | 06 Feb - 10 Feb |
| 6 | 6 | 11 Feb - 12 Feb |
| 7 | 10 | 21 Feb - 23 Feb |
| 8 | 10 | 27 Feb - 01 Mar |
| 9 | 10 | 04 Mar - 04 Mar |
| 10 | 10 | 14 Mar - 14 Mar |
| 11 | 10 | 21 Mar - 21 Mar |
| 12 | 10 | 11 Apr - 11 Apr |
| 13 | 10 | 18 Apr - 18 Apr |
| 14 | 10 | 25 Apr - 25 Apr |
| 15 | 10 | 02 May - 02 May |
| 16 | 10 | 09 May - 09 May |
| 17 | 10 | 17 May - 17 May |
| 18 | 10 | 24 May - 24 May |

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
