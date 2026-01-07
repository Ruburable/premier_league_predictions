# ⚽ Premier League Match Predictor

AI-powered predictions for Premier League matches using XGBoost and historical xG data.

## 🔮 Next Gameweek Predictions

### Gameweek 1


**West Ham** vs **Nott'ham Forest**  
📅 Tue 06 Jan, 00:00 | 🎯 Prediction: **1.2 - 1.5**  
📊 Win Probabilities: Home 33% | Draw 35% | Away 32%


**Bournemouth** vs **Tottenham**  
📅 Wed 07 Jan, 00:00 | 🎯 Prediction: **2.1 - 1.5**  
📊 Win Probabilities: Home 62% | Draw 29% | Away 10%


**Brentford** vs **Sunderland**  
📅 Wed 07 Jan, 00:00 | 🎯 Prediction: **2.2 - 1.2**  
📊 Win Probabilities: Home 69% | Draw 23% | Away 8%


**Burnley** vs **Manchester Utd**  
📅 Wed 07 Jan, 00:00 | 🎯 Prediction: **1.2 - 1.6**  
📊 Win Probabilities: Home 33% | Draw 35% | Away 32%


**Crystal Palace** vs **Aston Villa**  
📅 Wed 07 Jan, 00:00 | 🎯 Prediction: **2.0 - 2.0**  
📊 Win Probabilities: Home 33% | Draw 35% | Away 32%


**Everton** vs **Wolves**  
📅 Wed 07 Jan, 00:00 | 🎯 Prediction: **1.5 - 1.3**  
📊 Win Probabilities: Home 33% | Draw 35% | Away 32%


**Fulham** vs **Chelsea**  
📅 Wed 07 Jan, 00:00 | 🎯 Prediction: **1.8 - 1.8**  
📊 Win Probabilities: Home 33% | Draw 35% | Away 32%


**Manchester City** vs **Brighton**  
📅 Wed 07 Jan, 00:00 | 🎯 Prediction: **1.9 - 1.4**  
📊 Win Probabilities: Home 61% | Draw 29% | Away 10%


**Newcastle Utd** vs **Leeds United**  
📅 Wed 07 Jan, 00:00 | 🎯 Prediction: **1.8 - 1.4**  
📊 Win Probabilities: Home 33% | Draw 35% | Away 32%


**Arsenal** vs **Liverpool**  
📅 Thu 08 Jan, 00:00 | 🎯 Prediction: **2.1 - 1.5**  
📊 Win Probabilities: Home 62% | Draw 29% | Away 10%


### Quick Picks

- **Draw** predicted
- **Bournemouth** to win
- **Brentford** to win
- **Draw** predicted
- **Draw** predicted


## 📅 All Upcoming Gameweeks

Total fixtures predicted: **180**

- **Gameweek 1**: 10 matches (06 Jan - 08 Jan)
- **Gameweek 2**: 10 matches (17 Jan - 19 Jan)
- **Gameweek 3**: 10 matches (24 Jan - 26 Jan)
- **Gameweek 4**: 10 matches (31 Jan - 02 Feb)
- **Gameweek 5**: 14 matches (06 Feb - 10 Feb)
- **Gameweek 6**: 6 matches (11 Feb - 12 Feb)
- **Gameweek 7**: 10 matches (21 Feb - 23 Feb)
- **Gameweek 8**: 10 matches (27 Feb - 01 Mar)
- **Gameweek 9**: 10 matches (04 Mar - 04 Mar)
- **Gameweek 10**: 10 matches (14 Mar - 14 Mar)
- **Gameweek 11**: 10 matches (21 Mar - 21 Mar)
- **Gameweek 12**: 10 matches (11 Apr - 11 Apr)
- **Gameweek 13**: 10 matches (18 Apr - 18 Apr)
- **Gameweek 14**: 10 matches (25 Apr - 25 Apr)
- **Gameweek 15**: 10 matches (02 May - 02 May)
- **Gameweek 16**: 10 matches (09 May - 09 May)
- **Gameweek 17**: 10 matches (17 May - 17 May)
- **Gameweek 18**: 10 matches (24 May - 24 May)


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
3. **Model Training**: XGBoost regressor trained on 2480 historical matches
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

*Last updated: 2026-01-07 22:36*  
*Predictions generated using machine learning on historical xG data*  
*For entertainment purposes - always gamble responsibly* 🎲
