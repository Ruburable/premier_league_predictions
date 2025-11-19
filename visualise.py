#!/usr/bin/env python3
"""
visualise_predictions.py

Reads:
  - data/predictions_gameweek.csv

Creates:
  - data/gameweek_predictions.html

Generates a clean HTML dashboard showing:
  - Expected scores (pred_xg)
  - Most likely scoreline
  - Win/draw probabilities
  - Top predicted scorers with probabilities
"""

import pandas as pd
import json
from pathlib import Path

PRED_FILE = Path("data/predictions_gameweek.csv")
OUT_FILE = Path("data/gameweek_predictions.html")

CSS_STYLE = """
<style>
body {
    font-family: Arial, sans-serif;
    background: #f5f6fa;
    margin: 0;
    padding: 20px;
}
h1 {
    text-align: center;
    margin-bottom: 30px;
    color: #333;
}
.match-card {
    background: white;
    padding: 20px;
    margin: 20px auto;
    width: 90%;
    max-width: 900px;
    border-radius: 12px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
}
.teams {
    display: flex;
    justify-content: space-between;
    font-size: 22px;
    font-weight: bold;
}
.xg {
    margin-top: 10px;
    font-size: 18px;
    color: #555;
}
.scoreline {
    margin-top: 10px;
    font-size: 18px;
}
.probs {
    margin-top: 10px;
    font-size: 15px;
    color: #333;
}
.scorers {
    margin-top: 15px;
}
.scorers table {
    width: 100%;
    border-collapse: collapse;
}
.scorers th, .scorers td {
    text-align: left;
    padding: 6px 4px;
}
.section-title {
    margin-top: 15px;
    font-weight: bold;
    font-size: 16px;
    color: #333;
}
</style>
"""

def format_scorers(json_list):
    """
    Input format:  [ ["Player", prob], ["Player2", prob2], ... ]
    Returns HTML table rows.
    """
    try:
        items = json.loads(json_list)
    except:
        return "<i>No scorer data</i>"

    if not items:
        return "<i>No scorer data</i>"

    rows = ""
    for player, prob in items:
        rows += f"<tr><td>{player}</td><td>{prob*100:.1f}%</td></tr>"
    return rows

def generate_html(df):
    cards = []

    for _, r in df.iterrows():
        home = r["home_team"]
        away = r["away_team"]

        # Decode scorer lists
        home_scorers = format_scorers(r["top_home_scorers"])
        away_scorers = format_scorers(r["top_away_scorers"])

        card = f"""
        <div class="match-card">
            <div class="teams">{home} <span>vs</span> {away}</div>

            <div class="xg">
                Expected goals (xG): <b>{r['pred_home_xg']:.2f} - {r['pred_away_xg']:.2f}</b>
            </div>

            <div class="scoreline">
                Most likely scoreline: <b>{int(r['most_likely_score_home'])} - {int(r['most_likely_score_away'])}</b>
            </div>

            <div class="probs">
                Win probabilities (Monte Carlo):<br>
                ▶ Home win: <b>{r['p_home_win_mc']*100:.1f}%</b><br>
                ▶ Draw: <b>{r['p_draw_mc']*100:.1f}%</b><br>
                ▶ Away win: <b>{r['p_away_win_mc']*100:.1f}%</b>
            </div>

            <div class="section-title">Top predicted Home scorers</div>
            <div class="scorers">
                <table>
                    {home_scorers}
                </table>
            </div>

            <div class="section-title">Top predicted Away scorers</div>
            <div class="scorers">
                <table>
                    {away_scorers}
                </table>
            </div>
        </div>
        """
        cards.append(card)

    html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Premier League Gameweek Predictions</title>
        {CSS_STYLE}
    </head>
    <body>
        <h1>Premier League Gameweek Predictions</h1>
        {''.join(cards)}
    </body>
    </html>
    """

    return html


def main():
    if not PRED_FILE.exists():
        raise FileNotFoundError("You must run the prediction script first. 'data/predictions_gameweek.csv' not found.")

    df = pd.read_csv(PRED_FILE)
    html = generate_html(df)

    OUT_FILE.write_text(html, encoding="utf-8")
    print(f"HTML page created: {OUT_FILE}")


if __name__ == "__main__":
    main()
