#!/usr/bin/env python3
"""
visualise_predictions.py

Reads:
  - output/predictions_upcoming.csv

Creates:
  - output/predictions_dashboard.html

Generates a light-themed HTML dashboard showing:
  - Predicted goals
  - Win/draw probabilities
  - Top predicted scorers
  - Club logos
"""

import pandas as pd
from pathlib import Path
import base64
import numpy as np

PRED_CSV = Path("output/predictions_upcoming.csv")
LOGO_DIR = Path("output/logos")
OUT_HTML = Path("output/predictions_dashboard.html")


def svg_to_base64(path: Path):
    if not path.exists():
        return None
    try:
        return base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception:
        return None


def scorers_html(s):
    """Convert list of scorers to HTML."""
    if not s or pd.isna(s):
        return "<span class='muted'>–</span>"
    # Expecting CSV column to be JSON list: [["Player", prob], ...]
    import json
    try:
        lst = json.loads(s)
    except Exception:
        return "<span class='muted'>–</span>"
    if not lst:
        return "<span class='muted'>–</span>"
    return "<br>".join([f"• {p[0]} ({p[1]*100:.1f}%)" for p in lst])


def build_match_html(row, logos_b64):
    home = row["home_team"]
    away = row["away_team"]

    # Use correct column names from predictions CSV
    pred_home = row.get("pred_home_goals", 0)
    pred_away = row.get("pred_away_goals", 0)
    prob_home = row.get("prob_home", row.get("p_home_win_mc", 0))
    prob_draw = row.get("prob_draw", row.get("p_draw_mc", 0))
    prob_away = row.get("prob_away", row.get("p_away_win_mc", 0))

    return f"""
    <div class="match-row">

      <div class="team team-left">
        <img class="badge" src="data:image/svg+xml;base64,{logos_b64.get(home,'')}" alt="{home}">
        <div class="team-name">{home}</div>
        <div class="scorers">{scorers_html(row.get("top_home_scorers",""))}</div>
      </div>

      <div class="center">
        <div class="scoreline">{pred_home:.2f} – {pred_away:.2f}</div>
        <div class="prob">
          <span>Home: {prob_home*100:.1f}%</span> |
          <span>Draw: {prob_draw*100:.1f}%</span> |
          <span>Away: {prob_away*100:.1f}%</span>
        </div>
      </div>

      <div class="team team-right">
        <img class="badge" src="data:image/svg+xml;base64,{logos_b64.get(away,'')}" alt="{away}">
        <div class="team-name">{away}</div>
        <div class="scorers">{scorers_html(row.get("top_away_scorers",""))}</div>
      </div>

    </div>
    """


def build_page(rows_html):
    css = """
    body { font-family: Arial, sans-serif; background: #fafbfc; margin: 0; padding: 30px; color: #111; }
    .container { max-width: 900px; margin: auto; }
    h1 { text-align: center; margin-bottom: 24px; font-size: 28px; }
    .match-row {
        display: flex; align-items: center; justify-content: space-between;
        background: white; border-radius: 10px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        padding: 18px; margin-bottom: 20px;
        transition: transform 0.2s ease;
    }
    .match-row:hover { transform: translateY(-3px); }
    .team { width: 32%; text-align: center; }
    .badge { width: 64px; height: 64px; margin-bottom: 6px; }
    .team-name { font-weight: bold; margin-top: 4px; font-size: 18px; }
    .scorers { margin-top: 6px; font-size: 14px; color: #444; }
    .center { width: 25%; text-align: center; }
    .scoreline { font-size: 28px; font-weight: bold; margin-bottom: 6px; }
    .prob { font-size: 13px; color: #555; }
    """

    return f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Premier League Predictions</title><style>{css}</style></head>
<body>
  <div class="container">
    <h1>Upcoming Fixtures & Predictions</h1>
    {rows_html}
  </div>
</body>
</html>
"""


def main():
    df = pd.read_csv(PRED_CSV)
    clubs = set(df["home_team"].tolist() + df["away_team"].tolist())

    logos_b64 = {}
    for club in clubs:
        svg_path = LOGO_DIR / f"{club}.svg"
        b64 = svg_to_base64(svg_path)
        if b64:
            logos_b64[club] = b64

    rows_html = "\n".join(build_match_html(r, logos_b64) for _, r in df.iterrows())

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(build_page(rows_html), encoding="utf-8")
    print("✔ Saved dashboard to:", OUT_HTML)


if __name__ == "__main__":
    main()
