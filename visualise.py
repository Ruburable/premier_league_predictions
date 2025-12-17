#!/usr/bin/env python3
"""
visualise_predictions.py

Reads:
  - output/predictions_upcoming.csv

Creates:
  - output/predictions_dashboard.html
"""

import pandas as pd
from pathlib import Path
import base64
import json

PRED_CSV = Path("output/predictions_upcoming.csv")
LOGO_DIR = Path("output/logos")
OUT_HTML = Path("output/predictions_dashboard.html")


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def svg_to_base64(path: Path):
    if not path.exists():
        return ""
    try:
        return base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception:
        return ""


def scorers_html(s):
    if not s or pd.isna(s):
        return "<span class='muted'>–</span>"
    try:
        lst = json.loads(s)
        return "<br>".join([f"• {p[0]} ({p[1]*100:.1f}%)" for p in lst])
    except Exception:
        return "<span class='muted'>–</span>"


# --------------------------------------------------
# Match card HTML
# --------------------------------------------------
def build_match_html(row, logos):
    home = row["home_team"]
    away = row["away_team"]

    ph = row.get("pred_home_goals", 0.0)
    pa = row.get("pred_away_goals", 0.0)

    prob_home = row.get("prob_home", None)
    prob_draw = row.get("prob_draw", None)
    prob_away = row.get("prob_away", None)

    prob_html = (
        f"Home: {prob_home*100:.1f}% | Draw: {prob_draw*100:.1f}% | Away: {prob_away*100:.1f}%"
        if all(v is not None for v in [prob_home, prob_draw, prob_away])
        else "Probabilities unavailable"
    )

    return f"""
    <div class="match-row">

      <div class="team team-left">
        <img class="badge" src="data:image/svg+xml;base64,{logos.get(home,'')}" />
        <div class="team-name">{home}</div>
        <div class="scorers">{scorers_html(row.get("top_home_scorers"))}</div>
      </div>

      <div class="center">
        <div class="scoreline">{ph:.2f} – {pa:.2f}</div>
        <div class="prob">{prob_html}</div>
      </div>

      <div class="team team-right">
        <img class="badge" src="data:image/svg+xml;base64,{logos.get(away,'')}" />
        <div class="team-name">{away}</div>
        <div class="scorers">{scorers_html(row.get("top_away_scorers"))}</div>
      </div>

    </div>
    """


# --------------------------------------------------
# Page shell
# --------------------------------------------------
def build_page(rows):
    css = """
    body { font-family: Arial, sans-serif; background:#fafbfc; padding:30px; }
    .container { max-width:900px; margin:auto; }
    h1 { text-align:center; margin-bottom:30px; }
    .match-row {
        display:flex; justify-content:space-between; align-items:center;
        background:white; padding:18px; border-radius:10px;
        box-shadow:0 3px 10px rgba(0,0,0,0.1);
        margin-bottom:20px;
    }
    .team { width:32%; text-align:center; }
    .badge { width:64px; height:64px; }
    .team-name { font-size:18px; font-weight:bold; }
    .scorers { font-size:14px; margin-top:6px; color:#444; }
    .center { width:25%; text-align:center; }
    .scoreline { font-size:28px; font-weight:bold; }
    .prob { font-size:13px; color:#555; margin-top:6px; }
    .muted { color:#aaa; }
    """

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Premier League Predictions</title>
<style>{css}</style>
</head>
<body>
<div class="container">
<h1>Upcoming Fixtures & Predictions</h1>
{rows}
</div>
</body>
</html>
"""


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    df = pd.read_csv(PRED_CSV)

    clubs = set(df["home_team"]) | set(df["away_team"])
    logos = {c: svg_to_base64(LOGO_DIR / f"{c}.svg") for c in clubs}

    rows_html = "\n".join(
        build_match_html(row, logos) for _, row in df.iterrows()
    )

    OUT_HTML.parent.mkdir(exist_ok=True, parents=True)
    OUT_HTML.write_text(build_page(rows_html), encoding="utf-8")

    print("✔ Dashboard created:", OUT_HTML.resolve())


if __name__ == "__main__":
    main()
