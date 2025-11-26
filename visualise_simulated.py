#!/usr/bin/env python3
"""
build_dashboard.py
Creates a light-themed HTML dashboard using *local* club logos that must be
stored under: output/logos/<ClubName>.svg

Requires a CSV file with:
home_team, away_team, pred_home_goals, pred_away_goals, prob_home, prob_draw, prob_away, home_scorers, away_scorers
"""

import pandas as pd
from pathlib import Path
import base64

PRED_CSV = Path("sim_predictions.csv")
LOGO_DIR = Path("output/logos")
OUT_HTML = Path("output/predictions_dashboard.html")


def svg_to_base64(path: Path):
    if not path.exists():
        return None
    try:
        b = path.read_bytes()
        return base64.b64encode(b).decode("utf-8")
    except:
        return None


def scorers_html(s):
    if not s or pd.isna(s):
        return "<span class='muted'>–</span>"
    return "<br>".join([f"• {p.strip()}" for p in s.split(",") if p.strip()])


def build_match_html(row, logos_b64):
    home = row["home_team"]
    away = row["away_team"]

    return f"""
    <div class="match-row">

      <div class="team team-left">
        <img class="badge" src="data:image/svg+xml;base64,{logos_b64.get(home,'')}" alt="{home}">
        <div class="team-name">{home}</div>
        <div class="scorers">{scorers_html(row.get("home_scorers",""))}</div>
      </div>

      <div class="center">
        <div class="scoreline">{int(row['pred_home_goals'])} – {int(row['pred_away_goals'])}</div>
        <div class="prob">
          <span>Home: {row['prob_home']:.2f}</span> |
          <span>Draw: {row['prob_draw']:.2f}</span> |
          <span>Away: {row['prob_away']:.2f}</span>
        </div>
      </div>

      <div class="team team-right">
        <img class="badge" src="data:image/svg+xml;base64,{logos_b64.get(away,'')}" alt="{away}">
        <div class="team-name">{away}</div>
        <div class="scorers">{scorers_html(row.get("away_scorers",""))}</div>
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
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        padding: 16px; margin-bottom: 16px;
    }
    .team { width: 32%; text-align: center; }
    .badge { width: 64px; height: 64px; }
    .team-name { font-weight: bold; margin-top: 8px; font-size: 18px; }
    .scorers { margin-top: 6px; font-size: 14px; color: #444; }
    .center { width: 25%; text-align: center; }
    .scoreline { font-size: 28px; font-weight: bold; margin-bottom: 6px; }
    .prob { font-size: 13px; color: #555; }
    """

    return f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Predictions</title><style>{css}</style></head>
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
    print("Saved dashboard to:", OUT_HTML)


if __name__ == "__main__":
    main()
