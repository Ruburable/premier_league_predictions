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
    except Exception as e:
        print(f"Warning: Could not encode {path}: {e}")
        return ""


def scorers_html(s):
    if not s or pd.isna(s):
        return "<span class='muted'>–</span>"
    try:
        lst = json.loads(s)
        return "<br>".join([f"• {p[0]} ({p[1] * 100:.1f}%)" for p in lst])
    except Exception:
        return "<span class='muted'>–</span>"


# --------------------------------------------------
# Match card HTML
# --------------------------------------------------
def build_match_html(row, logos):
    home = row["home_team"]
    away = row["away_team"]

    ph = float(row.get("pred_home_goals", 0.0))
    pa = float(row.get("pred_away_goals", 0.0))

    prob_home = row.get("prob_home", None)
    prob_draw = row.get("prob_draw", None)
    prob_away = row.get("prob_away", None)

    # Format date if available
    date_str = ""
    if "datetime" in row and pd.notna(row["datetime"]):
        try:
            date_obj = pd.to_datetime(row["datetime"])
            date_str = f"<div class='match-date'>{date_obj.strftime('%a, %d %b %Y')}</div>"
        except:
            pass

    prob_html = (
        f"Home: {float(prob_home) * 100:.1f}% | Draw: {float(prob_draw) * 100:.1f}% | Away: {float(prob_away) * 100:.1f}%"
        if all(v is not None and pd.notna(v) for v in [prob_home, prob_draw, prob_away])
        else "Probabilities unavailable"
    )

    return f"""
    <div class="match-row">

      <div class="team team-left">
        <img class="badge" src="data:image/svg+xml;base64,{logos.get(home, '')}" onerror="this.style.display='none'" />
        <div class="team-name">{home}</div>
        <div class="scorers">{scorers_html(row.get("top_home_scorers"))}</div>
      </div>

      <div class="center">
        {date_str}
        <div class="scoreline">{ph:.2f} – {pa:.2f}</div>
        <div class="prob">{prob_html}</div>
      </div>

      <div class="team team-right">
        <img class="badge" src="data:image/svg+xml;base64,{logos.get(away, '')}" onerror="this.style.display='none'" />
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
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background:#f5f7fa; padding:30px; }
    .container { max-width:1000px; margin:auto; }
    h1 { text-align:center; margin-bottom:10px; color:#2c3e50; font-size:36px; }
    .subtitle { text-align:center; color:#7f8c8d; margin-bottom:40px; font-size:16px; }

    .gameweek-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px 30px;
        border-radius: 12px;
        margin: 30px 0 20px 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .gameweek-header h2 {
        margin: 0 0 8px 0;
        font-size: 28px;
        font-weight: 600;
    }
    .gameweek-date {
        font-size: 15px;
        opacity: 0.95;
        font-weight: 500;
    }
    .gameweek-count {
        font-size: 13px;
        opacity: 0.85;
        margin-top: 5px;
    }

    .match-row {
        display:flex; justify-content:space-between; align-items:center;
        background:white; padding:20px; border-radius:12px;
        box-shadow:0 2px 8px rgba(0,0,0,0.08);
        margin-bottom:16px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .match-row:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    }
    .team { width:32%; text-align:center; }
    .badge { width:64px; height:64px; object-fit: contain; }
    .team-name { font-size:17px; font-weight:600; margin-top:8px; color:#2c3e50; }
    .scorers { font-size:13px; margin-top:6px; color:#7f8c8d; }
    .center { width:25%; text-align:center; }
    .match-date { font-size:12px; color:#95a5a6; margin-bottom:8px; font-weight:500; }
    .scoreline { font-size:32px; font-weight:700; color:#2c3e50; letter-spacing: 2px; }
    .prob { font-size:12px; color:#7f8c8d; margin-top:8px; background:#f8f9fa; padding:6px 10px; border-radius:6px; }
    .muted { color:#bdc3c7; }
    .error { color: #e74c3c; padding: 40px; text-align: center; background:white; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,0.08); }
    """

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Premier League Predictions</title>
<style>{css}</style>
</head>
<body>
<div class="container">
<h1>⚽ Premier League Predictions</h1>
<div class="subtitle">AI-powered match predictions using XGBoost and xG data</div>
{rows}
</div>
</body>
</html>
"""


# --------------------------------------------------
# Main
# --------------------------------------------------
def group_by_gameweek(df):
    """Group matches by gameweek based on dates."""
    if df.empty:
        return {}

    df = df.sort_values("datetime").reset_index(drop=True)

    gameweeks = {}
    current_gw = 1
    current_gw_start = df.iloc[0]["datetime"]

    for idx, row in df.iterrows():
        match_date = row["datetime"]

        # If more than 4 days from current gameweek start, start new gameweek
        days_diff = (match_date - current_gw_start).total_seconds() / 86400
        if days_diff > 4:
            current_gw += 1
            current_gw_start = match_date

        if current_gw not in gameweeks:
            gameweeks[current_gw] = []

        gameweeks[current_gw].append(row)

    return gameweeks


def build_gameweek_section(gw_num, matches, logos):
    """Build HTML for a gameweek section."""
    # Get date range for this gameweek
    first_date = matches[0]["datetime"]
    last_date = matches[-1]["datetime"]

    if first_date.date() == last_date.date():
        date_range = first_date.strftime("%A, %d %B %Y")
    else:
        date_range = f"{first_date.strftime('%d %b')} - {last_date.strftime('%d %b %Y')}"

    header = f"""
    <div class="gameweek-header">
        <h2>Gameweek {gw_num}</h2>
        <div class="gameweek-date">{date_range}</div>
        <div class="gameweek-count">{len(matches)} matches</div>
    </div>
    """

    matches_html = "\n".join(
        build_match_html(match, logos) for match in matches
    )

    return header + matches_html


def main():
    # Check if predictions file exists
    if not PRED_CSV.exists():
        print(f"ERROR: Predictions file not found at {PRED_CSV}")
        print("Please run predict_scores.py first to generate predictions.")
        return

    # Read predictions
    df = pd.read_csv(PRED_CSV, parse_dates=["datetime"])
    print(f"Loaded {len(df)} predictions from {PRED_CSV}")

    # Debug: print columns and first few rows
    print("\nColumns in predictions file:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())

    # Check if dataframe is empty
    if df.empty:
        print("ERROR: Predictions dataframe is empty!")
        OUT_HTML.parent.mkdir(exist_ok=True, parents=True)
        OUT_HTML.write_text(
            build_page("<div class='error'>No predictions available. Please check your data pipeline.</div>"),
            encoding="utf-8")
        return

    # Load club logos
    clubs = set(df["home_team"]) | set(df["away_team"])
    print(f"\nLoading logos for {len(clubs)} clubs...")
    logos = {}
    for c in clubs:
        logo_path = LOGO_DIR / f"{c}.svg"
        if logo_path.exists():
            logos[c] = svg_to_base64(logo_path)
            print(f"  ✓ {c}")
        else:
            logos[c] = ""
            print(f"  ✗ {c} (logo not found)")

    # Group by gameweek
    print("\nGrouping matches by gameweek...")
    gameweeks = group_by_gameweek(df)
    print(f"Found {len(gameweeks)} gameweek(s)")

    # Generate HTML for each gameweek
    print("\nGenerating match cards by gameweek...")
    gameweek_sections = []

    for gw_num in sorted(gameweeks.keys()):
        matches = gameweeks[gw_num]
        print(f"\n  Gameweek {gw_num}: {len(matches)} matches")

        try:
            gw_html = build_gameweek_section(gw_num, matches, logos)
            gameweek_sections.append(gw_html)

            for match in matches:
                print(f"    ✓ {match['home_team']} vs {match['away_team']}")
        except Exception as e:
            print(f"    ✗ Error processing gameweek {gw_num}: {e}")

    rows_html = "\n".join(gameweek_sections)

    if not rows_html.strip():
        print("WARNING: No match HTML was generated!")
        rows_html = "<div class='error'>Error generating match predictions. Check console for details.</div>"

    # Write output
    OUT_HTML.parent.mkdir(exist_ok=True, parents=True)
    OUT_HTML.write_text(build_page(rows_html), encoding="utf-8")

    print(f"\n✔ Dashboard created: {OUT_HTML.resolve()}")
    print(f"Generated {len(gameweeks)} gameweek sections with {len(df)} total matches")


if __name__ == "__main__":
    main()