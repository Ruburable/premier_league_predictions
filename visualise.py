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
    .team-name { font-size:18px; font-weight:bold; margin-top:8px; }
    .scorers { font-size:14px; margin-top:6px; color:#444; }
    .center { width:25%; text-align:center; }
    .match-date { font-size:13px; color:#666; margin-bottom:8px; }
    .scoreline { font-size:28px; font-weight:bold; }
    .prob { font-size:13px; color:#555; margin-top:6px; }
    .muted { color:#aaa; }
    .error { color: red; padding: 20px; text-align: center; }
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
    # Check if predictions file exists
    if not PRED_CSV.exists():
        print(f"ERROR: Predictions file not found at {PRED_CSV}")
        print("Please run predict_scores.py first to generate predictions.")
        return

    # Read predictions
    df = pd.read_csv(PRED_CSV)
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

    # Generate HTML for each match
    print("\nGenerating match cards...")
    rows_html_list = []
    for idx, row in df.iterrows():
        try:
            match_html = build_match_html(row, logos)
            rows_html_list.append(match_html)
            print(f"  ✓ Match {idx + 1}: {row['home_team']} vs {row['away_team']}")
        except Exception as e:
            print(f"  ✗ Error processing match {idx + 1}: {e}")
            print(f"    Row data: {row.to_dict()}")

    rows_html = "\n".join(rows_html_list)

    if not rows_html.strip():
        print("WARNING: No match HTML was generated!")
        rows_html = "<div class='error'>Error generating match predictions. Check console for details.</div>"

    # Write output
    OUT_HTML.parent.mkdir(exist_ok=True, parents=True)
    OUT_HTML.write_text(build_page(rows_html), encoding="utf-8")

    print(f"\n✔ Dashboard created: {OUT_HTML.resolve()}")
    print(f"Generated {len(rows_html_list)} match cards")


if __name__ == "__main__":
    main()