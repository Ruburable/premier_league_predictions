#!/usr/bin/env python3
"""
visualize_premium_light.py

Reads:
  - sim_predictions.csv

Writes:
  - output/predictions_dashboard.html

Produces a premium-quality, light-themed HTML dashboard (one match per line),
with embedded SVG club badges, probability bars, xG-style bars, and scorer lists.
"""

import pandas as pd
from pathlib import Path
import base64
import math

INPUT_CSV = Path("sim_predictions.csv")
OUT_HTML = Path("output") / "predictions_dashboard.html"
OUT_HTML.parent.mkdir(parents=True, exist_ok=True)

# Minimal club palette for the placeholder circular badges (can be extended)
CLUB_COLORS = {
    "Arsenal": "#EF0107",
    "Chelsea": "#034694",
    "Liverpool": "#C8102E",
    "Manchester City": "#6CABDD",
    "Manchester United": "#DA291C",
    "Tottenham": "#132257",
    "Newcastle": "#241F20",
    "Aston Villa": "#95BFE5",
    "Brighton": "#0057B8",
    "Brentford": "#E30613",
    "Bournemouth": "#DA291C",
    "Crystal Palace": "#1B458F",
    "Everton": "#003399",
    "Fulham": "#000000",
    "Luton": "#FA4616",
    "Man City": "#6CABDD",
    "Man United": "#DA291C",
    "Wolves": "#FDB913",
    "West Ham": "#7A263A",
    "Nottingham Forest": "#DD0000",
}

def short_text(s, n=18):
    return s if len(s) <= n else s[:n-1] + "…"

def svg_badge_base64(club_name, size=84):
    color = CLUB_COLORS.get(club_name, "#667788")
    initials = "".join([w[0] for w in club_name.split()][:2]).upper()
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 100 100'>
  <defs>
    <linearGradient id='g{initials}' x1='0' x2='1'>
      <stop offset='0' stop-color='{color}' stop-opacity='1'/>
      <stop offset='1' stop-color='#ffffff' stop-opacity='0.07'/>
    </linearGradient>
    <filter id='s{initials}' x='-20%' y='-20%' width='140%' height='140%'>
      <feDropShadow dx='0' dy='2' stdDeviation='3' flood-color='#000' flood-opacity='0.12'/>
    </filter>
  </defs>
  <g filter='url(#s{initials})'>
    <circle cx='50' cy='50' r='46' fill='url(#g{initials})' stroke='rgba(0,0,0,0.06)' stroke-width='1'/>
  </g>
  <text x='50' y='58' font-family='Segoe UI, Roboto, Arial' font-size='34' text-anchor='middle' fill='white' font-weight='700'>{initials}</text>
</svg>"""
    b = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b}"

def prob_bar_html(p_home, p_draw, p_away, w=360, h=18):
    # Create an inline SVG bar with three segments
    total = max(p_home + p_draw + p_away, 1e-9)
    ph = p_home / total
    pd = p_draw / total
    pa = p_away / total
    x1 = round(ph * w, 2)
    x2 = round((ph + pd) * w, 2)
    svg = f"""<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="border-radius:6px;overflow:visible">
  <rect x="0" y="0" width="{x1}" height="{h}" rx="6" ry="6" fill="#3b82f6" />
  <rect x="{x1}" y="0" width="{max(x2-x1,0.5)}" height="{h}" fill="#6b7280" />
  <rect x="{x2}" y="0" width="{max(w-x2,0.5)}" height="{h}" fill="#ef4444" />
</svg>"""
    return svg

def xg_bar_html(hxg, axg, w=260, h=12):
    # Render two small bars for home and away xG proportions
    maxv = max(hxg, axg, 0.5)
    hw = min(round((hxg / maxv) * w), w)
    aw = min(round((axg / maxv) * w), w)
    svg = f"""<svg width="{w}" height="{h*2+8}" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="2" width="{hw}" height="{h}" rx="6" fill="#60a5fa" />
  <rect x="0" y="{h+6}" width="{aw}" height="{h}" rx="6" fill="#fb7185" />
</svg>"""
    return svg

def format_scorers(s):
    if not isinstance(s, str) or s.strip()=="":
        return "<span class='muted'>No scorers predicted</span>"
    parts = [p.strip() for p in s.split(",") if p.strip()]
    rows = "".join(f"<div class='scorer'>• {p}</div>" for p in parts)
    return rows

def build_match_row(r):
    home = r.get("home_team","")
    away = r.get("away_team","")
    home_logo = svg_badge_base64(home)
    away_logo = svg_badge_base64(away)
    # numeric fields (with safe defaults)
    try:
        ph = float(r.get("prob_home", 0.5))
    except Exception:
        ph = 0.5
    try:
        pd = float(r.get("prob_draw", 0.0))
    except Exception:
        pd = 0.0
    try:
        pa = float(r.get("prob_away", 0.5))
    except Exception:
        pa = 0.5
    try:
        hxg = float(r.get("pred_home_goals", 0.0))
    except Exception:
        hxg = 0.0
    try:
        axg = float(r.get("pred_away_goals", 0.0))
    except Exception:
        axg = 0.0

    prob_svg = prob_bar_html(ph, pd, pa, w=380, h=18)
    xg_svg = xg_bar_html(hxg, axg, w=260, h=10)
    home_scorers_html = format_scorers(r.get("home_scorers",""))
    away_scorers_html = format_scorers(r.get("away_scorers",""))

    html = f"""
    <div class="match-row">
      <div class="left">
        <img class="badge" src="{home_logo}" alt="{home}">
        <div class="team-name">{short_text(home,26)}</div>
        <div class="sub muted">Pred xG: {hxg:.2f}</div>
      </div>

      <div class="center">
        <div class="scoreline">{int(round(hxg))} — {int(round(axg))}</div>
        <div class="prob">{prob_svg}</div>
        <div class="xg">{xg_svg}</div>
      </div>

      <div class="right">
        <img class="badge" src="{away_logo}" alt="{away}">
        <div class="team-name">{short_text(away,26)}</div>
        <div class="sub muted">Pred xG: {axg:.2f}</div>
      </div>

      <div class="scorer-block">
        <div class="scorer-col">{home_scorers_html}</div>
        <div class="scorer-col">{away_scorers_html}</div>
      </div>
    </div>
    """
    return html

def build_html(rows_html):
    css = """
    :root{
      --card-bg: #ffffff;
      --muted: #6b7280;
      --accent: #0f172a;
      --panel: #f8fafc;
      --round: 12px;
    }
    body{
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
      background: linear-gradient(180deg, #f4f7fb 0%, #ffffff 100%);
      margin:0;
      padding:28px;
      color: #0f172a;
    }
    .container{
      max-width: 1100px;
      margin: 0 auto;
    }
    header{
      display:flex;
      align-items:center;
      justify-content:space-between;
      margin-bottom:20px;
    }
    h1{
      margin:0;
      font-size:24px;
      letter-spacing: -0.2px;
    }
    .subtitle{
      color: var(--muted);
      font-size:13px;
      margin-top:6px;
    }
    .panel{
      background: var(--card-bg);
      border-radius: var(--round);
      padding: 18px;
      box-shadow: 0 6px 18px rgba(16,24,40,0.06);
      margin-bottom:14px;
    }
    .match-row{
      display:grid;
      grid-template-columns: 220px 1fr 220px;
      grid-template-rows: auto auto;
      gap:10px 20px;
      align-items:center;
      padding:14px;
      border-radius:10px;
      background: linear-gradient(90deg, rgba(255,255,255,0.85), rgba(250,250,250,0.85));
      border: 1px solid rgba(15,23,42,0.04);
      margin-bottom:12px;
    }
    .left, .right{
      display:flex;
      flex-direction:column;
      align-items:center;
      gap:6px;
    }
    .badge{
      width:64px;
      height:64px;
      border-radius:8px;
      box-shadow: 0 6px 18px rgba(16,24,40,0.06);
    }
    .team-name{
      font-weight:700;
      font-size:16px;
      text-align:center;
    }
    .sub{
      font-size:12px;
      color:var(--muted);
    }
    .center{
      display:flex;
      flex-direction:column;
      align-items:center;
    }
    .scoreline{
      font-size:28px;
      font-weight:700;
      margin-bottom:8px;
      color: var(--accent);
    }
    .prob{ margin-bottom:8px; }
    .xg{ opacity:0.95; }
    .scorer-block{
      grid-column:1 / span 3;
      display:flex;
      justify-content:space-between;
      gap:20px;
      margin-top:8px;
    }
    .scorer-col{ width:48%; }
    .scorer{ font-size:13px; color:#0b1220; margin-bottom:6px; }
    .muted{ color: var(--muted); }
    footer{ margin-top:20px; color:var(--muted); font-size:13px; text-align:center; }
    @media (max-width:900px){
      .match-row{ grid-template-columns: 1fr; grid-template-rows: auto auto auto; }
      .scorer-block{ flex-direction:column; gap:10px; }
      .left, .right{ flex-direction:row; gap:12px; align-items:center; }
    }
    """
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Predicted Gameweek — Premium (Light)</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>{css}</style>
</head>
<body>
<div class="container">
  <header>
    <div>
      <h1>Upcoming Gameweek — Predictions</h1>
      <div class="subtitle">Predicted scorelines, probabilities, expected xG and likely scorers</div>
    </div>
    <div style="text-align:right;">
      <div style="font-weight:700;color:var(--muted);">Generated</div>
      <div style="font-size:13px;color:var(--muted);">{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</div>
    </div>
  </header>

  <section class="panel">
    {rows_html}
  </section>

  <footer>
    Data simulated for testing. Replace sim_predictions.csv with real predictions to view live output.
  </footer>
</div>
</body>
</html>
"""
    return html

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"{INPUT_CSV} not found. Run the simulation/prediction to create it first.")
    df = pd.read_csv(INPUT_CSV)
    # ensure columns exist and provide defaults
    df = df.fillna("")
    rows = []
    for _, r in df.iterrows():
        rows.append(build_match_row(r))
    rows_html = "\n".join(rows)
    content = build_html(rows_html)
    OUT_HTML.write_text(content, encoding="utf-8")
    print(f"Dashboard saved to: {OUT_HTML}")

if __name__ == "__main__":
    main()
