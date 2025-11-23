import pandas as pd
from pathlib import Path

PRED_FILE = "output/predictions.csv"
OUTFILE = "output/predictions_dashboard.html"


# ----------------------------------------------------------
# 1. Built-in Base64 SVG logos for all Premier League clubs
# (Clean minimalist circular logo with club name)
# ----------------------------------------------------------
def simple_svg(club_name, color="#333"):
    """Returns a Base64 data URI SVG with club name text."""
    import base64
    svg = f'''
    <svg width="90" height="90" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <circle cx="50" cy="50" r="48" fill="{color}" stroke="white" stroke-width="3"/>
      <text x="50" y="56" font-size="22" text-anchor="middle" fill="white" font-family="Arial">{club_name[:3]}</text>
    </svg>
    '''
    enc = base64.b64encode(svg.encode()).decode()
    return f"data:image/svg+xml;base64,{enc}"


# fallback colors (club → color)
club_colors = {
    "Arsenal": "#EF0107",
    "Aston Villa": "#95BFE5",
    "Bournemouth": "#DA291C",
    "Brentford": "#E30613",
    "Brighton": "#0057B8",
    "Chelsea": "#034694",
    "Crystal Palace": "#1B458F",
    "Everton": "#003399",
    "Fulham": "#000000",
    "Liverpool": "#C8102E",
    "Luton": "#FA4616",
    "Man City": "#6CABDD",
    "Man United": "#DA291C",
    "Newcastle": "#241F20",
    "Nottingham Forest": "#DD0000",
    "Sheffield United": "#EE2737",
    "Tottenham": "#132257",
    "West Ham": "#7A263A",
    "Wolves": "#FDB913",
    "Burnley": "#6C1D45",
}


def load_logo(club):
    """Return base64 SVG (no external files required)."""
    color = club_colors.get(club, "#444")
    return f'<img src="{simple_svg(club, color)}" class="club-logo">'


# ----------------------------------------------------------
# Build each fixture card
# ----------------------------------------------------------
def build_fixture_card(r):
    home_logo = load_logo(r.home_team)
    away_logo = load_logo(r.away_team)

    home_scorers = ", ".join(r.expected_home_scorers.strip("[]").replace("'", "").split(","))
    away_scorers = ", ".join(r.expected_away_scorers.strip("[]").replace("'", "").split(","))

    return f"""
    <div class="fixture-card">
        <div class="teams">
            <div class="team-block">
                {home_logo}
                <div class="team-name">{r.home_team}</div>
            </div>
            <div class="score-block">
                <div class="pred-score">{r.scoreline}</div>
                <div class="prob-block">
                    <div><b>Home win:</b> {r.p_home_win:.2f}</div>
                    <div><b>Draw:</b> {r.p_draw:.2f}</div>
                    <div><b>Away win:</b> {r.p_away_win:.2f}</div>
                </div>
            </div>
            <div class="team-block">
                {away_logo}
                <div class="team-name">{r.away_team}</div>
            </div>
        </div>

        <div class="scorers">
            <div><b>{r.home_team} scorers:</b> {home_scorers}</div>
            <div><b>{r.away_team} scorers:</b> {away_scorers}</div>
        </div>
    </div>
    """


# ----------------------------------------------------------
# Main: Build HTML dashboard
# ----------------------------------------------------------
def main():
    df = pd.read_csv(PRED_FILE)

    html_cards = "\n".join(build_fixture_card(r) for _, r in df.iterrows())

    html = f"""
    <html>
    <head>
        <meta charset="UTF-8" />
        <title>Premier League Predictions</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #f6f7f9;
                margin: 0;
                padding: 30px;
            }}
            h1 {{
                text-align: center;
                margin-bottom: 30px;
                font-size: 36px;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
                gap: 20px;
            }}
            .fixture-card {{
                background: white;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.08);
                display: flex;
                flex-direction: column;
            }}
            .teams {{
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .team-block {{
                text-align: center;
            }}
            .team-name {{
                margin-top: 8px;
                font-size: 18px;
                font-weight: bold;
            }}
            .club-logo {{
                width: 70px;
                height: 70px;
            }}
            .pred-score {{
                font-size: 42px;
                font-weight: bold;
                margin-bottom: 10px;
            }}
            .prob-block {{
                font-size: 14px;
                opacity: 0.8;
            }}
            .scorers {{
                margin-top: 20px;
                font-size: 15px;
                line-height: 1.4;
            }}
        </style>
    </head>

    <body>
        <h1>Premier League: Predicted Scores & Scorers</h1>
        <div class="grid">
            {html_cards}
        </div>
    </body>
    </html>
    """

    Path("output").mkdir(exist_ok=True)
    with open(OUTFILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✔ Dashboard saved to {OUTFILE}\n")


if __name__ == "__main__":
    main()
