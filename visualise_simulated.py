import pandas as pd
import base64
import requests
from io import BytesIO

# Remote club logos (light, consistent, high quality)
CLUB_LOGOS = {
    "Arsenal": "https://resources.premierleague.com/premierleague/badges/t3.svg",
    "Chelsea": "https://resources.premierleague.com/premierleague/badges/t8.svg",
    "Liverpool": "https://resources.premierleague.com/premierleague/badges/t14.svg",
    "Manchester City": "https://resources.premierleague.com/premierleague/badges/t43.svg",
    "Manchester United": "https://resources.premierleague.com/premierleague/badges/t1.svg",
    "Tottenham": "https://resources.premierleague.com/premierleague/badges/t6.svg",
    "Newcastle": "https://resources.premierleague.com/premierleague/badges/t4.svg",
    "Aston Villa": "https://resources.premierleague.com/premierleague/badges/t7.svg",
    "Brighton": "https://resources.premierleague.com/premierleague/badges/t36.svg",
}


def fetch_logo_as_base64(url):
    try:
        r = requests.get(url)
        r.raise_for_status()
        return base64.b64encode(r.content).decode()
    except:
        return ""


def build_match_row_html(row):
    hl = CLUB_LOGOS.get(row["home_team"], "")
    al = CLUB_LOGOS.get(row["away_team"], "")

    hl_encoded = fetch_logo_as_base64(hl) if hl else ""
    al_encoded = fetch_logo_as_base64(al) if al else ""

    scorers_home = "<br>".join(row["home_scorers"].split(","))
    scorers_away = "<br>".join(row["away_scorers"].split(","))

    return f"""
    <div class="match-card">
        <div class="team home">
            <img src="data:image/svg+xml;base64,{hl_encoded}" class="logo">
            <div class="team-name">{row['home_team']}</div>
            <div class="scorers">{scorers_home}</div>
        </div>

        <div class="score-block">
            <div class="score">{row['pred_home_goals']} - {row['pred_away_goals']}</div>
            <div class="prob-block">
                <span>Home: {row['prob_home']*100:.1f}%</span> |
                <span>Draw: {row['prob_draw']*100:.1f}%</span> |
                <span>Away: {row['prob_away']*100:.1f}%</span>
            </div>
        </div>

        <div class="team away">
            <img src="data:image/svg+xml;base64,{al_encoded}" class="logo">
            <div class="team-name">{row['away_team']}</div>
            <div class="scorers">{scorers_away}</div>
        </div>
    </div>
    """


def build_html(df):
    match_rows = "\n".join(df.apply(build_match_row_html, axis=1))

    html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Gameweek Predictions</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #1c1f26, #2c333d);
                margin: 0;
                padding: 30px;
                color: #f0f0f0;
            }}

            .title {{
                text-align: center;
                font-size: 38px;
                margin-bottom: 40px;
                font-weight: 700;
                letter-spacing: 1px;
            }}

            .match-card {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                background: rgba(255,255,255,0.08);
                border-radius: 12px;
                padding: 20px 25px;
                margin-bottom: 18px;
                box-shadow: 0 4px 18px rgba(0,0,0,0.4);
                backdrop-filter: blur(6px);
            }}

            .team {{
                width: 32%;
                text-align: center;
            }}

            .team-name {{
                font-size: 20px;
                font-weight: 600;
                margin-top: 8px;
            }}

            .logo {{
                width: 65px;
                height: 65px;
                filter: drop-shadow(0 1px 3px rgba(0,0,0,0.6));
            }}

            .score-block {{
                width: 33%;
                text-align: center;
            }}

            .score {{
                font-size: 42px;
                font-weight: 700;
                margin-bottom: 5px;
            }}

            .prob-block {{
                font-size: 14px;
                opacity: 0.9;
            }}

            .scorers {{
                margin-top: 10px;
                font-size: 14px;
                opacity: 0.85;
                line-height: 1.25;
            }}
        </style>
    </head>
    <body>
        <div class="title">Predicted Results – Upcoming Gameweek</div>

        {match_rows}

    </body>
    </html>
    """

    return html


def main():
    df = pd.read_csv("sim_predictions.csv")

    html = build_html(df)

    with open("predicted_gameweek.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("✔ saved predicted_gameweek.html")


if __name__ == "__main__":
    main()
