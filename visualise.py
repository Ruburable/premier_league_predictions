#!/usr/bin/env python3
"""
visualise.py

Creates enhanced HTML dashboard with:
1. Past Performance (current season in-sample predictions)
2. Upcoming Gameweek (next fixtures)
3. Future Predictions (all upcoming fixtures)
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

PRED_UPCOMING = Path("output/predictions_upcoming.csv")
PRED_HISTORICAL = Path("output/predictions_historical.csv")
PROJECTED_TABLE = Path("output/projected_table.csv")
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


def calculate_gameweek_number(df):
    """
    Calculate actual gameweek numbers from the start of the season.
    Matches within 4 days are in the same gameweek.
    """
    if df.empty:
        return df

    df = df.sort_values("datetime").reset_index(drop=True)
    df["gameweek"] = 0

    current_gw = 1
    current_gw_start = df.iloc[0]["datetime"]

    for idx, row in df.iterrows():
        match_date = row["datetime"]

        # If more than 4 days from current gameweek start, start new gameweek
        days_diff = (match_date - current_gw_start).total_seconds() / 86400
        if days_diff > 4 and idx > 0:
            current_gw += 1
            current_gw_start = match_date

        df.at[idx, "gameweek"] = current_gw

    return df


def group_by_gameweek(df):
    """Group matches by gameweek with actual numbers."""
    if df.empty:
        return {}

    df = calculate_gameweek_number(df)

    gameweeks = {}
    for gw_num in sorted(df["gameweek"].unique()):
        matches = df[df["gameweek"] == gw_num].to_dict('records')
        gameweeks[int(gw_num)] = matches

    return gameweeks


def get_logo_path(team_name):
    """Get relative path to team logo."""
    # Check if logo exists
    logo_path = LOGO_DIR / f"{team_name}.svg"
    if logo_path.exists():
        return f"logos/{team_name}.svg"
    else:
        return ""


# --------------------------------------------------
# Match card HTML for UPCOMING matches
# --------------------------------------------------
def build_upcoming_match_html(row, logos):
    home = row["home_team"]
    away = row["away_team"]

    ph = float(row.get("pred_home_goals", 0.0))
    pa = float(row.get("pred_away_goals", 0.0))

    prob_home = row.get("prob_home", None)
    prob_draw = row.get("prob_draw", None)
    prob_away = row.get("prob_away", None)

    # Format date
    date_str = ""
    if "datetime" in row and pd.notna(row["datetime"]):
        try:
            date_obj = pd.to_datetime(row["datetime"])
            date_str = f"<div class='match-date'>{date_obj.strftime('%a, %d %b %Y - %H:%M')}</div>"
        except:
            pass

    prob_html = (
        f"H: {float(prob_home) * 100:.0f}% | D: {float(prob_draw) * 100:.0f}% | A: {float(prob_away) * 100:.0f}%"
        if all(v is not None and pd.notna(v) for v in [prob_home, prob_draw, prob_away])
        else "N/A"
    )

    # Use relative paths instead of base64
    home_logo = logos.get(home, "")
    away_logo = logos.get(away, "")

    return f"""
    <div class="match-row">
      <div class="team team-left">
        <img class="badge" src="{home_logo}" onerror="this.style.display='none'" />
        <div class="team-name">{home}</div>
      </div>

      <div class="center">
        {date_str}
        <div class="scoreline">{ph:.1f} - {pa:.1f}</div>
        <div class="prob">{prob_html}</div>
      </div>

      <div class="team team-right">
        <img class="badge" src="{away_logo}" onerror="this.style.display='none'" />
        <div class="team-name">{away}</div>
      </div>
    </div>
    """


# --------------------------------------------------
# Match card HTML for HISTORICAL matches
# --------------------------------------------------
def build_historical_match_html(row, logos):
    home = row["home_team"]
    away = row["away_team"]

    # Actual results
    ah = float(row.get("home_goals", 0.0))
    aa = float(row.get("away_goals", 0.0))

    # Predictions
    ph = float(row.get("pred_home_goals", 0.0))
    pa = float(row.get("pred_away_goals", 0.0))

    # xG
    xg_home = float(row.get("home_xg", 0.0))
    xg_away = float(row.get("away_xg", 0.0))

    # Calculate accuracy
    error_home = abs(ah - ph)
    error_away = abs(aa - pa)
    avg_error = (error_home + error_away) / 2

    # Color code accuracy
    if avg_error < 0.5:
        accuracy_class = "accuracy-good"
        accuracy_text = "Excellent"
    elif avg_error < 1.0:
        accuracy_class = "accuracy-ok"
        accuracy_text = "Good"
    else:
        accuracy_class = "accuracy-poor"
        accuracy_text = "Poor"

    # Format date
    date_str = ""
    if "datetime" in row and pd.notna(row["datetime"]):
        try:
            date_obj = pd.to_datetime(row["datetime"])
            date_str = f"<div class='match-date'>{date_obj.strftime('%a, %d %b %Y')}</div>"
        except:
            pass

    # Use relative paths instead of base64
    home_logo = logos.get(home, "")
    away_logo = logos.get(away, "")

    return f"""
    <div class="match-row historical">
      <div class="team team-left">
        <img class="badge" src="{home_logo}" onerror="this.style.display='none'" />
        <div class="team-name">{home}</div>
        <div class="xg-display">xG: {xg_home:.2f}</div>
      </div>

      <div class="center">
        {date_str}
        <div class="actual-score">{ah:.0f} - {aa:.0f}</div>
        <div class="predicted-score">Predicted: {ph:.1f} - {pa:.1f}</div>
        <div class="accuracy {accuracy_class}">{accuracy_text} (±{avg_error:.2f})</div>
      </div>

      <div class="team team-right">
        <img class="badge" src="{away_logo}" onerror="this.style.display='none'" />
        <div class="team-name">{away}</div>
        <div class="xg-display">xG: {xg_away:.2f}</div>
      </div>
    </div>
    """


# --------------------------------------------------
# Section builders
# --------------------------------------------------
def build_gameweek_section(gw_num, matches, logos, is_historical=False):
    """Build HTML for a gameweek section."""
    first_date = matches[0]["datetime"]
    last_date = matches[-1]["datetime"]

    if first_date.date() == last_date.date():
        date_range = first_date.strftime("%A, %d %B %Y")
    else:
        date_range = f"{first_date.strftime('%d %b')} - {last_date.strftime('%d %b %Y')}"

    header = f"""
    <div class="gameweek-header {'historical' if is_historical else ''}">
        <h3>Gameweek {gw_num}</h3>
        <div class="gameweek-date">{date_range}</div>
        <div class="gameweek-count">{len(matches)} matches</div>
    </div>
    """

    if is_historical:
        matches_html = "\n".join(
            build_historical_match_html(match, logos) for match in matches
        )
    else:
        matches_html = "\n".join(
            build_upcoming_match_html(match, logos) for match in matches
        )

    return header + matches_html


def build_table_html(table_df, logos):
    """Build HTML for the projected league table."""
    if table_df.empty:
        return '<div class="error">No table projection available. Run predict_final_table.py to generate.</div>'

    rows_html = ""
    for _, row in table_df.iterrows():
        pos = row['Pos']
        team = row['Team']

        # Position indicator
        if pos <= 4:
            indicator = '<span class="pos-indicator champions-league"></span>'
            indicator_class = 'champions-league'
        elif pos == 5:
            indicator = '<span class="pos-indicator europa-league"></span>'
            indicator_class = 'europa-league'
        elif pos >= 18:
            indicator = '<span class="pos-indicator relegation"></span>'
            indicator_class = 'relegation'
        else:
            indicator = '<span class="pos-indicator"></span>'
            indicator_class = ''

        # Goal difference color
        gd = row['GD']
        gd_class = 'positive' if gd > 0 else ('negative' if gd < 0 else '')
        gd_display = f"{gd:+d}" if gd != 0 else "0"

        # Team logo
        team_logo = logos.get(team, "")
        logo_html = f'<img src="{team_logo}" class="team-logo" onclick="showTeamPage(\'{team}\')" />' if team_logo else ''

        rows_html += f"""
        <tr class="{indicator_class}">
            <td class="pos">{indicator}{pos}</td>
            <td class="team-name" onclick="showTeamPage('{team}')">{logo_html}{team}</td>
            <td style="text-align: center;">{row['P']}</td>
            <td style="text-align: center;">{row['W']}</td>
            <td style="text-align: center;">{row['D']}</td>
            <td style="text-align: center;">{row['L']}</td>
            <td style="text-align: center;">{row['GF']}</td>
            <td style="text-align: center;">{row['GA']}</td>
            <td style="text-align: center;" class="{gd_class}">{gd_display}</td>
            <td class="points" style="text-align: center;">{row['Pts']}</td>
        </tr>
        """

    legend_html = """
    <div class="table-legend">
        <span><span class="pos-indicator champions-league"></span> Champions League (Top 4)</span>
        <span><span class="pos-indicator europa-league"></span> Europa League (5th)</span>
        <span><span class="pos-indicator relegation"></span> Relegation (Bottom 3)</span>
    </div>
    """

    return f"""
    <div class="league-table">
        <table>
            <thead>
                <tr>
                    <th>Pos</th>
                    <th>Team</th>
                    <th style="text-align: center;">P</th>
                    <th style="text-align: center;">W</th>
                    <th style="text-align: center;">D</th>
                    <th style="text-align: center;">L</th>
                    <th style="text-align: center;">GF</th>
                    <th style="text-align: center;">GA</th>
                    <th style="text-align: center;">GD</th>
                    <th style="text-align: center;">Pts</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        {legend_html}
    </div>
    """


def build_team_pages_data(upcoming_df, historical_df, logos):
    """Build JSON data for individual team pages."""
    if upcoming_df.empty and historical_df.empty:
        return "{}"

    teams_data = {}

    # Combine all teams
    all_teams = set()
    if not upcoming_df.empty:
        all_teams.update(upcoming_df["home_team"])
        all_teams.update(upcoming_df["away_team"])
    if not historical_df.empty:
        all_teams.update(historical_df["home_team"])
        all_teams.update(historical_df["away_team"])

    for team in all_teams:
        team_matches = []

        # Get historical matches
        if not historical_df.empty:
            team_historical = historical_df[
                (historical_df["home_team"] == team) |
                (historical_df["away_team"] == team)
                ].copy()

            for _, match in team_historical.iterrows():
                is_home = match["home_team"] == team
                team_matches.append({
                    "type": "historical",
                    "date": match["datetime"].strftime("%Y-%m-%d"),
                    "opponent": match["away_team"] if is_home else match["home_team"],
                    "venue": "Home" if is_home else "Away",
                    "actual_score": f"{int(match['home_goals'])}-{int(match['away_goals'])}",
                    "predicted_score": f"{match['pred_home_goals']:.1f}-{match['pred_away_goals']:.1f}",
                    "result": "W" if (is_home and match['home_goals'] > match['away_goals']) or (
                                not is_home and match['away_goals'] > match['home_goals']) else (
                        "D" if match['home_goals'] == match['away_goals'] else "L")
                })

        # Get upcoming matches
        if not upcoming_df.empty:
            team_upcoming = upcoming_df[
                (upcoming_df["home_team"] == team) |
                (upcoming_df["away_team"] == team)
                ].copy()

            for _, match in team_upcoming.iterrows():
                is_home = match["home_team"] == team
                team_matches.append({
                    "type": "upcoming",
                    "date": match["datetime"].strftime("%Y-%m-%d"),
                    "opponent": match["away_team"] if is_home else match["home_team"],
                    "venue": "Home" if is_home else "Away",
                    "predicted_score": f"{match['pred_home_goals']:.1f}-{match['pred_away_goals']:.1f}",
                    "prob_win": match['prob_home'] if is_home else match['prob_away'],
                    "prob_draw": match['prob_draw'],
                    "prob_loss": match['prob_away'] if is_home else match['prob_home']
                })

        teams_data[team] = {
            "matches": team_matches,
            "logo": logos.get(team, "")
        }

    import json
    return json.dumps(teams_data)


def build_section_header(title, subtitle, icon):
    """Build HTML for a section header."""
    return f"""
    <div class="section-header">
        <h2>{title}</h2>
        <p class="section-subtitle">{subtitle}</p>
    </div>
    """


# --------------------------------------------------
# Page shell
# --------------------------------------------------
def build_page(past_html, upcoming_html, future_html, table_html, stats_html, season_display, teams_data_json):
    css = """
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { 
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        min-height: 100vh;
    }
    .container { max-width: 1200px; margin: auto; }

    /* Tabs */
    .tabs {
        display: flex;
        gap: 10px;
        margin-bottom: 30px;
        background: white;
        padding: 10px;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .tab {
        flex: 1;
        padding: 15px 25px;
        border: none;
        background: transparent;
        color: #7f8c8d;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    .tab:hover {
        background: #f8f9fa;
        color: #2c3e50;
    }
    .tab.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    /* Tab content */
    .tab-content {
        display: none;
    }
    .tab-content.active {
        display: block;
        animation: fadeIn 0.3s ease;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Table styles */
    .league-table {
        background: white;
        border-radius: 12px;
        padding: 30px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-top: 20px;
    }
    .league-table table {
        width: 100%;
        border-collapse: collapse;
    }
    .league-table th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 10px;
        text-align: left;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .league-table th:first-child {
        border-radius: 8px 0 0 0;
    }
    .league-table th:last-child {
        border-radius: 0 8px 0 0;
    }
    .league-table td {
        padding: 14px 10px;
        border-bottom: 1px solid #f0f0f0;
        font-size: 14px;
    }
    .league-table tr:hover {
        background: #f8f9fa;
    }
    .league-table .team-name {
        font-weight: 600;
        color: #2c3e50;
        cursor: pointer;
        transition: color 0.2s ease;
    }
    .league-table .team-name:hover {
        color: #667eea;
        text-decoration: underline;
    }
    .league-table .team-logo {
        width: 24px;
        height: 24px;
        vertical-align: middle;
        margin-right: 8px;
        cursor: pointer;
        transition: transform 0.2s ease;
    }
    .league-table .team-logo:hover {
        transform: scale(1.2);
    }
    .league-table .pos {
        font-weight: 700;
        font-size: 16px;
        color: #7f8c8d;
        width: 50px;
    }
    .league-table .pos-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .league-table .champions-league {
        background: #81C784;
    }
    .league-table .europa-league {
        background: #FFD54F;
    }
    .league-table .relegation {
        background: #E57373;
    }
    .league-table .points {
        font-weight: 700;
        font-size: 16px;
        color: #667eea;
    }
    .league-table .positive {
        color: #2E7D32;
        font-weight: 600;
    }
    .league-table .negative {
        color: #C62828;
        font-weight: 600;
    }
    .table-legend {
        margin-top: 20px;
        padding: 15px;
        background: #f8f9fa;
        border-radius: 8px;
        font-size: 13px;
        color: #7f8c8d;
    }
    .table-legend span {
        margin-right: 20px;
        display: inline-block;
    }

    /* Team overlay */
    .team-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.8);
        z-index: 1000;
        opacity: 0;
        transition: opacity 0.3s ease;
        overflow-y: auto;
    }
    .team-overlay.active {
        opacity: 1;
    }
    .team-page {
        max-width: 900px;
        margin: 50px auto;
        background: white;
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    .team-page-header {
        margin-bottom: 30px;
    }
    .back-button {
        background: #667eea;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        cursor: pointer;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 20px;
        transition: background 0.2s ease;
    }
    .back-button:hover {
        background: #764ba2;
    }
    .team-title {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .team-title h2 {
        margin: 0;
        font-size: 32px;
        color: #2c3e50;
    }
    .team-page-logo {
        width: 60px;
        height: 60px;
    }
    .team-matches h3 {
        color: #2c3e50;
        margin: 30px 0 15px 0;
        font-size: 24px;
    }
    .matches-list {
        display: grid;
        gap: 15px;
    }
    .team-match {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #ddd;
    }
    .team-match.upcoming {
        border-left-color: #667eea;
    }
    .team-match.win {
        border-left-color: #2E7D32;
        background: #f1f8f4;
    }
    .team-match.draw {
        border-left-color: #F9A825;
        background: #fffbf0;
    }
    .team-match.loss {
        border-left-color: #C62828;
        background: #fef5f5;
    }
    .match-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 10px;
        font-size: 13px;
        color: #7f8c8d;
    }
    .match-result-badge {
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
    }
    .match-result-badge.win {
        background: #2E7D32;
        color: white;
    }
    .match-result-badge.draw {
        background: #F9A825;
        color: white;
    }
    .match-result-badge.loss {
        background: #C62828;
        color: white;
    }
    .match-opponent {
        font-size: 18px;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 10px;
    }
    .match-prediction, .match-scores {
        font-size: 14px;
    }
    .predicted-score {
        font-weight: 600;
        color: #667eea;
        margin-bottom: 5px;
    }
    .match-probs {
        font-size: 13px;
        color: #7f8c8d;
    }
    .actual-score {
        font-weight: 700;
        font-size: 16px;
        color: #2c3e50;
        margin-bottom: 3px;
    }
    .predicted-score-small {
        font-size: 13px;
        color: #7f8c8d;
    }

    /* Main header */
    .main-header {
        text-align: center;
        color: white;
        padding: 40px 20px 20px 20px;
        margin-bottom: 20px;
    }
    .main-header h1 {
        font-size: 48px;
        font-weight: 700;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .main-header .subtitle {
        font-size: 18px;
        opacity: 0.9;
        margin-bottom: 15px;
    }
    .season-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 8px 20px;
        border-radius: 20px;
        font-size: 16px;
        font-weight: 600;
        margin-top: 10px;
        backdrop-filter: blur(10px);
    }

    /* Stats cards */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-bottom: 30px;
    }
    .stat-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .stat-value {
        font-size: 36px;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 5px;
    }
    .stat-label {
        font-size: 14px;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Section headers */
    .section-header {
        background: white;
        padding: 30px;
        border-radius: 15px;
        margin: 40px 0 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .section-header h2 {
        font-size: 32px;
        color: #2c3e50;
        margin-bottom: 8px;
    }
    .section-subtitle {
        color: #7f8c8d;
        font-size: 16px;
    }

    /* Gameweek headers */
    .gameweek-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px 30px;
        border-radius: 12px;
        margin: 25px 0 15px 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .gameweek-header.historical {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .gameweek-header h3 {
        margin: 0 0 8px 0;
        font-size: 24px;
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

    /* Match rows */
    .match-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: white;
        padding: 25px 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 15px;
        transition: all 0.3s ease;
    }
    .match-row:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    .match-row.historical {
        background: #fafbfc;
    }

    /* Teams */
    .team {
        flex: 1;
        text-align: center;
        padding: 0 15px;
    }
    .badge {
        width: 60px;
        height: 60px;
        object-fit: contain;
        margin-bottom: 10px;
    }
    .team-name {
        font-size: 16px;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 5px;
    }
    .xg-display {
        font-size: 13px;
        color: #7f8c8d;
        font-weight: 500;
    }

    /* Center section */
    .center {
        flex: 0 0 280px;
        text-align: center;
        padding: 0 20px;
    }
    .match-date {
        font-size: 13px;
        color: #95a5a6;
        margin-bottom: 10px;
        font-weight: 500;
    }

    /* Scores */
    .scoreline {
        font-size: 36px;
        font-weight: 700;
        color: #667eea;
        letter-spacing: 3px;
        margin: 5px 0;
    }
    .actual-score {
        font-size: 42px;
        font-weight: 700;
        color: #2c3e50;
        letter-spacing: 3px;
        margin: 5px 0;
    }
    .predicted-score {
        font-size: 14px;
        color: #7f8c8d;
        margin: 8px 0;
    }

    /* Probabilities */
    .prob {
        font-size: 13px;
        color: #7f8c8d;
        margin-top: 10px;
        background: #f8f9fa;
        padding: 8px 12px;
        border-radius: 6px;
        font-weight: 500;
    }

    /* Accuracy indicators */
    .accuracy {
        font-size: 12px;
        padding: 6px 12px;
        border-radius: 20px;
        margin-top: 8px;
        display: inline-block;
        font-weight: 600;
    }
    .accuracy-good {
        background: #d4edda;
        color: #155724;
    }
    .accuracy-ok {
        background: #fff3cd;
        color: #856404;
    }
    .accuracy-poor {
        background: #f8d7da;
        color: #721c24;
    }

    /* Responsive */
    @media (max-width: 768px) {
        .match-row {
            flex-direction: column;
            padding: 20px 15px;
        }
        .team, .center {
            width: 100%;
            margin: 10px 0;
        }
        .badge {
            width: 50px;
            height: 50px;
        }
        .main-header h1 {
            font-size: 32px;
        }
        .stats-grid {
            grid-template-columns: 1fr;
        }
    }

    .error {
        color: #e74c3c;
        padding: 40px;
        text-align: center;
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    """

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Premier League Predictions Dashboard</title>
<style>{css}</style>
</head>
<body>
<div class="container">

<div class="main-header">
<h1>Premier League Predictions</h1>
<div class="subtitle">AI-powered match predictions using XGBoost and xG data</div>
<div class="season-badge">Season {season_display}</div>
</div>

{stats_html}

<div class="tabs">
    <button class="tab {'active' if past_html else ''}" onclick="switchTab('past')">Past Performance</button>
    <button class="tab {'active' if not past_html and not table_html else ''}" onclick="switchTab('future')">Upcoming Fixtures</button>
    <button class="tab {'active' if not past_html and table_html else ''}" onclick="switchTab('table')">Projected Table</button>
</div>

<div id="past-content" class="tab-content {'active' if past_html else ''}">
{past_html if past_html else '<div class="error">No historical data available. Run the pipeline to generate predictions for current season matches.</div>'}
</div>

<div id="future-content" class="tab-content {'active' if not past_html and not table_html else ''}">
{upcoming_html}
{future_html}
</div>

<div id="table-content" class="tab-content {'active' if not past_html and table_html else ''}">
{build_section_header("Projected Final Table", f"{season_display} season - Based on actual results + predicted outcomes", "")}
{table_html}
</div>

<script>
// Team data embedded
const teamsData = {teams_data_json};

function showTeamPage(teamName) {{
    const teamData = teamsData[teamName];
    if (!teamData) {{
        alert('No data available for ' + teamName);
        return;
    }}

    // Build team page HTML
    let html = `
    <div class="team-page">
        <div class="team-page-header">
            <button class="back-button" onclick="closeTeamPage()">← Back to Table</button>
            <div class="team-title">
                ${{teamData.logo ? '<img src="' + teamData.logo + '" class="team-page-logo" />' : ''}}
                <h2>${{teamName}}</h2>
            </div>
        </div>
        <div class="team-matches">
    `;

    // Upcoming matches
    const upcomingMatches = teamData.matches.filter(m => m.type === 'upcoming');
    if (upcomingMatches.length > 0) {{
        html += '<h3>Upcoming Fixtures</h3>';
        html += '<div class="matches-list">';
        upcomingMatches.forEach(match => {{
            html += `
            <div class="team-match upcoming">
                <div class="match-header">
                    <span class="match-date">${{match.date}}</span>
                    <span class="match-venue">${{match.venue}}</span>
                </div>
                <div class="match-opponent">vs ${{match.opponent}}</div>
                <div class="match-prediction">
                    <div class="predicted-score">Predicted: ${{match.predicted_score}}</div>
                    <div class="match-probs">
                        Win: ${{(match.prob_win * 100).toFixed(0)}}% | 
                        Draw: ${{(match.prob_draw * 100).toFixed(0)}}% | 
                        Loss: ${{(match.prob_loss * 100).toFixed(0)}}%
                    </div>
                </div>
            </div>
            `;
        }});
        html += '</div>';
    }}

    // Historical matches
    const historicalMatches = teamData.matches.filter(m => m.type === 'historical');
    if (historicalMatches.length > 0) {{
        html += '<h3>Recent Results</h3>';
        html += '<div class="matches-list">';
        historicalMatches.forEach(match => {{
            const resultClass = match.result === 'W' ? 'win' : (match.result === 'D' ? 'draw' : 'loss');
            html += `
            <div class="team-match historical ${{resultClass}}">
                <div class="match-header">
                    <span class="match-date">${{match.date}}</span>
                    <span class="match-venue">${{match.venue}}</span>
                    <span class="match-result-badge ${{resultClass}}">${{match.result}}</span>
                </div>
                <div class="match-opponent">vs ${{match.opponent}}</div>
                <div class="match-scores">
                    <div class="actual-score">Result: ${{match.actual_score}}</div>
                    <div class="predicted-score-small">Predicted: ${{match.predicted_score}}</div>
                </div>
            </div>
            `;
        }});
        html += '</div>';
    }}

    html += '</div></div>';

    // Create overlay
    const overlay = document.createElement('div');
    overlay.id = 'team-overlay';
    overlay.className = 'team-overlay';
    overlay.innerHTML = html;
    document.body.appendChild(overlay);

    // Fade in
    setTimeout(() => overlay.classList.add('active'), 10);
}}

function closeTeamPage() {{
    const overlay = document.getElementById('team-overlay');
    if (overlay) {{
        overlay.classList.remove('active');
        setTimeout(() => overlay.remove(), 300);
    }}
}}

function switchTab(tab) {{
    // Update tab buttons
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(t => t.classList.remove('active'));

    // Update content
    const contents = document.querySelectorAll('.tab-content');
    contents.forEach(c => c.classList.remove('active'));

    if (tab === 'past') {{
        tabs[0].classList.add('active');
        document.getElementById('past-content').classList.add('active');
    }} else if (tab === 'future') {{
        tabs[1].classList.add('active');
        document.getElementById('future-content').classList.add('active');
    }} else if (tab === 'table') {{
        tabs[2].classList.add('active');
        document.getElementById('table-content').classList.add('active');
    }}
}}
</script>

</div>
</body>
</html>
"""


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("=" * 80)
    print("ENHANCED DASHBOARD GENERATOR")
    print("=" * 80)

    # Determine current season
    from datetime import datetime
    now = datetime.now()
    if now.month >= 8:
        season_start = now.year
    else:
        season_start = now.year - 1
    season_end = season_start + 1
    season_display = f"{season_start}/{str(season_end)[-2:]}"

    # Load data
    upcoming_df = pd.read_csv(PRED_UPCOMING, parse_dates=["datetime"]) if PRED_UPCOMING.exists() else pd.DataFrame()
    historical_df = pd.read_csv(PRED_HISTORICAL,
                                parse_dates=["datetime"]) if PRED_HISTORICAL.exists() else pd.DataFrame()
    table_df = pd.read_csv(PROJECTED_TABLE) if PROJECTED_TABLE.exists() else pd.DataFrame()

    print(f"\nLoaded {len(upcoming_df)} upcoming predictions")
    print(f"Loaded {len(historical_df)} historical predictions")
    print(f"Loaded projected table: {len(table_df)} teams" if not table_df.empty else "No projected table found")

    # Get unique clubs and create logo path mapping
    clubs = set()
    if not upcoming_df.empty:
        clubs.update(upcoming_df["home_team"])
        clubs.update(upcoming_df["away_team"])
    if not historical_df.empty:
        clubs.update(historical_df["home_team"])
        clubs.update(historical_df["away_team"])

    print(f"\nMapping logos for {len(clubs)} clubs...")
    logos = {}
    for c in clubs:
        logo_path = get_logo_path(c)
        if logo_path:
            logos[c] = logo_path
            print(f"  Found: {c}")
        else:
            logos[c] = ""
            print(f"  Missing: {c}")

    # Build stats summary
    stats_html = ""
    if not historical_df.empty:
        total_matches = len(historical_df)
        avg_mae = (
                          abs(historical_df["home_goals"] - historical_df["pred_home_goals"]).mean() +
                          abs(historical_df["away_goals"] - historical_df["pred_away_goals"]).mean()
                  ) / 2

        # Calculate accuracy percentage (within 1 goal)
        accurate_predictions = (
                (abs(historical_df["home_goals"] - historical_df["pred_home_goals"]) <= 1.0) &
                (abs(historical_df["away_goals"] - historical_df["pred_away_goals"]) <= 1.0)
        ).sum()
        accuracy_pct = (accurate_predictions / total_matches * 100) if total_matches > 0 else 0

        upcoming_count = len(upcoming_df)

        stats_html = f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{total_matches}</div>
                <div class="stat-label">Season Matches</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">±{avg_mae:.2f}</div>
                <div class="stat-label">Avg Prediction Error</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{accuracy_pct:.0f}%</div>
                <div class="stat-label">Accuracy (±1 goal)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{upcoming_count}</div>
                <div class="stat-label">Upcoming Fixtures</div>
            </div>
        </div>
        """
    elif not upcoming_df.empty:
        # Only show upcoming fixtures count if no historical data
        upcoming_count = len(upcoming_df)
        stats_html = f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{upcoming_count}</div>
                <div class="stat-label">Upcoming Fixtures</div>
            </div>
        </div>
        """

    # Build past performance section
    past_html = ""
    if not historical_df.empty:
        past_html = build_section_header(
            "Season Performance Analysis",
            f"{season_display} season - Model predictions vs actual results",
            "📊"
        )

        gameweeks = group_by_gameweek(historical_df)
        for gw_num in sorted(gameweeks.keys(), reverse=True):  # Most recent first
            past_html += build_gameweek_section(gw_num, gameweeks[gw_num], logos, is_historical=True)

    # Build upcoming gameweek section
    upcoming_html = ""
    future_html = ""

    if not upcoming_df.empty:
        gameweeks = group_by_gameweek(upcoming_df)

        if gameweeks:
            # Next gameweek
            next_gw = min(gameweeks.keys())
            upcoming_html = build_section_header(
                "Next Gameweek",
                f"Gameweek {next_gw} - Upcoming matches with win probabilities",
                "🔮"
            )
            upcoming_html += build_gameweek_section(next_gw, gameweeks[next_gw], logos)

            # Future gameweeks
            if len(gameweeks) > 1:
                future_html = build_section_header(
                    "Future Gameweeks",
                    f"{len(gameweeks) - 1} additional gameweeks predicted",
                    "📅"
                )

                for gw_num in sorted(gameweeks.keys())[1:]:  # Skip first (already shown)
                    future_html += build_gameweek_section(gw_num, gameweeks[gw_num], logos)
    else:
        upcoming_html = '<div class="error">No upcoming fixtures available. The season may have ended or there is a break in fixtures.</div>'

    # Build table section
    table_html = build_table_html(table_df, logos)

    # Build team pages data
    teams_data_json = build_team_pages_data(upcoming_df, historical_df, logos)

    # Generate page with season info and table
    html_content = build_page(past_html, upcoming_html, future_html, table_html, stats_html, season_display,
                              teams_data_json)

    # Write output
    OUT_HTML.parent.mkdir(exist_ok=True, parents=True)
    OUT_HTML.write_text(html_content, encoding="utf-8")

    # Get file size
    file_size = OUT_HTML.stat().st_size / 1024  # KB
    if file_size > 1024:
        size_display = f"{file_size / 1024:.1f} MB"
    else:
        size_display = f"{file_size:.1f} KB"

    print(f"\n✅ Enhanced dashboard created: {OUT_HTML.resolve()}")
    print(f"   File size: {size_display}")
    print(f"\n    Dashboard sections:")
    print(f"      Season: {season_display}")
    print(f"      Past Performance: {len(historical_df)} matches")
    print(f"      Upcoming Gameweek: {len(gameweeks.get(min(gameweeks.keys()), [])) if gameweeks else 0} matches")
    print(
        f"      Future Predictions: {len(upcoming_df) - len(gameweeks.get(min(gameweeks.keys()), [])) if gameweeks else 0} matches")
    print(
        f"      Projected Table: {len(table_df)} teams" if not table_df.empty else "      Projected Table: Not available")

    print(f"\n💡 Note: Keep the 'logos' folder in the same directory as the HTML file")
    print(f"   HTML location: output/predictions_dashboard.html")
    print(f"   Logos location: output/logos/")


if __name__ == "__main__":
    main()