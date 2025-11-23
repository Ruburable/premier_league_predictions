import pandas as pd
import numpy as np

# ---------------------------------------------------
# STEP 1 — Simulate data that resembles real match features
# ---------------------------------------------------
def simulate_match_data():
    teams = [
        "Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester United",
        "Tottenham", "Newcastle", "Aston Villa"
    ]

    fixtures = [
        ("Arsenal", "Chelsea"),
        ("Manchester City", "Liverpool"),
        ("Tottenham", "Manchester United"),
        ("Aston Villa", "Newcastle"),
    ]

    rows = []
    for home, away in fixtures:
        rows.append({
            "home_team": home,
            "away_team": away,
            "home_strength": np.random.uniform(0.6, 1.0),
            "away_strength": np.random.uniform(0.5, 0.9),
            "home_form": np.random.uniform(0.4, 1.0),
            "away_form": np.random.uniform(0.4, 1.0),
            "home_xG": np.random.uniform(1.0, 2.5),
            "away_xG": np.random.uniform(0.8, 2.2),
        })

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------
# STEP 2 — Predict goals using simple statistical rules
# ---------------------------------------------------
def predict_scores(df):
    df["pred_home_goals"] = (df["home_xG"] * df["home_strength"]).round()
    df["pred_away_goals"] = (df["away_xG"] * df["away_strength"]).round()

    df["pred_home_goals"] = df["pred_home_goals"].clip(0, 4)
    df["pred_away_goals"] = df["pred_away_goals"].clip(0, 4)

    df["prob_home"] = np.random.uniform(0.35, 0.65)
    df["prob_away"] = np.random.uniform(0.2, 0.45)
    df["prob_draw"] = 1 - (df["prob_home"] + df["prob_away"])

    df["prob_draw"] = df["prob_draw"].clip(0.05, 0.45)

    df["prob_home"] /= (df["prob_home"] + df["prob_away"] + df["prob_draw"])
    df["prob_away"] /= (df["prob_home"] + df["prob_away"] + df["prob_draw"])
    df["prob_draw"] /= (df["prob_home"] + df["prob_away"] + df["prob_draw"])

    return df


# ---------------------------------------------------
# STEP 3 — Predict goal scorers (randomized)
# ---------------------------------------------------
def predict_scorers(df):
    players = {
        "Arsenal": ["Saka", "Havertz", "Martinelli"],
        "Chelsea": ["Palmer", "Nkunku", "Jackson"],
        "Liverpool": ["Salah", "Núñez", "Diaz"],
        "Manchester City": ["Haaland", "Foden", "De Bruyne"],
        "Manchester United": ["Rashford", "Fernandes", "Højlund"],
        "Tottenham": ["Son", "Maddison", "Kulusevski"],
        "Newcastle": ["Isak", "Gordon", "Barnes"],
        "Aston Villa": ["Watkins", "Bailey", "Diaby"],
    }

    home_scorers = []
    away_scorers = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        hg = int(row["pred_home_goals"])
        ag = int(row["pred_away_goals"])

        h_pick = list(np.random.choice(players[home], size=max(hg, 1), replace=True))
        a_pick = list(np.random.choice(players[away], size=max(ag, 1), replace=True))

        home_scorers.append(", ".join(h_pick) if hg > 0 else "")
        away_scorers.append(", ".join(a_pick) if ag > 0 else "")

    df["home_scorers"] = home_scorers
    df["away_scorers"] = away_scorers

    return df


# ---------------------------------------------------
# STEP 4 — Save predictions CSV automatically
# ---------------------------------------------------
def save_predictions(df):
    df.to_csv("sim_predictions.csv", index=False)
    print("✔ Saved: sim_predictions.csv")


# ---------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------
def main():
    print("Simulating match data...")
    df = simulate_match_data()

    print("Predicting scores...")
    df = predict_scores(df)

    print("Predicting goal scorers...")
    df = predict_scorers(df)

    save_predictions(df)

    print("✔ Simulation + prediction complete.")
    print("You can now run: python visualize_predictions.py")


if __name__ == "__main__":
    main()
