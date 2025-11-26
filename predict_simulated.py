import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

INPUT_DIR = "output"

def load_data():
    matches = pd.read_csv(f"{INPUT_DIR}/matches_master.csv")
    fixtures = pd.read_csv(f"{INPUT_DIR}/upcoming_fixtures.csv")
    return matches, fixtures

def prepare_training_data(matches):
    df = matches.copy()

    df["goal_diff"] = df["home_goals"] - df["away_goals"]
    df["xg_diff"] = df["home_xg"] - df["away_xg"]

    X = df[["home_xg", "away_xg", "xg_diff"]]
    y = df["goal_diff"]

    return X, y, df

def train_model(X, y):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model

def predict_upcoming(model, fixtures):
    fixtures["home_xg"] = np.random.uniform(0.8, 2.2, len(fixtures))
    fixtures["away_xg"] = np.random.uniform(0.8, 2.2, len(fixtures))
    fixtures["xg_diff"] = fixtures["home_xg"] - fixtures["away_xg"]

    pred = model.predict(fixtures[["home_xg","away_xg","xg_diff"]])
    fixtures["pred_goal_diff"] = pred
    fixtures["pred_score_home"] = (pred.clip(min=0) + np.random.uniform(0,1,len(pred))).round()
    fixtures["pred_score_away"] = (pred.clip(max=0)*-1 + np.random.uniform(0,1,len(pred))).round()

    fixtures.to_csv("output/predictions_upcoming.csv", index=False)
    print("✔ Saved: output/predictions_upcoming.csv")

def evaluate_past(model, df):
    df["pred"] = model.predict(df[["home_xg","away_xg","xg_diff"]])

    # fix clip() usage
    df["pred_home"] = df["pred"].clip(lower=0).round()
    df["pred_away"] = df["pred"].clip(upper=0).abs().round()

    df["error"] = abs(df["goal_diff"] - df["pred"])

    df.to_csv("output/predictions_past.csv", index=False)
    print("✔ Saved: output/predictions_past.csv")

    print("\nModel Evaluation:")
    print("----------------")
    print("MAE (Goal diff prediction):", df["error"].mean().round(3))


def main():
    matches, fixtures = load_data()
    X, y, df = prepare_training_data(matches)

    model = train_model(X, y)

    print("\n--- Predicting upcoming fixtures ---")
    predict_upcoming(model, fixtures)

    print("\n--- Evaluating accuracy on past matches ---")
    evaluate_past(model, df)


if __name__ == "__main__":
    main()
