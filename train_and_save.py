"""
train_and_save.py
==================
Trains the final XGBoost model (exactly as in src/train.py) on the
augmented dataset and saves it as fibroid_model.pkl

Usage:
    python train_and_save.py
    python train_and_save.py --data augmented_fibroid_data.csv --out fibroid_model.pkl
"""

import argparse
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier


def create_model(random_state: int = 42) -> Pipeline:
    """Same pipeline as src/train.py -> create_model()."""
    return Pipeline([
        ("scaler", MinMaxScaler()),
        ("clf", XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=random_state,
        )),
    ])


def train(data_path: str, output_path: str = "fibroid_model.pkl"):
    print(f"[1/3] Loading augmented dataset: {data_path}")
    df = pd.read_csv(data_path)
    y = df["fibroid"]
    X = df.drop(columns=["fibroid"])
    print(f"      Shape: {X.shape}  |  Class counts: {y.value_counts().to_dict()}")

    print("[2/3] Training XGBoost pipeline (scaler + classifier)...")
    model = create_model()
    model.fit(X, y)

    print(f"[3/3] Saving model -> {output_path}")
    joblib.dump(model, output_path)
    print("      Done.")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="augmented_fibroid_data.csv")
    parser.add_argument("--out", default="fibroid_model.pkl")
    args = parser.parse_args()
    train(args.data, args.out)
