
from pathlib import Path

import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "fraud_model.pkl"
DATASET_PATH = PROJECT_DIR / "dataset" / "creditcard.csv"


def predict_transaction():
    if not MODEL_PATH.exists():
        print("Model not found. Run src/train_model.py first.")
        return

    if not DATASET_PATH.exists():
        print("Dataset not found.")
        return

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Use a real transaction row as a demo input
    df = pd.read_csv(DATASET_PATH)
    sample = df.drop(columns=["Class"]).iloc[[0]]

    probability = model.predict_proba(sample)[0][1]
    prediction = int(model.predict(sample)[0])

    print("\n=== Banking Fraud Detection ===")
    print("Demo transaction features loaded from dataset.")
    print(f"Estimated fraud probability: {probability:.2%}")

    if prediction == 1:
        print("Model prediction: Potential Fraud")
    else:
        print("Model prediction: Legitimate")

    print("\nNote: This is a demo prediction, not a financial decision.")


if __name__ == "__main__":
    predict_transaction()
