
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_DIR / "dataset" / "creditcard.csv"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_models():
    if not DATASET_PATH.exists():
        print("Dataset not found:", DATASET_PATH)
        return

    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        print(f"\n{name} - Classification Report")
        print(classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            zero_division=0
        ))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions, labels=[0, 1]))

        roc_auc = roc_auc_score(y_test, probabilities)
        pr_auc = average_precision_score(y_test, probabilities)

        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Average Precision (PR-AUC): {pr_auc:.4f}")

        results.append({
            "model": name,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "f1_fraud": classification_report(
                y_test, predictions, output_dict=True, zero_division=0
            )["1"]["f1-score"]
        })

        # Save each trained model
        filename = (
            "logistic_regression.pkl"
            if name == "Logistic Regression"
            else "fraud_model.pkl"
        )
        joblib.dump(model, MODEL_DIR / filename)
        print("Saved model:", MODEL_DIR / filename)

    results_df = pd.DataFrame(results).sort_values(
        "pr_auc", ascending=False
    )
    results_df.to_csv(MODEL_DIR / "model_comparison.csv", index=False)

    print("\n--- Model Comparison ---")
    print(results_df.to_string(index=False))
    print("\nModel training completed!")


if __name__ == "__main__":
    train_models()
