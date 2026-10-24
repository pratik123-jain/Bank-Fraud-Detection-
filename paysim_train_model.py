
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    roc_auc_score,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "dataset" / "paysim_cleaned.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "paysim_fraud_model.pkl"

FEATURES = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
TARGET = "isFraud"

def main():
    print("Loading cleaned PaySim dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    categorical = ["type"]
    numerical = [col for col in FEATURES if col not in categorical]

    preprocessor = ColumnTransformer([
        (
            "type",
            OneHotEncoder(handle_unknown="ignore"),
            categorical
        ),
        ("numbers", "passthrough", numerical),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )),
    ])

    print("Training Random Forest model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions,
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    ))

    print(
        "Average Precision (PR-AUC):",
        round(average_precision_score(y_test, probabilities), 4)
    )
    print(
        "ROC-AUC:",
        round(roc_auc_score(y_test, probabilities), 4)
    )

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nModel saved to:", MODEL_PATH)

if __name__ == "__main__":
    main()
