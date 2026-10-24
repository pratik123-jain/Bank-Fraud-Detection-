
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_DIR
    / "dataset"
    / "Paysim"
    / "PS_20174392719_1491204439457_log.csv"
)

OUTPUT_PATH = PROJECT_DIR / "dataset" / "paysim_cleaned.csv"

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
    print("Loading PaySim dataset...")

    df = pd.read_csv(DATASET_PATH)

    selected_columns = FEATURES + [TARGET]
    df = df[selected_columns].copy()

    df = df.dropna()

    df["type"] = df["type"].astype(str)

    for column in [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        TARGET,
    ]:
        df[column] = pd.to_numeric(df[column], errors="raise")

    df = df.drop_duplicates()

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nPreprocessing completed!")
    print("Saved file:", OUTPUT_PATH)
    print("Dataset shape:", df.shape)
    print("\nFraud distribution:")
    print(df[TARGET].value_counts())


if __name__ == "__main__":
    main()
