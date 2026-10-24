


from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_DIR / "dataset" / "creditcard.csv"


def inspect_dataset():
    print("Banking Fraud Detection - Dataset Inspection")
    print("-" * 45)

    if not DATASET_PATH.exists():
        print("ERROR: Dataset file not found!")
        print("Expected path:", DATASET_PATH)
        return

    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)

    print("\nDataset shape:", df.shape)

    print("\nFirst 5 records:")
    print(df.head())

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:", df.isnull().sum().sum())

    print("\nClass distribution:")
    print(df["Class"].value_counts())

    print("\nClass percentage:")
    print(
        (df["Class"].value_counts(normalize=True) * 100).round(3)
    )

    print("\nDataset inspection completed!")


if __name__ == "__main__":
    inspect_dataset()
