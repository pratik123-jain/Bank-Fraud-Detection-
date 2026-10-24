

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_DIR / "dataset" / "creditcard.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_visualizations():
    if not DATASET_PATH.exists():
        print("Dataset not found:", DATASET_PATH)
        return

    df = pd.read_csv(DATASET_PATH)
    sns.set_theme(style="whitegrid")

    # Chart 1: Fraud vs legitimate transactions
    counts = df["Class"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(x=["Legitimate", "Fraud"], y=counts.values)
    plt.title("Transaction Class Distribution")
    plt.ylabel("Number of Transactions")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150)
    plt.close()

    # Chart 2: Transaction amount distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df[df["Amount"] <= df["Amount"].quantile(0.99)],
        x="Amount",
        bins=50
    )
    plt.title("Transaction Amount Distribution (up to 99th percentile)")
    plt.xlabel("Transaction Amount")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "amount_distribution.png", dpi=150)
    plt.close()

    # Chart 3: Compare transaction amounts
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Class", y="Amount", showfliers=False)
    plt.title("Transaction Amount by Class")
    plt.xticks([0, 1], ["Legitimate", "Fraud"])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "amount_by_class.png", dpi=150)
    plt.close()

    print("EDA charts created successfully!")
    print("Charts saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    create_visualizations()
