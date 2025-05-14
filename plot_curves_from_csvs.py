# plot_pr_curves_from_csvs.py
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_dir",
        type=str,
        default="pr_curves/",
        help="Directory containing PR curve CSVs",
    )
    args = parser.parse_args()

    plt.figure(figsize=(10, 7))

    for fname in os.listdir(args.csv_dir):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(args.csv_dir, fname)
        df = pd.read_csv(path)

        label = os.path.splitext(fname)[0]
        plt.plot(df["recall"], df["precision"], label=label)

    # Add perfect classification line
    plt.plot([0, 1], [1, 1], "k--", label="Perfect Classifier")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves on Test Data Beat")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("TEST_AUC_PR_curves.png")


if __name__ == "__main__":
    main()
