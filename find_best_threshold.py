import argparse
import logging

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

logging.basicConfig(level=logging.INFO)


def load_predictions_labels(pred_path, label_path):
    preds = np.load(pred_path)
    labels = np.load(label_path)
    return preds, labels


def find_best_threshold(preds, labels, metric="f1", step=0.001):
    thresholds = np.arange(0.0, 1.0 + step, step)
    best_score = -1
    best_threshold = 0.5

    for t in thresholds:
        binary_preds = (preds >= t).astype(int)

        if metric == "f1":
            score = f1_score(labels, binary_preds)
        elif metric == "precision":
            score = precision_score(labels, binary_preds)
        elif metric == "recall":
            score = recall_score(labels, binary_preds)
        elif metric == "auc":
            score = roc_auc_score(labels, preds)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold, best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to .npy file with predicted probabilities",
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="Path to .npy file with ground truth labels (0 or 1)",
    )
    parser.add_argument(
        "--metric",
        default="f1",
        choices=["f1", "precision", "recall", "auc"],
        help="Metric to optimize for",
    )
    parser.add_argument(
        "--step", type=float, default=0.001, help="Step size for threshold search"
    )

    args = parser.parse_args()

    preds, labels = load_predictions_labels(args.predictions, args.labels)

    best_threshold, best_score = find_best_threshold(
        preds, labels, args.metric, args.step
    )

    logging.info(
        f"Best threshold for {args.metric}: {best_threshold:.3f} (score: {best_score:.4f})"
    )


if __name__ == "__main__":
    main()
