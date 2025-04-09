import os
import json
import argparse
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, accuracy_score
)
from tornet.data.loader import get_dataloader
import tornet.data.tfds.tornet.tornet_dataset_builder

logging.basicConfig(level=logging.INFO)

# Constants
DATA_ROOT = '/home/ubuntu/tfds'
TORNET_ROOT = DATA_ROOT
TFDS_DATA_DIR = DATA_ROOT
os.environ['TORNET_ROOT'] = DATA_ROOT
os.environ['TFDS_DATA_DIR'] = TFDS_DATA_DIR

def soft_voting_ensemble(models, dataset):
    y_trues, y_preds_all = [], []

    for batch in dataset:
        x, y = batch
        y_trues.append(y.numpy())

        preds = [m.predict(x, verbose=0) for m in models]
        preds = np.array(preds)  # shape: (n_models, batch, 1)

        soft_vote = np.mean(preds, axis=0)  # average across models
        y_preds_all.append(soft_vote)

    y_true_all = np.concatenate(y_trues)
    y_pred_all = np.concatenate(y_preds_all)
    return y_true_all, y_pred_all

def evaluate(y_true, y_pred):
    binary_preds = (y_pred > 0.5).astype(int)

    return {
        "AUCPR": average_precision_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, binary_preds),
        "Precision": precision_score(y_true, binary_preds),
        "Recall": recall_score(y_true, binary_preds),
        "F1": f1_score(y_true, binary_preds)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", nargs="+", required=True,
                        help="List of pretrained .keras model paths")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    logging.info("Loading models...")
    models = [keras.models.load_model(p, compile=False) for p in args.model_paths]
    logging.info(f"Loaded {len(models)} models")

    # Assuming same input keys across models
    input_keys = list(models[0].input.keys())

    # Load dataset
    test_years = list(range(2013, 2023))
    ds_test = get_dataloader("tensorflow-tfds", DATA_ROOT, test_years, "test", args.batch_size,
                             select_keys=input_keys)

    # Run ensemble predictions
    y_true, y_pred = soft_voting_ensemble(models, ds_test)

    # Evaluate
    results = evaluate(y_true, y_pred)
    logging.info("Ensemble Evaluation Results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
