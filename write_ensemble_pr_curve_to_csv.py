import argparse
import logging
import os
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from sklearn.metrics import precision_recall_curve
from tensorflow import keras

from tornet.data.loader import get_dataloader

# Set environment variables
TFDS_DATA_DIR = "/home/ubuntu/tfds"
DATA_ROOT = TFDS_DATA_DIR
os.environ["TORNET_ROOT"] = DATA_ROOT
os.environ["TFDS_DATA_DIR"] = TFDS_DATA_DIR

logging.basicConfig(level=logging.INFO)


@keras.utils.register_keras_serializable()
class FillNaNs(keras.layers.Layer):
    def __init__(self, fill_val, **kwargs):
        super().__init__(**kwargs)
        self.fill_val = tf.convert_to_tensor(fill_val, dtype=tf.float32)

    def call(self, x):
        return tf.where(tf.math.is_nan(x), self.fill_val, x)


@keras.utils.register_keras_serializable()
class ExpandDimsTwice(keras.layers.Layer):
    def call(self, inputs):
        return tf.expand_dims(tf.expand_dims(inputs, axis=1), axis=1)


@keras.utils.register_keras_serializable()
class StackAvgMax(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.stack(inputs, axis=1)


@keras.utils.register_keras_serializable()
class FastNormalize(keras.layers.Layer):
    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        self.mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        self.std = tf.convert_to_tensor(std, dtype=tf.float32)
        self._mean_list = (
            mean.numpy().tolist() if hasattr(mean, "numpy") else list(mean)
        )
        self._std_list = std.numpy().tolist() if hasattr(std, "numpy") else list(std)

    def call(self, x):
        return tf.math.subtract(x, self.mean) / (self.std + 1e-6)

    def get_config(self):
        return {**super().get_config(), "mean": self._mean_list, "std": self._std_list}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_paths", nargs="+", required=True, help="Paths to .keras model files"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV path for ensemble PR curve",
    )
    args = parser.parse_args()

    model_paths = args.model_paths
    model_names = [os.path.splitext(os.path.basename(p))[0] for p in model_paths]

    # Default output path
    if args.output_csv:
        output_csv = args.output_csv
    else:
        os.makedirs("pr_curves", exist_ok=True)
        joined_name = "_".join(model_names)
        output_csv = os.path.join("pr_curves", f"ensemble_{joined_name}.csv")

    # Load dataset once
    logging.info("Loading test data into memory...")
    dataset = get_dataloader(
        dataloader="tensorflow-tfds",
        data_root=DATA_ROOT,
        years=range(2013, 2023),
        data_type="test",
        batch_size=128,
        weights={"wN": 1.0, "w0": 1.0, "w1": 1.0, "w2": 1.0, "wW": 1.0},
        select_keys=None,  # Lazy load input shape from first model
    )

    # Store batches for reuse
    batches = []
    y_true_all = []
    for batch in dataset:
        inputs, labels, _ = batch
        batches.append(inputs)
        y_true_all.append(labels.numpy())
    y_true = np.concatenate(y_true_all, axis=0).ravel()
    logging.info(f"Cached {len(batches)} batches.")

    # Init score array
    ensemble_sum = np.zeros_like(y_true, dtype=np.float32)
    num_models = 0

    for path in model_paths:
        model_name = os.path.basename(path)
        logging.info(f"Loading model: {model_name}")
        model = keras.models.load_model(
            path,
            compile=False,
            safe_mode=False,
            custom_objects={"<lambda>": lambda x: tf.abs(x - 0.5)},
        )

        batch_preds = []
        for inputs in batches:
            preds = model.predict_on_batch(inputs)
            batch_preds.append(preds)

        y_scores = np.concatenate(batch_preds, axis=0).ravel()
        ensemble_sum += y_scores
        num_models += 1

        del model
        tf.keras.backend.clear_session()

    # Average predictions across models
    ensemble_scores = ensemble_sum / num_models

    # PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, ensemble_scores)
    df = pd.DataFrame(
        {
            "threshold": np.append(thresholds, 1.0),
            "precision": precision,
            "recall": recall,
        }
    )

    df.to_csv(output_csv, index=False)
    logging.info(f"Saved ensemble PR curve to {output_csv}")


if __name__ == "__main__":
    main()
