import argparse
import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
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
        config = super().get_config()
        config.update(
            {
                "mean": self._mean_list,
                "std": self._std_list,
            }
        )
        return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the .keras model file"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        help="Optional: output CSV path for precision-recall data",
    )
    args = parser.parse_args()

    model_path = args.model_path

    # Derive default output CSV path if not provided
    if args.output_csv:
        output_csv = args.output_csv
    else:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_dir = "pr_curves"
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(output_dir, f"{model_name}.csv")

    # Load model
    logging.info(f"Loading model from {model_path}")
    model = keras.models.load_model(
        model_path,
        compile=False,
        safe_mode=False,
        custom_objects={"<lambda>": lambda x: tf.abs(x - 0.5)},
    )

    # Load test data
    dataset = get_dataloader(
        dataloader="tensorflow-tfds",
        data_root=DATA_ROOT,
        years=range(2013, 2023),
        data_type="test",
        batch_size=128,
        weights={"wN": 1.0, "w0": 1.0, "w1": 1.0, "w2": 1.0, "wW": 1.0},
        select_keys=list(model.input.keys()),
    )

    y_true_all = []
    y_score_all = []

    logging.info("Running inference...")
    import tqdm

    for batch in tqdm.tqdm(dataset):
        inputs, labels, _ = batch
        preds = model.predict_on_batch(inputs)
        y_true_all.append(labels.numpy())
        y_score_all.append(preds)

    y_true = np.concatenate(y_true_all, axis=0).ravel()
    y_scores = np.concatenate(y_score_all, axis=0).ravel()

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    df = pd.DataFrame(
        {
            "threshold": np.append(
                thresholds, 1.0
            ),  # Ensure matching shape with precision/recall
            "precision": precision,
            "recall": recall,
        }
    )

    df.to_csv(output_csv, index=False)
    logging.info(f"Saved PR curve data to {output_csv}")


if __name__ == "__main__":
    main()
