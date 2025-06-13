import argparse
import logging
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import precision_recall_curve
import tqdm

from tornet.data.loader import get_dataloader
from tornet.metrics.keras import metrics as tfm
from custom_func import FalseAlarmRate, ThreatScore
from tornet.models.keras.layers import CoordConv2D
import tensorflow_datasets as tfds
import tornet.data.tfds.tornet.tornet_dataset_builder  # registers 'tornet'

# Set environment variables
TFDS_DATA_DIR = "/home/ubuntu/tfds"
DATA_ROOT = TFDS_DATA_DIR
os.environ["TFDS_DATA_DIR"] = TFDS_DATA_DIR
os.environ["TORNET_ROOT"] = DATA_ROOT
tf.config.optimizer.set_jit(True)

logging.basicConfig(level=logging.INFO)


@keras.utils.register_keras_serializable()
class FillNaNs(keras.layers.Layer):
    def __init__(self, fill_val, **kwargs):
        super().__init__(**kwargs)
        self.fill_val = tf.convert_to_tensor(fill_val, dtype=tf.float32)

    @tf.function(jit_compile=True)
    def call(self, x):
        return tf.where(tf.math.is_nan(x), self.fill_val, x)

    def get_config(self):
        return {**super().get_config(), "fill_val": self.fill_val.numpy().item()}


@keras.utils.register_keras_serializable()
class SelectAttentionBranch(tf.keras.layers.Layer):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, x):
        # x has shape (batch, num_branches)
        return tf.expand_dims(x[:, self.index], axis=-1)  # shape: (batch, 1)


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


@keras.utils.register_keras_serializable()
class ExpandDimsTwice(keras.layers.Layer):
    def call(self, inputs):
        return tf.expand_dims(tf.expand_dims(inputs, axis=1), axis=1)


@keras.utils.register_keras_serializable()
class StackAvgMax(keras.layers.Layer):
    def call(self, inputs):
        return tf.stack(inputs, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing .keras model files",
    )
    parser.add_argument(
        "--output_csv", type=str, default=None, help="Output CSV path for PR curve"
    )
    parser.add_argument(
        "--dataloader",
        help="Which data loader to use for loading test data",
        default="tensorflow-tfds",
        choices=["keras", "tensorflow", "tensorflow-tfds", "torch", "torch-tfds"],
    )
    args = parser.parse_args()

    # Load model paths from directory
    model_dir = args.model_dir
    model_paths = sorted(
        [
            os.path.join(model_dir, f)
            for f in os.listdir(model_dir)
            if f.endswith(".keras")
        ]
    )
    if not model_paths:
        raise RuntimeError(f"No .keras model files found in {model_dir}")

    model_names = [os.path.splitext(os.path.basename(p))[0] for p in model_paths]

    if args.output_csv:
        output_csv = args.output_csv
    else:
        os.makedirs("pr_curves", exist_ok=True)
        joined_name = "_".join(model_names)
        output_csv = os.path.join("pr_curves", f"ensemble_{joined_name}.csv")

    logging.info(f"Found {len(model_paths)} model files in {model_dir}")
    logging.info("Loading models...")
    models = []
    for path in model_paths:
        model = tf.keras.models.load_model(
            path,
            safe_mode=False,
            compile=False,
        )
        models.append(model)
        logging.info(f"Loaded model from: {path}")

    # (Rest of your script continues unchanged here — load data, run predictions, save PR curve)

    # Set up data loader
    test_years = range(2013, 2023)
    ds_test = get_dataloader(
        args.dataloader,
        DATA_ROOT,
        random_state=42,
        years=test_years,
        data_type="test",
        batch_size=128,
        weights={"wN": 1.0, "w0": 1.0, "w1": 1.0, "w2": 1.0, "wW": 1.0},
        select_keys=list(models[0].input.keys()),
    )

    y_true_all = []
    y_score_all = []
    predict_fns = [
        tf.function(m.predict_on_batch, reduce_retracing=True) for m in models
    ]

    for batch in tqdm.tqdm(ds_test):
        inputs, labels, _ = batch
        preds = [model.predict_on_batch(inputs) for model in models]
        preds_stack = tf.stack(preds, axis=0)
        ensemble_preds = tf.reduce_mean(preds_stack, axis=0)
        y_true_all.append(labels.numpy())
        y_score_all.append(ensemble_preds)

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
