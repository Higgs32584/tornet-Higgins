import argparse
import json
import logging
import os
import random
import sys

import keras
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

import tornet.data.tfds.tornet.tornet_dataset_builder
from custom_func import FalseAlarmRate, ThreatScore
from tornet.data.constants import CHANNEL_MIN_MAX
from tornet.data.loader import get_dataloader
from tornet.metrics.keras import metrics as tfm

logging.basicConfig(level=logging.INFO)

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_ROOT = "/home/ubuntu/tfds"
os.environ["TORNET_ROOT"] = DATA_ROOT
os.environ["TFDS_DATA_DIR"] = DATA_ROOT
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"


# Custom Layers
@keras.utils.register_keras_serializable()
class FillNaNs(tf.keras.layers.Layer):
    def __init__(self, fill_val, **kwargs):
        super().__init__(**kwargs)
        self.fill_val = tf.convert_to_tensor(fill_val, dtype=tf.float32)

    def call(self, x):
        return tf.where(tf.math.is_nan(x), self.fill_val, x)

    def get_config(self):
        return {**super().get_config(), "fill_val": float(self.fill_val.numpy())}


@keras.utils.register_keras_serializable()
class ExpandDimsTwice(keras.layers.Layer):
    def call(self, inputs):
        return tf.expand_dims(tf.expand_dims(inputs, axis=1), axis=1)


@keras.utils.register_keras_serializable()
class StackAvgMax(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.stack(inputs, axis=1)


@keras.utils.register_keras_serializable()
class FastNormalize(tf.keras.layers.Layer):
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
        return {
            **super().get_config(),
            "mean": self._mean_list,
            "std": self._std_list,
        }


# Custom Threat Score
def threat_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fn + fp + 1e-6)


DEFAULT_CONFIG = {
    "epochs": 100,
    "input_variables": ["DBZ", "VEL", "KDP", "ZDR", "RHOHV", "WIDTH"],
    "train_years": [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "val_years": [2021, 2022],
    "batch_size": 128,
    "dataloader": "tensorflow-tfds",
    "dataloader_kwargs": {
        "select_keys": [
            "DBZ",
            "VEL",
            "KDP",
            "RHOHV",
            "ZDR",
            "WIDTH",
            "range_folded_mask",
            "coordinates",
        ]
    },
    "weights": {"wN": 1.0, "w0": 1.0, "w1": 1.0, "w2": 1.0, "wW": 1.0},
}


def main(config, model_paths):
    dataloader = config["dataloader"]
    input_variables = config["input_variables"]
    val_years = config["val_years"]
    batch_size = config["batch_size"]
    dataloader_kwargs = config["dataloader_kwargs"]
    dataloader_kwargs.update(
        {"select_keys": input_variables + ["range_folded_mask", "coordinates"]}
    )

    ds_val = get_dataloader(
        dataloader,
        DATA_ROOT,
        val_years,
        "train",
        batch_size,
        config["weights"],
        **dataloader_kwargs,
    )

    # Load models
    models = []
    for path in model_paths:
        model = tf.keras.models.load_model(path, compile=False, safe_mode=False)
        models.append(model)
        logging.info(f"Loaded model from {path}")

    thresholds = np.linspace(0, 1, 100)
    y_true = []
    y_pred_probs = []

    # Collect predictions from all models
    for x_batch, y_batch, _ in tqdm.tqdm(ds_val, desc="Predicting"):
        y_true.append(y_batch)
        preds = [model.predict(x_batch, verbose=0) for model in models]
        avg_pred = np.mean(preds, axis=0)
        y_pred_probs.append(avg_pred)

    y_true = np.concatenate(y_true).squeeze()
    y_pred_probs = np.concatenate(y_pred_probs).squeeze()

    # Evaluate metrics across thresholds
    precision, recall, f1, ts = [], [], [], []

    for t in tqdm.tqdm(thresholds, desc="Evaluating thresholds"):
        y_pred = (y_pred_probs >= t).astype(int)
        precision.append(precision_score(y_true, y_pred, zero_division=0))
        recall.append(recall_score(y_true, y_pred, zero_division=0))
        f1.append(f1_score(y_true, y_pred, zero_division=0))
        ts.append(threat_score(y_true, y_pred))

    best_f1_idx = np.argmax(f1)
    best_ts_idx = np.argmax(ts)

    print(
        f"Best F1-Score Threshold: {thresholds[best_f1_idx]:.4f} (F1: {f1[best_f1_idx]:.4f})"
    )
    print(
        f"Best Threat Score Threshold: {thresholds[best_ts_idx]:.4f} (TS: {ts[best_ts_idx]:.4f})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_paths",
        type=str,
        nargs="+",
        required=True,
        help="List of .keras model paths",
    )
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    main(config, args.model_paths)
