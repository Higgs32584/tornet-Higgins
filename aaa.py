"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

import os
import argparse
import logging
import numpy as np
import tensorflow as tf
import keras
from tornet.data.loader import get_dataloader
from tornet.metrics.keras import metrics as tfm

import tensorflow_datasets as tfds
import tornet.data.tfds.tornet.tornet_dataset_builder  # registers 'tornet'

logging.basicConfig(level=logging.INFO)

# Hardcoded for TornadoNet setup
DATA_ROOT = "/home/ubuntu/tfds"
os.environ['TFDS_DATA_DIR'] = DATA_ROOT
os.environ['TORNET_ROOT'] = DATA_ROOT

# Custom metric definitions
class FalseAlarmRate(tf.keras.metrics.Metric):
    def __init__(self, name="false_alarm_rate", **kwargs):
        super().__init__(name=name, **kwargs)
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.epsilon = 1e-7

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        tp = tf.reduce_sum(y_true * y_pred)
        self.false_positives.assign_add(fp)
        self.true_positives.assign_add(tp)

    def result(self):
        return self.false_positives / (self.false_positives + self.true_positives + self.epsilon)

    def reset_states(self):
        self.false_positives.assign(0)
        self.true_positives.assign(0)

class ThreatScore(tf.keras.metrics.Metric):
    def __init__(self, name="threat_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")
        self.epsilon = 1e-7

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.fp.assign_add(tf.reduce_sum((1 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1 - y_pred)))

    def result(self):
        return self.tp / (self.tp + self.fp + self.fn + self.epsilon)

    def reset_states(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)
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

class FalseAlarmRate(tf.keras.metrics.Metric):
    def __init__(self, name="false_alarm_rate", **kwargs):
        super().__init__(name=name, **kwargs)
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.epsilon = 1e-7

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Binary predictions
        y_true = tf.cast(y_true, tf.float32)

        fp = tf.reduce_sum((1 - y_true) * y_pred)
        tp = tf.reduce_sum(y_true * y_pred)

        self.false_positives.assign_add(fp)
        self.true_positives.assign_add(tp)

    def result(self):
        return self.false_positives / (self.false_positives + self.true_positives + self.epsilon)

    def reset_states(self):
        self.false_positives.assign(0)
        self.true_positives.assign(0)
@keras.utils.register_keras_serializable()
class FastNormalize(keras.layers.Layer):
    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        self.mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        self.std = tf.convert_to_tensor(std, dtype=tf.float32)
        self._mean_list = mean.numpy().tolist() if hasattr(mean, 'numpy') else list(mean)
        self._std_list = std.numpy().tolist() if hasattr(std, 'numpy') else list(std)

    def call(self, x):
        return tf.math.subtract(x, self.mean) / (self.std + 1e-6)

    def get_config(self):
        config = super().get_config()
        config.update({
            "mean": self._mean_list,
            "std": self._std_list,
        })
        return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", nargs='+', required=True, help="List of .keras model files for soft voting ensemble")
    args = parser.parse_args()

    logging.info(f"Using models: {args.model_paths}")

    # Load all models
    models = [keras.saving.load_model(path, compile=False) for path in args.model_paths]
    input_keys = list(models[0].input.keys())

    # Load test data
    test_years = list(range(2013, 2023))
    ds_test = get_dataloader("tensorflow-tfds", DATA_ROOT, test_years, "test", batch_size=128, select_keys=input_keys)

    # Run predictions and collect ground truth
    y_true_all = []
    y_prob_ensemble = []

    for batch in ds_test:
        x_batch, y_batch = batch
        y_true_all.append(y_batch.numpy())

        # Soft-voting ensemble
        preds = [m(x_batch, training=False).numpy() for m in models]
        avg_preds = np.mean(preds, axis=0)
        y_prob_ensemble.append(avg_preds)

    y_true = np.concatenate(y_true_all).squeeze()
    y_probs = np.concatenate(y_prob_ensemble).squeeze()
    y_pred_bin = (y_probs > 0.5).astype(np.float32)

    # Compute metrics manually
    metric_objects = [
        tf.keras.metrics.AUC(curve='PR', name='AUCPR', num_thresholds=1000),
        tf.keras.metrics.BinaryAccuracy(name='BinaryAccuracy'),
        tf.keras.metrics.Precision(name='Precision'),
        tf.keras.metrics.Recall(name='Recall'),
        tf.keras.metrics.TruePositives(name='TruePositives'),
        tf.keras.metrics.FalsePositives(name='FalsePositives'),
        tf.keras.metrics.TrueNegatives(name='TrueNegatives'),
        tf.keras.metrics.FalseNegatives(name='FalseNegatives'),
        FalseAlarmRate(),
        tfm.F1Score(from_logits=False, name='F1'),
        ThreatScore()
    ]

    for metric in metric_objects:
        metric.update_state(y_true, y_probs)

    # Log results
    logging.info("Soft-Voting Ensemble Metrics:")
    for metric in metric_objects:
        logging.info(f"{metric.name}: {metric.result().numpy():.4f}")

if __name__ == "__main__":
    main()
