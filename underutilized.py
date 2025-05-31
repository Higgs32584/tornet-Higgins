import argparse
import logging
import os

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tornet.data.loader import get_dataloader
from tornet.metrics.keras import metrics as tfm

logging.basicConfig(level=logging.INFO)
from custom_func import FalseAlarmRate, ThreatScore

TFDS_DATA_DIR = "/home/ubuntu/tfds"
EXP_DIR = os.environ.get("EXP_DIR", ".")
TORNET_ROOT = TFDS_DATA_DIR
# TFDS_DATA_DIR=os.environ['TFDS_DATA_DIR']
import tensorflow_datasets as tfds

import tornet.data.tfds.tornet.tornet_dataset_builder  # registers 'tornet'
from tornet.models.keras.layers import CoordConv2D

EXP_DIR = "."
DATA_ROOT = "/home/ubuntu/tfds"
TORNET_ROOT = DATA_ROOT
DATA_ROOT = "/home/ubuntu/tfds"
TFDS_DATA_DIR = "/home/ubuntu/tfds"
tf.config.optimizer.set_jit(True)

os.environ["TORNET_ROOT"] = DATA_ROOT
os.environ["TFDS_DATA_DIR"] = TFDS_DATA_DIR


class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        base_lr,
        warmup_steps,
        restart_steps=5000,
        t_mul=2.0,
        m_mul=1.0,
        alpha=1e-6,
    ):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.alpha = alpha

        self.cosine_decay = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=base_lr,
            first_decay_steps=restart_steps,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=alpha,
        )

    def __call__(self, step):
        warmup_lr = self.base_lr * (
            tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32)
        )
        lr = tf.cond(
            step < self.warmup_steps,
            lambda: warmup_lr,
            lambda: self.cosine_decay(step - self.warmup_steps),
        )
        return lr


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


@keras.utils.register_keras_serializable()
class ExpandDimsTwice(keras.layers.Layer):
    def call(self, inputs):
        return tf.expand_dims(tf.expand_dims(inputs, axis=1), axis=1)


@keras.utils.register_keras_serializable()
class StackAvgMax(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.stack(inputs, axis=1)


def normalize(x, name: str):
    min_val, max_val = CHANNEL_MIN_MAX[name]
    mean = np.float32((max_val + min_val) / 2)
    std = np.float32((max_val - min_val) / 2)
    n_sweeps = x.shape[-1]

    # Use tf.constant directly for faster graph compilation
    mean = tf.constant([mean] * n_sweeps, dtype=tf.float32)
    std = tf.constant([std] * n_sweeps, dtype=tf.float32)

    return FastNormalize(mean, std)(x)


custom_objects = {
    "FillNaNs": FillNaNs,
    "FastNormalize": FastNormalize,
    "FalseAlarmRate": FalseAlarmRate,
    "ThreatScore": ThreatScore,
}


def check_dead_weights(model):
    """
    Analyzes Keras model weights to identify potential "dead" weights.

    Args:
        model: A trained Keras model.
    """

    for layer_idx, layer in enumerate(model.layers):
        if hasattr(layer, "get_weights"):  # Only layers with weights
            weights = layer.get_weights()

            for weight_idx, weight_matrix in enumerate(weights):

                # Check for weights that are all zeros (or very close to zero)
                if np.all(
                    np.abs(weight_matrix) < 1e-6
                ):  # Threshold for "close to zero"
                    print(
                        "    Warning: All weights are zero or very close to zero - may be dead."
                    )
                else:
                    # Check for weights with very small variance
                    variance = np.var(weight_matrix)
                    if variance < 1e-6:  # Threshold for "small variance"
                        print(f"Layer {layer_idx} ({layer.name}):")
                        print(
                            "   Warning: Weights have very small variance - may be dead."
                        )


model = tf.keras.models.load_model(
    "/home/ubuntu/tornet-Higgins/ensemble_of_v7TH3899/fold5seed99TH4242.keras",
    compile=False,
    safe_mode=False,
    custom_objects={},
)

check_dead_weights(model)
