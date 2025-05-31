import argparse
import logging
import os

import tensorflow as tf
from tensorflow import keras

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


model_path = "/home/ubuntu/tornet-Higgins/ensemble_of_v7TH3899/fold5seed99TH4242.keras"

# Load the model
model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)

# Define the output file
summary_file = "model_summary.txt"

# Write model summary to file
with open(summary_file, "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

print(f"Model summary saved to {summary_file}")
