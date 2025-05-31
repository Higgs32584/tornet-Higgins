import json
import logging
import os
import random
import shutil
import sys
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras.layers import (
    GlobalMaxPooling2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Conv2D,
    Activation,
    Add,
    GlobalAveragePooling2D,
    Dense,
    LayerNormalization,
    Dropout,
    DepthwiseConv2D,
)
from tensorflow.keras.regularizers import l2
import tornet.data.tfds.tornet.tornet_dataset_builder
from custom_func import FalseAlarmRate, ThreatScore
from tornet.data import preprocess as pp
from tornet.data.constants import ALL_VARIABLES, CHANNEL_MIN_MAX
from tornet.data.loader import get_dataloader
from tornet.data.preprocess import get_shape
from tornet.metrics.keras import metrics as tfm

# from tornet.models.keras.layers import CoordConv2D
from tornet.utils.general import make_exp_dir

logging.basicConfig(level=logging.ERROR)
SEED = 98
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
DATA_ROOT = "/home/ubuntu/tfds"
TORNET_ROOT = DATA_ROOT
TFDS_DATA_DIR = DATA_ROOT
EXP_DIR = "."
os.environ["TORNET_ROOT"] = DATA_ROOT
os.environ["TFDS_DATA_DIR"] = TFDS_DATA_DIR
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
tf.config.optimizer.set_jit(True)

logging.info(f"TORNET_ROOT={DATA_ROOT}")


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


import tensorflow as tf
import numpy as np


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


def convnext_block(x, drop_rate=0.0, layer_scale_init_value=1e-6, name=None):
    input_channels = x.shape[-1]
    shortcut = x

    x = DepthwiseConv2D(kernel_size=7, padding="same", name=f"{name}_dwconv")(x)
    x = LayerNormalization(epsilon=1e-6, name=f"{name}_ln")(x)
    x = Dense(4 * input_channels, name=f"{name}_pw1")(x)
    x = Activation("gelu", name=f"{name}_gelu")(x)
    x = Dense(input_channels, name=f"{name}_pw2")(x)

    # Layer scale (if enabled)
    if layer_scale_init_value > 0:
        gamma = tf.Variable(
            initial_value=layer_scale_init_value
            * tf.ones((input_channels,), dtype=tf.float32),
            trainable=True,
            name=f"{name}_gamma",
        )
        x = x * gamma

    # Dropout (optional)
    if drop_rate > 0:
        x = Dropout(drop_rate, name=f"{name}_drop")(x)

    # Residual connection
    x = Add(name=f"{name}_add")([shortcut, x])
    return x


def build_model(
    shape: Tuple[int] = (120, 240, 2),
    c_shape: Tuple[int] = (120, 240, 2),
    input_variables: List[str] = ALL_VARIABLES,
    epsilon_layer: float = 1e-6,
    start_filters: int = 64,
    dropout_rate: float = 0.1,
) -> keras.Model:
    background_flag = -3.0
    include_range_folded = True

    # Inputs per variable
    inputs = {v: keras.Input(shape, name=v) for v in input_variables}
    n_sweeps = shape[2]
    x = keras.layers.Concatenate(axis=-1)(
        [normalize(inputs[v], v) for v in input_variables]
    )
    x = FillNaNs(background_flag)(x)
    if include_range_folded:
        range_folded = keras.Input(shape[:2] + (n_sweeps,), name="range_folded_mask")
        inputs["range_folded_mask"] = range_folded
        x = keras.layers.Concatenate(axis=-1)([x, range_folded])
    coords = keras.Input(c_shape, name="coordinates")
    inputs["coordinates"] = coords
    x = keras.layers.Concatenate(axis=-1)([x, coords])
    x = Conv2D(96, kernel_size=4, strides=4, name="stem_conv")(x)
    x = LayerNormalization(epsilon=epsilon_layer, name="stem_ln")(x)
    x = convnext_block(
        x, drop_rate=dropout_rate, layer_scale_init_value=1e-6, name="convnext_block"
    )
    x_avg = GlobalAveragePooling2D()(x)
    x_max = GlobalMaxPooling2D()(x)
    x = keras.layers.Concatenate()([x_avg, x_max])
    x = LayerNormalization(epsilon=epsilon_layer, name="head_ln")(x)
    x = Dense(128, activation="gelu", name="head_dense")(x)
    output = Dense(1, activation="sigmoid", name="pred")(x)
    return keras.Model(inputs=inputs, outputs=output)


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


def normalize(x, name: str):
    min_val, max_val = CHANNEL_MIN_MAX[name]
    mean = np.float32((max_val + min_val) / 2)
    std = np.float32((max_val - min_val) / 2)
    n_sweeps = x.shape[-1]

    mean = tf.constant([mean] * n_sweeps, dtype=tf.float32)
    std = tf.constant([std] * n_sweeps, dtype=tf.float32)

    return FastNormalize(mean, std)(x)


gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


DEFAULT_CONFIG = {
    "epochs": 100,
    "input_variables": ["DBZ", "VEL", "KDP", "ZDR", "RHOHV", "WIDTH"],
    "train_years": [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "val_years": [2021, 2022],
    "batch_size": 128,
    "start_filters": 48,
    "dropout_rate": 0.15,
    "warmup_epochs": 5,
    "learning_rate": 3e-4,
    "label_smoothing": 0.1,
    "t_mul": 2.0,
    "m_mul": 0.9,
    "wN": 1.0,
    "w0": 1.0,
    "w1": 1.0,
    "w2": 2.0,
    "wW": 0.5,
    "exp_name": "tornado_baseline",
    "exp_dir": ".",
    "dataloader": "tensorflow-tfds",
    "dataloader_kwargs": {
        "select_keys": [
            "DBZ",
            "VEL",
            "KDP",
            "RHOHV",
            "ZDR",
            "WIDTH",
            "coordinates",
            "range_folded_mask",
        ]
    },
}


def main(config):
    epochs = config.get("epochs")
    batch_size = config.get("batch_size")
    lr = config.get("learning_rate")
    wN = config.get("wN")
    w0 = config.get("w0")
    w1 = config.get("w1")
    w2 = config.get("w2")
    wW = config.get("wW")
    train_years = config.get("train_years")
    val_years = config.get("val_years")
    dataloader = config.get("dataloader")
    dataloader_kwargs = config.get("dataloader_kwargs")

    logging.info(f"Using {dataloader} dataloader")
    logging.info(f"Running with config: {config}")
    weights = {"wN": wN, "w0": w0, "w1": w1, "w2": w2, "wW": wW}
    # Apply to Train and Validation Data
    ds_train = get_dataloader(
        dataloader,
        DATA_ROOT,
        train_years,
        "train",
        batch_size,
        weights,
        **dataloader_kwargs,
    )
    ds_val = get_dataloader(
        dataloader,
        DATA_ROOT,
        val_years,
        "train",
        batch_size,
        {"wN": 1.0, "w0": 1.0, "w1": 1.0, "w2": 1.0, "wW": 1.0},
        **dataloader_kwargs,
    )
    x, _, _ = next(iter(ds_train))
    in_shapes = (None, None, get_shape(x)[-1])
    c_shapes = (None, None, x["coordinates"].shape[-1])

    in_shapes = (120, 240, get_shape(x)[-1])
    nn = build_model(
        shape=in_shapes,
        c_shape=c_shapes,
        input_variables=config["input_variables"],
        start_filters=config["start_filters"],
        dropout_rate=config["dropout_rate"],
    )

    print(nn.summary())
    from_logits = False
    steps_per_epoch = len(ds_train)

    lr = WarmUpCosine(
        base_lr=config["learning_rate"],
        warmup_steps=config["warmup_epochs"] * steps_per_epoch,
        restart_steps=10 * steps_per_epoch,  # first restart at epoch 10
        t_mul=config["t_mul"],
        m_mul=config["m_mul"],
        alpha=1e-6,
    )

    loss = keras.losses.BinaryCrossentropy(
        from_logits=from_logits, label_smoothing=config["label_smoothing"]
    )
    opt = keras.optimizers.Adam(learning_rate=lr)

    # Metrics (Optimize AUCPR)
    metrics = [
        keras.metrics.AUC(
            from_logits=from_logits, curve="PR", name="AUCPR", num_thresholds=2000
        ),
        keras.metrics.AUC(from_logits=from_logits, name="AUC", num_thresholds=2000),
        tfm.BinaryAccuracy(from_logits, name="BinaryAccuracy"),
        tfm.TruePositives(from_logits, name="TruePositives"),
        tfm.FalsePositives(from_logits, name="FalsePositives"),
        tfm.TrueNegatives(from_logits, name="TrueNegatives"),
        tfm.FalseNegatives(from_logits, name="FalseNegatives"),
        tfm.Precision(from_logits, name="Precision"),
        tfm.Recall(from_logits, name="Recall"),
        tfm.F1Score(from_logits=from_logits, name="F1"),
    ]

    nn.compile(loss=loss, metrics=metrics, optimizer=opt, jit_compile=True)
    callbacks = [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.EarlyStopping(
            monitor="val_AUCPR", patience=5, mode="max", restore_best_weights=True
        ),
    ]

    history = nn.fit(
        ds_train, epochs=epochs, validation_data=ds_val, callbacks=callbacks, verbose=1
    )

    # Get Best Metrics
    best_aucpr = max(history.history.get("val_AUCPR", [0]))
    return {"AUCPR": best_aucpr}


if __name__ == "__main__":
    config = DEFAULT_CONFIG
    if len(sys.argv) > 1:
        config.update(json.load(open(sys.argv[1], "r")))

    # Define CV folds
    folds = [
        {
            "train_years": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
            "val_years": [2013, 2014],
        },
        {
            "train_years": [2013, 2014, 2017, 2018, 2019, 2020, 2021, 2022],
            "val_years": [2015, 2016],
        },
        {
            "train_years": [2013, 2014, 2015, 2016, 2019, 2020, 2021, 2022],
            "val_years": [2017, 2018],
        },
        {
            "train_years": [2013, 2014, 2015, 2016, 2017, 2018, 2021, 2022],
            "val_years": [2019, 2020],
        },
        {
            "train_years": [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
            "val_years": [2021, 2022],
        },
    ]

    results = []

    for i, fold in enumerate(folds):
        print(
            f"\n================ Fold {i + 1}: Train {fold['train_years']} | Test {fold['val_years']} ================\n"
        )
        fold_config = config.copy()
        fold_config["train_years"] = fold["train_years"]
        fold_config["val_years"] = fold["val_years"]
        fold_config["exp_name"] = f"{config['exp_name']}_fold{i+1}"
        fold_result = main(fold_config)
        results.append({"fold": i + 1, **fold_result})

    # Print Summary
    print("\n================ Cross-Validation Summary ================\n")
    for res in results:
        print(f"Fold {res['fold']} - AUCPR: {res['AUCPR']:.4f}")
    mean_aucpr = np.mean([res["AUCPR"] for res in results])
    print(f"\nMean AUCPR over all folds: {mean_aucpr:.4f}")
