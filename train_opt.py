import json
import logging
import os
import shutil
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts

from custom_func import FalseAlarmRate, ThreatScore
from tornet.data.constants import ALL_VARIABLES, CHANNEL_MIN_MAX
from tornet.data.loader import get_dataloader
from tornet.data.preprocess import get_shape
from tornet.metrics.keras import metrics as tfm
from tornet.utils.general import make_exp_dir

# Mixed Precision
mixed_precision.set_global_policy("mixed_float16")

# Logging and seed setup
logging.basicConfig(level=logging.ERROR)
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random = np.random
random.seed(SEED)
tf.random.set_seed(SEED)

# Env vars
DATA_ROOT = "/home/ubuntu/tfds"
os.environ["TORNET_ROOT"] = DATA_ROOT
os.environ["TFDS_DATA_DIR"] = DATA_ROOT
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"


# Helper layers
class FastNormalize(keras.layers.Layer):
    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        self.mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        self.std = tf.convert_to_tensor(std, dtype=tf.float32)

    def call(self, x):
        return (x - self.mean) / (self.std + 1e-6)


def normalize(x, name):
    min_val, max_val = CHANNEL_MIN_MAX[name]
    mean = np.float32((max_val + min_val) / 2)
    std = np.float32((max_val - min_val) / 2)
    mean = tf.constant([mean] * x.shape[-1], dtype=tf.float32)
    std = tf.constant([std] * x.shape[-1], dtype=tf.float32)
    return FastNormalize(mean, std)(x)


class CoordConv2D(keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        kernel_regularizer,
        activation,
        padding="same",
        strides=(1, 1),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.conv = Conv2D(
            filters,
            kernel_size,
            padding=padding,
            strides=strides,
            kernel_regularizer=kernel_regularizer,
            activation=activation,
            use_bias=False,
        )

    def call(self, inputs):
        x, coords = inputs
        x = tf.concat([x, coords], axis=-1)
        return self.conv(x), coords


class FillNaNs(keras.layers.Layer):
    def __init__(self, fill_val, **kwargs):
        super().__init__(**kwargs)
        self.fill_val = tf.constant(fill_val, dtype=tf.float32)

    def call(self, x):
        return tf.where(tf.math.is_nan(x), self.fill_val, x)


# Model blocks
def se_block(x, ratio=16):
    filters = x.shape[-1]
    se = GlobalAveragePooling2D()(x)
    se = Dense(filters // ratio, activation="relu")(se)
    se = Dense(filters, activation="sigmoid")(se)
    se = Reshape((1, 1, filters))(se)
    return Multiply()([x, se])


def wide_resnet_block(
    x, c, filters, widen_factor=2, l2_reg=1e-6, drop_rate=0.0, nconvs=2
):
    shortcut_x, shortcut_c = x, c
    for _ in range(nconvs):
        x, c = CoordConv2D(
            filters * widen_factor, 3, keras.regularizers.l2(l2_reg), activation=None
        )([x, c])
        x = BatchNormalization()(x)
    shortcut_x, _ = CoordConv2D(
        filters * widen_factor, 1, keras.regularizers.l2(l2_reg), activation=None
    )([shortcut_x, shortcut_c])
    x = Activation("relu")(x)
    x = Add()([x, shortcut_x])
    x = MaxPool2D(2, 2, padding="same")(x)
    c = MaxPool2D(2, 2, padding="same")(c)
    if drop_rate > 0:
        x = Dropout(drop_rate)(x)
    return x, c


def build_model(
    model="wide_resnet",
    shape=(120, 240, 2),
    c_shape=(120, 240, 2),
    input_variables=ALL_VARIABLES,
    start_filters=64,
    l2_reg=1e-3,
    background_flag=-3.0,
    include_range_folded=True,
    dropout_rate=0.1,
):
    inputs = {v: keras.Input(shape, name=v) for v in input_variables}
    n_sweeps = shape[2]
    norm_inputs = [normalize(inputs[v], v) for v in input_variables]
    x = Concatenate(axis=-1)(norm_inputs)
    x = FillNaNs(background_flag)(x)
    if include_range_folded:
        rf = keras.Input(shape[:2] + (n_sweeps,), name="range_folded_mask")
        x = Concatenate(axis=-1)([x, rf])
        inputs["range_folded_mask"] = rf
    cin = keras.Input(c_shape, name="coordinates")
    inputs["coordinates"] = cin
    x, c = x, cin
    if model == "wide_resnet":
        for i in range(3):
            x, c = wide_resnet_block(
                x, c, start_filters * (2**i), l2_reg=l2_reg, drop_rate=dropout_rate
            )
        x = se_block(x)
    x = Conv2D(128, 3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    attn = Conv2D(1, 1, activation="sigmoid")(x)
    attn = Dropout(0.2)(attn)
    x = Multiply()([x, attn])
    x = Concatenate()([GlobalAveragePooling2D()(x), GlobalMaxPooling2D()(x)])
    x = Dense(64, activation="relu")(x)
    output = Dense(1, activation="sigmoid", dtype="float32")(x)
    return keras.Model(inputs=inputs, outputs=output)


# Main training
DEFAULT_CONFIG = {
    "epochs": 100,
    "input_variables": ["DBZ", "VEL", "KDP", "ZDR", "RHOHV", "WIDTH"],
    "train_years": list(range(2013, 2021)),
    "val_years": [2021, 2022],
    "batch_size": 128,
    "model": "wide_resnet",
    "start_filters": 48,
    "learning_rate": 0.0001596,
    "first_decay_steps": 2221,
    "weight_decay": 2.63e-7,
    "dropout_rate": 0.122,
    "l2_reg": 1.59e-7,
    "wN": 0.05069,
    "w0": 0.55924,
    "w1": 3.49593,
    "w2": 11.26479,
    "wW": 0.20101,
    "label_smooth": 0.1,
    "loss": "cce",
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
            "range_folded_mask",
            "coordinates",
        ]
    },
}


def main(config):
    ds_train = (
        get_dataloader(
            config["dataloader"],
            DATA_ROOT,
            config["train_years"],
            "train",
            config["batch_size"],
            {
                "wN": config["wN"],
                "w0": config["w0"],
                "w1": config["w1"],
                "w2": config["w2"],
                "wW": config["wW"],
            },
            **config["dataloader_kwargs"]
        )
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    ds_val = get_dataloader(
        config["dataloader"],
        DATA_ROOT,
        config["val_years"],
        "train",
        config["batch_size"],
        {"wN": 1, "w0": 1, "w1": 1, "w2": 1, "wW": 1},
        **config["dataloader_kwargs"]
    )
    ds_val = ds_val.cache().prefetch(tf.data.AUTOTUNE)

    x, _, _ = next(iter(ds_train))
    in_shape = (120, 240, get_shape(x)[-1])
    c_shape = (120, 240, x["coordinates"].shape[-1])
    model = build_model(
        model=config["model"],
        shape=in_shape,
        c_shape=c_shape,
        input_variables=config["input_variables"],
        start_filters=config["start_filters"],
        l2_reg=config["l2_reg"],
        dropout_rate=config["dropout_rate"],
    )

    loss = lambda y_true, y_pred: (
        0.7 * tversky_loss(0.5, 0.5)(y_true, y_pred)
        + 0.3 * focal_loss(2.0, 0.85)(y_true, y_pred)
    )

    lr_schedule = CosineDecayRestarts(
        config["learning_rate"], config["first_decay_steps"], t_mul=2.0, m_mul=0.9
    )

    opt = AdamW(
        learning_rate=lr_schedule,
        weight_decay=config["weight_decay"],
        beta_1=0.9,
        beta_2=0.999,
    )

    metrics = [
        keras.metrics.AUC(
            from_logits=False, curve="PR", name="AUCPR", num_thresholds=500
        ),
        tfm.F1Score(from_logits=False, name="F1"),
        tfm.BinaryAccuracy(from_logits=False),
        tfm.Precision(from_logits=False),
        tfm.Recall(from_logits=False),
        FalseAlarmRate(name="FalseAlarmRate"),
        ThreatScore(name="ThreatScore"),
    ]

    model.compile(optimizer=opt, loss=loss, metrics=metrics, jit_compile=True)

    expdir = make_exp_dir(exp_dir=config["exp_dir"], prefix=config["exp_name"])
    with open(os.path.join(expdir, "params.json"), "w") as f:
        json.dump({"config": config}, f)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(expdir, "model_{epoch:03d}.keras"), monitor="val_AUCPR"
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_AUCPR", patience=5, mode="max", restore_best_weights=True
        ),
        keras.callbacks.CSVLogger(os.path.join(expdir, "history.csv")),
        keras.callbacks.TerminateOnNaN(),
    ]

    if keras.config.backend() == "tensorflow":
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(expdir, "logs"), write_graph=False
            )
        )

    model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=config["epochs"],
        callbacks=callbacks,
        verbose=1,
    )


if __name__ == "__main__":
    config = DEFAULT_CONFIG
    if len(sys.argv) > 1:
        config.update(json.load(open(sys.argv[1], "r")))
    main(config)
