import json
import logging
import os
import random
import sys
from datetime import datetime
from typing import List, Tuple

import numpy as np
import optuna

# Loss Function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Lambda,
    MaxPool2D,
    Multiply,
    ReLU,
    Reshape,
    UpSampling2D,
)
from tensorflow.keras.optimizers import Adam, AdamW

from tornet.data.constants import ALL_VARIABLES, CHANNEL_MIN_MAX
from tornet.data.loader import get_dataloader
from tornet.data.preprocess import get_shape
from tornet.metrics.keras import metrics as tfm
from tornet.models.keras.cnn_baseline import normalize, vgg_block
from tornet.models.keras.layers import CoordConv2D, FillNaNs
from tornet.models.keras.losses import mae_loss
from tornet.utils.general import make_exp_dir

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
# Environment Variables
DATA_ROOT = "/home/ubuntu/tfds"
TFDS_DATA_DIR = DATA_ROOT
os.environ["TORNET_ROOT"] = DATA_ROOT
os.environ["TFDS_DATA_DIR"] = TFDS_DATA_DIR
# os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

from tensorflow.keras.optimizers.schedules import CosineDecayRestarts

DEFAULT_CONFIG = {
    "epochs": 5,
    "input_variables": ["DBZ", "VEL", "KDP", "ZDR", "RHOHV", "WIDTH"],
    "train_years": [2013],
    "val_years": [2014],
    "batch_size": 128,
    "model": "wide_resnet",
    "start_filters": 72,
    "learning_rate": 1e-1,
    "decay_steps": 6114,
    "decay_rate": 0.958,
    "dropout_rate": 0.0,
    "l2_reg": 0.0,
    "wN": 0.05069,
    "w0": 0.55924,
    "w1": 3.49593,
    "w2": 11.26479,
    "wW": 0.20101,
    "label_smooth": 0.1,
    "loss": "cce",
    "head": "custom_2",
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

from tensorflow.keras.optimizers import AdamW

DATA_ROOT = "/home/ubuntu/tfds"
TFDS_DATA_DIR = DATA_ROOT
os.environ["TORNET_ROOT"] = DATA_ROOT
os.environ["TFDS_DATA_DIR"] = TFDS_DATA_DIR


import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Lambda,
    MaxPool2D,
    Multiply,
    UpSampling2D,
)
from tensorflow.keras.optimizers import SGD


def build_model(
    model="wide_resnet",
    shape: Tuple[int] = (120, 240, 2),
    c_shape: Tuple[int] = (120, 240, 2),
    input_variables: List[str] = ALL_VARIABLES,
    start_filters: int = 64,
    l2_reg: float = 0.001,
    background_flag: float = -3.0,
    include_range_folded: bool = True,
    dropout_rate=0.1,
    head="custom_1",
):
    # Create input layers for each input_variables
    inputs = {}
    for v in input_variables:
        inputs[v] = keras.Input(shape, name=v)
    n_sweeps = shape[2]

    # Normalize inputs and concate along channel dim
    normalized_inputs = keras.layers.Concatenate(axis=-1, name="Concatenate1")(
        [normalize(inputs[v], v) for v in input_variables]
    )

    # Replace nan pixel with background flag
    normalized_inputs = FillNaNs(background_flag)(normalized_inputs)

    # Add channel for range folded gates
    if include_range_folded:
        range_folded = keras.Input(shape[:2] + (n_sweeps,), name="range_folded_mask")
        inputs["range_folded_mask"] = range_folded
        normalized_inputs = keras.layers.Concatenate(axis=-1, name="Concatenate2")(
            [normalized_inputs, range_folded]
        )

    # Input coordinate information
    cin = keras.Input(c_shape, name="coordinates")
    inputs["coordinates"] = cin

    x, c = normalized_inputs, cin

    if model == "wide_resnet":
        x, c = wide_resnet_block(
            x,
            c,
            filters=start_filters,
            widen_factor=2,
            l2_reg=l2_reg,
            nconvs=3,
            drop_rate=dropout_rate,
        )
        x, c = wide_resnet_block(
            x,
            c,
            filters=start_filters * 2,
            widen_factor=2,
            l2_reg=l2_reg,
            nconvs=3,
            drop_rate=dropout_rate,
        )
        x, c = wide_resnet_block(
            x,
            c,
            filters=start_filters * 4,
            widen_factor=2,
            l2_reg=l2_reg,
            nconvs=3,
            drop_rate=dropout_rate,
        )
        x = se_block(x)
    if head == "custom_1":
        x = Conv2D(128, 3, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = cbam_block(x, name="cbam")
        x = Dropout(rate=0.2)(x)  # Optional: still apply dropout after attention
    if head == "custom_2":
        x = Conv2D(128, 3, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        attention_map = Conv2D(
            1, 1, activation="sigmoid", name="attention_map", use_bias=False
        )(x)
        attention_map = Dropout(rate=0.2, name="attention_dropout")(attention_map)
        x = Multiply()([x, attention_map])
    x_avg = GlobalAveragePooling2D()(x_weighted)
    x_max = GlobalMaxPooling2D()(x_weighted)
    x_concat = keras.layers.Concatenate()([x_avg, x_max])
    x_concat = keras.layers.Concatenate()([x_avg, x_max])
    x_dense = Dense(64, activation="relu")(x_concat)
    output = Dense(1, activation="sigmoid", dtype="float32")(x_dense)
    return keras.Model(inputs=inputs, outputs=output)


def se_block(x, ratio=16, name=None):
    filters = x.shape[-1]

    # Squeeze
    se = GlobalAveragePooling2D(name=f"{name}_gap" if name else None)(x)

    # Excite
    se = Dense(
        filters // ratio, activation="relu", name=f"{name}_fc1" if name else None
    )(se)
    se = Dense(filters, activation="sigmoid", name=f"{name}_fc2" if name else None)(se)

    # Explicit broadcast shape for Multiply
    se = Reshape((1, 1, filters), name=f"{name}_reshape" if name else None)(se)

    # Multiply
    x = Multiply(name=f"{name}_scale" if name else None)([x, se])
    return x


def cbam_block(x, ratio=8, name=None):
    filters = x.shape[-1]

    # ----- Channel Attention -----
    avg_pool = GlobalAveragePooling2D()(x)
    max_pool = GlobalMaxPooling2D()(x)

    shared_dense_one = Dense(
        filters // ratio, activation="relu", name=f"{name}_mlp1" if name else None
    )
    shared_dense_two = Dense(
        filters, activation="sigmoid", name=f"{name}_mlp2" if name else None
    )

    mlp_avg = shared_dense_two(shared_dense_one(avg_pool))
    mlp_max = shared_dense_two(shared_dense_one(max_pool))

    channel_attention = Add()([mlp_avg, mlp_max])
    channel_attention = Reshape((1, 1, filters))(channel_attention)
    x = Multiply()([x, channel_attention])  # Apply channel attention

    # ----- Spatial Attention -----
    avg_pool_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool_spatial = tf.reduce_max(x, axis=-1, keepdims=True)
    concat = tf.concat([avg_pool_spatial, max_pool_spatial], axis=-1)

    spatial_attention = Conv2D(
        1,
        kernel_size=7,
        padding="same",
        activation="sigmoid",
        name=f"{name}_spatial" if name else None,
    )(concat)
    x = Multiply()([x, spatial_attention])  # Apply spatial attention

    return x


def wide_resnet_block(
    x, c, filters=64, widen_factor=2, l2_reg=1e-6, drop_rate=0.0, nconvs=2
):
    """Wide ResNet Block with CoordConv2D"""
    shortcut_x, shortcut_c = x, c  # Skip connection

    # 3x3 CoordConv2D (Wider filters)
    for i in range(nconvs):
        x, c = CoordConv2D(
            filters=filters * widen_factor,
            kernel_size=3,
            padding="same",
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            activation=None,
        )([x, c])
        x = BatchNormalization(momentum=0.95)(x)
    # Skip Connection
    shortcut_x, shortcut_c = CoordConv2D(
        filters=filters * widen_factor,
        kernel_size=1,
        padding="same",
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        activation=None,
    )([shortcut_x, shortcut_c])

    # x = se_block(x)

    # Add Residual Connection
    x = Activation("relu")(x)
    x = Add()([x, shortcut_x])

    # Pooling and dropout
    x = MaxPool2D(pool_size=2, strides=2, padding="same")(x)
    c = MaxPool2D(pool_size=2, strides=2, padding="same")(c)

    if drop_rate > 0:
        x = Dropout(rate=drop_rate)(x)

    return x, c


tf.config.optimizer.set_jit(True)  # Enable XLA


def main(config):

    # Gather all hyperparams
    epochs = config.get("epochs")
    start_filters = config.get("start_filters")
    decay_steps = config.get("decay_steps")
    lr = config.get("learning_rate")
    wN = config.get("wN")
    w0 = config.get("w0")
    w1 = config.get("w1")
    w2 = config.get("w2")
    wW = config.get("wW")
    input_variables = config.get("input_variables")
    model = config.get("model")
    dataloader = config.get("dataloader")
    dataloader_kwargs = config.get("dataloader_kwargs")

    logging.info(f"Using {dataloader} dataloader")
    logging.info(f"Running with config: {config}")
    weights = {"wN": wN, "w0": w0, "w1": w1, "w2": w2, "wW": wW}

    # Data Loaders
    dataloader_kwargs.update(
        {"select_keys": input_variables + ["range_folded_mask", "coordinates"]}
    )
    import tensorflow_datasets as tfds

    import tornet.data.tfds.tornet.tornet_dataset_builder

    # Apply to Train and Validation Data
    ds_train = get_dataloader(
        config["dataloader"],
        DATA_ROOT,
        config["train_years"],
        "train",
        config["batch_size"],
        weights,
        **config["dataloader_kwargs"],
    )
    ds_val = get_dataloader(
        config["dataloader"],
        DATA_ROOT,
        config["train_years"],
        "train",
        config["batch_size"],
        {"wN": 1.0, "w0": 1.0, "w1": 1.0, "w2": 1.0, "wW": 1.0},
        **config["dataloader_kwargs"],
    )
    x, _, _ = next(iter(ds_train))
    in_shapes = (None, None, get_shape(x)[-1])
    c_shapes = (None, None, x["coordinates"].shape[-1])

    nn = build_model(
        shape=in_shapes,
        c_shape=c_shapes,
        start_filters=start_filters,
        l2_reg=config["l2_reg"],
        input_variables=input_variables,
        model=model,
        head=config["head"],
    )
    print(nn.summary())
    from_logits = False

    def focal_loss(gamma=2.0, alpha=0.85):
        def loss_fn(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            return -tf.reduce_mean(alpha * tf.pow(1.0 - pt, gamma) * tf.math.log(pt))

        return loss_fn

    def tversky_loss(alpha=0.5, beta=0.5, smooth=1e-6):
        """
        Tversky Loss: adjusts trade-off between FP and FN.
        alpha = weight for FP
        beta = weight for FN
        """

        def loss_fn(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            tp = tf.reduce_sum(y_true * y_pred)
            fp = tf.reduce_sum((1 - y_true) * y_pred)
            fn = tf.reduce_sum(y_true * (1 - y_pred))
            return 1 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)

        return loss_fn

    def combo_loss(alpha=0.7):
        return lambda y_true, y_pred: alpha * tversky_loss(alpha=0.5, beta=0.5)(
            y_true, y_pred
        ) + (1 - alpha) * focal_loss(gamma=2.0, alpha=0.85)(y_true, y_pred)

    loss = combo_loss(alpha=0.7)
    lr_schedule = CosineDecayRestarts(
        initial_learning_rate=lr,
        first_decay_steps=decay_steps,  # 1 epoch
        t_mul=2.0,  # each cycle doubles
        m_mul=0.9,  # restart peak decays slightly
    )

    opt = SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    # opt=Adam(learning_rate=lr)
    metrics = [
        keras.metrics.AUC(
            from_logits=from_logits, curve="PR", name="AUCPR", num_thresholds=2000
        )
    ]
    nn.compile(loss=loss, metrics=metrics, optimizer=opt, jit_compile=True)

    # Experiment Directory

    # Callbacks (Now Monitors AUCPR)
    callbacks = [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.EarlyStopping(
            monitor="val_AUCPR", patience=3, mode="max", restore_best_weights=True
        ),
    ]

    # Train Model
    history = nn.fit(ds_train, epochs=epochs, validation_data=ds_val, verbose=1)
    scores = nn.evaluate(ds_val)

    # Get Best Metrics
    best_aucpr = max(history.history.get("val_AUCPR", [0]))
    return {"AUCPR": best_aucpr}


if __name__ == "__main__":
    config = DEFAULT_CONFIG
    if len(sys.argv) > 1:
        config.update(json.load(open(sys.argv[1], "r")))
    main(config)
