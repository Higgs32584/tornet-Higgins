import json
import logging
import os
import random
import shutil
import sys
from typing import List, Tuple

import keras
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    MaxPool2D,
    Multiply,
    Reshape,
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts

import tornet.data.tfds.tornet.tornet_dataset_builder
from tornet.data.constants import ALL_VARIABLES, CHANNEL_MIN_MAX
from tornet.data.loader import get_dataloader
from tornet.data.preprocess import get_shape
from tornet.metrics.keras import metrics as tfm
from tornet.models.keras.layers import CoordConv2D, FillNaNs
from tornet.utils.general import make_callback_dirs, make_exp_dir

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# Environment Variables
DATA_ROOT = "/home/ubuntu/tfds"
TFDS_DATA_DIR = DATA_ROOT
os.environ["TORNET_ROOT"] = DATA_ROOT
os.environ["TFDS_DATA_DIR"] = TFDS_DATA_DIR

# Environment Variables
EXP_DIR = "."
DATA_ROOT = "/home/ubuntu/tfds"
TORNET_ROOT = DATA_ROOT
TFDS_DATA_DIR = "/home/ubuntu/tfds"
DATA_ROOT = "/home/ubuntu/tfds"
TFDS_DATA_DIR = "/home/ubuntu/tfds"
os.environ["TORNET_ROOT"] = DATA_ROOT
os.environ["TFDS_DATA_DIR"] = TFDS_DATA_DIR
import logging

logging.basicConfig(level=logging.INFO)

from tornet.data.constants import ALL_VARIABLES
from tornet.data.loader import get_dataloader
from tornet.data.preprocess import get_shape
from tornet.metrics.keras import metrics as tfm
from tornet.models.keras.cnn_baseline import build_model
from tornet.models.keras.losses import mae_loss
from tornet.utils.general import make_callback_dirs, make_exp_dir

EXP_DIR = os.environ.get("EXP_DIR", ".")
DATA_ROOT = os.environ["TORNET_ROOT"]
logging.info("TORNET_ROOT=" + DATA_ROOT)
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
tf.config.optimizer.set_jit(True)


DEFAULT_CONFIG = {
    "epochs": 100,
    "input_variables": ALL_VARIABLES,
    "train_years": list(range(2013, 2021)),
    "val_years": list(range(2021, 2023)),
    "batch_size": 128,
    "model": "vgg",
    "start_filters": 48,
    "learning_rate": 1e-4,
    "decay_steps": 1386,
    "decay_rate": 0.958,
    "l2_reg": 1e-5,
    "wN": 1.0,
    "w0": 1.0,
    "w1": 1.0,
    "w2": 2.0,
    "wW": 0.5,
    "label_smooth": 0.1,
    "loss": "cce",
    "head": "maxpool",
    "exp_name": "tornet_baseline",
    "exp_dir": EXP_DIR,
    "dataloader": "tensorflow-tfds",
    "dataloader_kwargs": {},
}


def vgg_block(x, c, filters=64, ksize=3, n_convs=2, l2_reg=1e-6, drop_rate=0.0):

    for _ in range(n_convs):
        x, c = CoordConv2D(
            filters=filters,
            kernel_size=ksize,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            padding="same",
            activation=None,
        )([x, c])
    x = keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")(x)
    c = keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")(c)
    if drop_rate > 0:
        x = keras.layers.Dropout(rate=drop_rate)(x)
    return x, c


def normalize(x, name: str):
    """
    Channel-wise normalization using known CHANNEL_MIN_MAX
    """
    min_max = np.array(CHANNEL_MIN_MAX[name])  # [2,]
    n_sweeps = x.shape[-1]

    # choose mean,var to get approximate [-1,1] scaling
    var = ((min_max[1] - min_max[0]) / 2) ** 2  # scalar
    var = np.array(
        n_sweeps
        * [
            var,
        ]
    )  # [n_sweeps,]

    offset = (min_max[0] + min_max[1]) / 2  # scalar
    offset = np.array(
        n_sweeps
        * [
            offset,
        ]
    )  # [n_sweeps,]

    return keras.layers.Normalization(
        mean=offset, variance=var, name="Normalize_%s" % name
    )(x)


def build_model(
    shape: Tuple[int] = (120, 240, 2),
    c_shape: Tuple[int] = (120, 240, 2),
    input_variables: List[str] = ALL_VARIABLES,
    start_filters: int = 64,
    l2_reg: float = 0.001,
    background_flag: float = -3.0,
    include_range_folded: bool = True,
    head="maxpool",
    dropout_rate=0.1,
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

    x, c = vgg_block(
        x, c, filters=start_filters, ksize=3, l2_reg=l2_reg, n_convs=2, drop_rate=0.1
    )  # (60,120)
    x, c = vgg_block(
        x,
        c,
        filters=2 * start_filters,
        ksize=3,
        l2_reg=l2_reg,
        n_convs=2,
        drop_rate=0.1,
    )  # (30,60)
    x, c = vgg_block(
        x,
        c,
        filters=4 * start_filters,
        ksize=3,
        l2_reg=l2_reg,
        n_convs=3,
        drop_rate=0.1,
    )  # (15,30)
    x, c = vgg_block(
        x,
        c,
        filters=8 * start_filters,
        ksize=3,
        l2_reg=l2_reg,
        n_convs=3,
        drop_rate=0.1,
    )  # (7,15)

    if head == "mlp":
        # MLP head
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(units=4096, activation="relu")(x)
        x = keras.layers.Dense(units=2024, activation="relu")(x)
        output = keras.layers.Dense(1)(x)
    elif head == "maxpool":
        # Per gridcell
        x = keras.layers.Conv2D(
            filters=512,
            kernel_size=1,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            activation="relu",
        )(x)
        x = keras.layers.Conv2D(
            filters=256,
            kernel_size=1,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            activation="relu",
        )(x)
        x = keras.layers.Conv2D(filters=1, kernel_size=1, name="heatmap")(x)
        # Max in scene
        output = keras.layers.GlobalMaxPooling2D()(x)
        output = keras.layers.Activation("sigmoid")(output)

    return keras.Model(inputs=inputs, outputs=output)


DEFAULT_CONFIG = {
    "epochs": 100,
    "input_variables": ["DBZ", "VEL", "KDP", "ZDR", "RHOHV", "WIDTH"],
    "train_years": [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "val_years": [2021, 2022],
    "batch_size": 128,
    "model": "wide_resnet",
    "start_filters": 48,
    "learning_rate": 1e-4,
    "decay_steps": 1386,
    "decay_rate": 0.958,
    "l2_reg": 1e-5,
    "wN": 1.0,
    "w0": 1.0,
    "w1": 1.0,
    "w2": 2.0,
    "wW": 0.5,
    "nconvs": 2,
    "dropout_rate": 0.1,
    "loss": "cce",
    "head": "maxpool",
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
    # Gather all hyperparams
    epochs = config.get("epochs")
    batch_size = config.get("batch_size")
    start_filters = config.get("start_filters")
    dropout_rate = config.get("dropout_rate")
    first_decay_steps = config.get("first_decay_steps")
    lr = config.get("learning_rate")
    l2_reg = config.get("l2_reg")
    wN = config.get("wN")
    w0 = config.get("w0")
    w1 = config.get("w1")
    w2 = config.get("w2")
    wW = config.get("wW")
    input_variables = config.get("input_variables")
    exp_name = config.get("exp_name")
    model = config.get("model")
    exp_dir = config.get("exp_dir")
    train_years = config.get("train_years")
    val_years = config.get("val_years")
    dataloader = config.get("dataloader")
    dataloader_kwargs = config.get("dataloader_kwargs")

    logging.info(f"Using {dataloader} dataloader")
    logging.info(f"Running with config: {config}")
    weights = {"wN": wN, "w0": w0, "w1": w1, "w2": w2, "wW": wW}

    # Data Loaders
    dataloader_kwargs.update(
        {"select_keys": input_variables + ["range_folded_mask", "coordinates"]}
    )
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

    in_shapes = (120, 240, get_shape(x)[-1])
    c_shapes = (120, 240, x["coordinates"].shape[-1])
    nn = build_model(
        shape=in_shapes,
        c_shape=c_shapes,
        start_filters=start_filters,
        l2_reg=l2_reg,
        input_variables=input_variables,
        head=config["head"],
    )
    print(nn.summary())

    # Loss Function
    # Optimizer with Learning Rate Decay
    from_logits = False
    lr = keras.optimizers.schedules.ExponentialDecay(
        config["learning_rate"],
        config["decay_steps"],
        config["decay_rate"],
        staircase=False,
        name="exp_decay",
    )
    loss = keras.losses.BinaryCrossentropy(from_logits=from_logits, label_smoothing=0.1)
    opt = keras.optimizers.Adam(learning_rate=lr)
    # Metrics (Optimize AUCPR)
    metrics = [
        keras.metrics.AUC(
            from_logits=from_logits, curve="PR", name="AUCPR", num_thresholds=1000
        ),
        keras.metrics.AUC(from_logits=from_logits, name="AUC", num_thresholds=1000),
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
