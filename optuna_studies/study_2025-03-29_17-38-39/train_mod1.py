import json
import logging
import os
import shutil
import sys
from typing import List, Tuple

import numpy as np
import optuna
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
from custom_func import FalseAlarmRate, ThreatScore, combo_loss
from tornet.data.constants import ALL_VARIABLES, CHANNEL_MIN_MAX
from tornet.data.loader import get_dataloader
from tornet.data.preprocess import get_shape
from tornet.metrics.keras import metrics as tfm
from tornet.models.keras.layers import CoordConv2D, FillNaNs
from tornet.utils.general import make_callback_dirs, make_exp_dir

logging.basicConfig(level=logging.ERROR)
SEED = 42

# Set random seeds for reproducibility
import random

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# Environment Variables
DATA_ROOT = "/home/ubuntu/tfds"
TORNET_ROOT = DATA_ROOT
TFDS_DATA_DIR = DATA_ROOT
EXP_DIR = "."
os.environ["TORNET_ROOT"] = DATA_ROOT
os.environ["TFDS_DATA_DIR"] = TFDS_DATA_DIR


logging.info(f"TORNET_ROOT={DATA_ROOT}")


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
    x = Conv2D(128, 3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    attention_map = Conv2D(1, 1, activation="sigmoid", name="attention_map")(
        x
    )  # shape (B, H, W, 1)
    attention_map = Dropout(rate=0.2, name="attention_dropout")(attention_map)
    x_weighted = Multiply()([x, attention_map])

    x_avg = GlobalAveragePooling2D()(x_weighted)
    x_max = GlobalMaxPooling2D()(x_weighted)
    x_concat = keras.layers.Concatenate()([x_avg, x_max])

    x_dense = Dense(64, activation="relu")(x_concat)
    output = Dense(1, activation="sigmoid", name="output")(x_dense)
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
        x = BatchNormalization()(x)
    # Skip Connection
    shortcut_x, shortcut_c = CoordConv2D(
        filters=filters * widen_factor,
        kernel_size=1,
        padding="same",
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        activation=None,
    )([shortcut_x, shortcut_c])

    # Add Residual Connection
    x = Activation("relu")(x)
    x = Add()([x, shortcut_x])

    # Pooling and dropout
    x = MaxPool2D(pool_size=2, strides=2, padding="same")(x)
    c = MaxPool2D(pool_size=2, strides=2, padding="same")(c)

    if drop_rate > 0:
        x = Dropout(rate=drop_rate)(x)

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


# Enable GPU Memory Growth
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
import datetime

strategy = tf.distribute.MirroredStrategy()
# os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
strategy = tf.distribute.MultiWorkerMirroredStrategy()
tf.config.optimizer.set_jit(True)

# Create a unique directory for this study
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
STUDY_DIR = os.path.join("optuna_studies", f"study_{timestamp}")
os.makedirs(STUDY_DIR, exist_ok=True)
# tf.config.optimizer.set_jit(True)  # Enable XLA (Accelerated Linear Algebra)
logging.info(f"Number of devices: {strategy.num_replicas_in_sync}")
# Default Configuration
DEFAULT_CONFIG = {
    "epochs": 100,
    "input_variables": ["DBZ", "VEL", "KDP", "ZDR", "RHOHV", "WIDTH"],
    "train_years": [2014],
    "val_years": [2015],
    "batch_size": 128,
    "model": "wide_resnet",
    "start_filters": 48,
    "learning_rate": 3e-4,
    "decay_steps": 1386,
    "decay_rate": 0.958,
    "dropout_rate": 0.1,
    "l2_reg": 1e-4,
    "wN": 0.25,
    "wW": 0.75,
    "w0": 3.0,
    "w1": 5.0,
    "w2": 8.0,
    "label_smooth": 0.1,
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


def objective(trial):
    config = DEFAULT_CONFIG.copy()
    config["learning_rate"] = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    config["l2_reg"] = trial.suggest_float("l2_reg", 1e-7, 1e-5, log=True)
    config["weight_decay"] = trial.suggest_float("weight_decay", 1e-7, 1e-5, log=True)
    config["dropout_rate"] = trial.suggest_float("dropout_rate", 0.05, 0.2)
    config["weights"] = {
        "wN": trial.suggest_float("wN", 0.05, 0.2),
        "w0": trial.suggest_int("w0", 0.5, 3.0),
        "w1": trial.suggest_int("w1", 2.0, 5.0),
        "w2": trial.suggest_int("w2", 5.0, 15.0),
        "wW": trial.suggest_float("wN", 0.01, 0.25),
    }
    config["first_decay_steps"] = (
        trial.suggest_int("first_decay_steps", 1000, 2500) * 1.0
    )  # Controls decay restart interval
    logging.info(f"Tuning with config: {config}")
    config["batch_size"] = 128
    # Load dataset
    import tensorflow_datasets as tfds

    import tornet.data.tfds.tornet.tornet_dataset_builder

    ds_train = get_dataloader(
        config["dataloader"],
        DATA_ROOT,
        config["train_years"],
        "train",
        config["batch_size"],
        config["weights"],
        **config["dataloader_kwargs"],
    )
    ds_val = get_dataloader(
        config["dataloader"],
        DATA_ROOT,
        config["val_years"],
        "train",
        config["batch_size"],
        {"wN": 1.0, "w0": 1.0, "w1": 1.0, "w2": 1.0, "wW": 1.0},
        **config["dataloader_kwargs"],
    )

    x, _, _ = next(iter(ds_train))
    in_shapes = (None, None, get_shape(x)[-1])
    c_shapes = (None, None, x["coordinates"].shape[-1])

    model = build_model(
        model=config["model"],
        shape=in_shapes,
        c_shape=c_shapes,
        input_variables=config["input_variables"],
        l2_reg=config["l2_reg"],
        start_filters=config["start_filters"],
        dropout_rate=config["dropout_rate"],
    )
    loss = combo_loss()

    # Optimizer with Learnindg Rate Decay

    lr_schedule = CosineDecayRestarts(
        initial_learning_rate=config["learning_rate"],
        first_decay_steps=config["first_decay_steps"],  # 1 epoch
        t_mul=2.0,  # each cycle doubles
        m_mul=0.9,  # restart peak decays slightly
    )

    opt = AdamW(
        learning_rate=lr_schedule,
        weight_decay=config["weight_decay"],
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
    )

    from_logits = False

    metrics = [
        keras.metrics.AUC(
            from_logits=from_logits, curve="PR", name="AUCPR", num_thresholds=1000
        ),
        tfm.BinaryAccuracy(from_logits, name="BinaryAccuracy"),
        tfm.TruePositives(from_logits, name="TruePositives"),
        tfm.FalsePositives(from_logits, name="FalsePositives"),
        tfm.TrueNegatives(from_logits, name="TrueNegatives"),
        tfm.FalseNegatives(from_logits, name="FalseNegatives"),
        tfm.Precision(from_logits, name="Precision"),
        tfm.Recall(from_logits, name="Recall"),
        FalseAlarmRate(name="FalseAlarmRate"),
        tfm.F1Score(from_logits=from_logits, name="F1"),
        ThreatScore(name="ThreatScore"),
    ]
    model.compile(loss=loss, optimizer=opt, metrics=metrics, jit_compile=True)

    # Train the model

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_AUCPR", patience=2, mode="max", restore_best_weights=True
    )
    history = model.fit(
        ds_train,
        epochs=config["epochs"],
        validation_data=ds_val,
        verbose=1,
        callbacks=[early_stopping],
    )

    # Get best validation AUCPR
    best_aucpr = max(history.history.get("val_AUCPR", [0]))

    # Log trial results
    log_trial(trial, best_aucpr)
    tf.keras.backend.clear_session()
    import gc

    gc.collect()

    return best_aucpr


def log_trial(trial, trial_results):
    """Logs individual trial results to a JSON file."""
    trial_data = {
        "trial_id": trial.number,
        "params": trial.params,
        "value": trial_results,
        "DEFAULT_CONFIG": DEFAULT_CONFIG,
    }
    trial_log_path = os.path.join(STUDY_DIR, f"trial_{trial.number}.json")
    with open(trial_log_path, "w") as f:
        json.dump(trial_data, f, indent=4)


if __name__ == "__main__":
    with strategy.scope():
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            pruner=optuna.pruners.HyperbandPruner(),
        )
        study.optimize(objective, n_trials=100)

    # Save best trial parameters
    best_trial_data = {
        "best_trial_id": study.best_trial.number,
        "best_params": study.best_trial.params,
        "best_value": study.best_trial.value,
        "DEFAULT_CONFIG": DEFAULT_CONFIG.copy(),
    }

    best_params_path = os.path.join(STUDY_DIR, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(best_trial_data, f, indent=4)

    print(f"Best trial saved to: {best_params_path}")
    print("Best trial:", study.best_trial)
