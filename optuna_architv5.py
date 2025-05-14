import json
import logging
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import optuna
import tensorflow as tf
from optuna.integration import TFKerasPruningCallback
from tensorflow import keras
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    MaxPool2D,
    Multiply,
    ReLU,
    Reshape,
)

from tornet.metrics.keras import metrics as tfm
from tornet.models.keras.layers import CoordConv2D

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
STUDY_DIR = os.path.join("optuna_studies", f"study_{timestamp}")
os.makedirs(STUDY_DIR, exist_ok=True)
import json
import logging
import os
import random
import shutil
import sys
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts

from tornet.data import preprocess as pp
from tornet.data.constants import ALL_VARIABLES, CHANNEL_MIN_MAX
from tornet.data.loader import get_dataloader
from tornet.data.preprocess import get_shape

logging.basicConfig(level=logging.ERROR)
SEED = 42
# Set random seeds for reproducibility
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
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"


logging.info(f"TORNET_ROOT={DATA_ROOT}")


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


def build_model(
    model="wide_resnet",
    shape: Tuple[int] = (120, 240, 2),
    c_shape: Tuple[int] = (120, 240, 2),
    input_variables: List[str] = ALL_VARIABLES,
    start_filters: int = 64,
    nconvs: int = 2,
    l2_reg: float = 0.001,
    background_flag: float = -3.0,
    include_range_folded: bool = True,
    dropout_rate=0.1,
    **config,
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
            widen_factor=config["widen_factor"],
            l2_reg=l2_reg,
            nconvs=nconvs,
            drop_rate=dropout_rate,
        )
        x = se_block(x)
    x = Conv2D(128, 3, padding="same", use_bias=False)(x)  # <-- no bias
    x = BatchNormalization()(x)
    x = ReLU()(x)
    attention_map = Conv2D(
        1, 1, activation="sigmoid", name="attention_map", use_bias=False
    )(
        x
    )  # shape (B, H, W, 1)
    attention_map = Dropout(rate=config["attn_dropout"], name="attention_dropout")(
        attention_map
    )
    x_weighted = Multiply()([x, attention_map])

    x_avg = GlobalAveragePooling2D()(x_weighted)
    x_max = GlobalMaxPooling2D()(x_weighted)
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


def wide_resnet_block(
    x, c, filters=64, widen_factor=2, l2_reg=1e-6, drop_rate=0.1, nconvs=2
):
    """Wide ResNet Block with CoordConv2D"""
    shortcut_x, shortcut_c = x, c  # Skip connection

    # 3x3 CoordConv2D (Wider filters)
    for i in range(nconvs):
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x, c = CoordConv2D(
            filters=filters * widen_factor,
            kernel_size=3,
            padding="same",
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            activation=None,
        )([x, c])
    # Skip Connection
    shortcut_x, shortcut_c = CoordConv2D(
        filters=filters * widen_factor,
        kernel_size=1,
        padding="same",
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        activation=None,
    )([shortcut_x, shortcut_c])

    # Add Residual Connection
    x = ReLU()(x)
    x = Add()([x, shortcut_x])

    # Pooling and dropout
    x = MaxPool2D(pool_size=2, strides=2, padding="same")(x)
    c = MaxPool2D(pool_size=2, strides=2, padding="same")(c)

    if drop_rate > 0:
        x = Dropout(rate=drop_rate)(x)

    return x, c


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

    # Use tf.constant directly for faster graph compilation
    mean = tf.constant([mean] * n_sweeps, dtype=tf.float32)
    std = tf.constant([std] * n_sweeps, dtype=tf.float32)

    return FastNormalize(mean, std)(x)


# Enable GPU Memory Growth
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.optimizer.set_jit(True)

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


DEFAULT_CONFIG = {
    "epochs": 100,
    "input_variables": ["DBZ", "VEL", "KDP", "ZDR", "RHOHV", "WIDTH"],
    "train_years": [2019, 2020],
    "val_years": [2021, 2022],
    "batch_size": 128,
    "model": "wide_resnet",
    "start_filters": 48,
    "learning_rate": 1e-4,
    "decay_steps": 1386,
    "decay_rate": 0.958,
    "l2_reg": 1e-5,
    "weights": {
        "wN": 1.0,
        "w0": 1.0,
        "w1": 1.0,
        "w2": 2.0,
        "wW": 0.5,
    },
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


@keras.utils.register_keras_serializable()
class ExpandDimsTwice(keras.layers.Layer):
    def call(self, inputs):
        return tf.expand_dims(tf.expand_dims(inputs, axis=1), axis=1)


@keras.utils.register_keras_serializable()
class StackAvgMax(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.stack(inputs, axis=1)


def objective(trial):
    config = DEFAULT_CONFIG.copy()

    config.update(
        {
            "dropout_rate": round(trial.suggest_float("dropout_rate", 0.05, 0.5), 5),
            "learning_rate": round(
                trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True), 5
            ),
            "l2_reg": round(trial.suggest_float("l2_reg", 1e-6, 1e-3, log=True), 5),
            "decay_steps": trial.suggest_int("decay_steps", 970, 1800),
            "label_smoothing": round(
                trial.suggest_float("label_smoothing", 0.05, 0.2), 5
            ),
            "nconvs": trial.suggest_categorical("nconvs", [1, 2, 3]),
            "widen_factor": trial.suggest_categorical("widen_factor", [1, 2, 4]),
            "start_filters": trial.suggest_categorical("start_filters", [48, 96]),
            "attn_dropout": round(trial.suggest_float("attn_dropout", 0.1, 0.3), 5),
        }
    )

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
    model = build_model(shape=in_shapes, c_shape=c_shapes, **config)

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        config["learning_rate"], config["decay_steps"], config["decay_rate"]
    )
    opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    loss = keras.losses.BinaryCrossentropy(
        from_logits=False, label_smoothing=config["label_smoothing"]
    )

    try:
        from_logits = False
        model.compile(
            loss=loss,
            optimizer=opt,
            metrics=[
                keras.metrics.AUC(
                    from_logits=from_logits,
                    curve="PR",
                    name="AUCPR",
                    num_thresholds=1000,
                ),
                tfm.BinaryAccuracy(from_logits, name="BinaryAccuracy"),
                tfm.TruePositives(from_logits, name="TruePositives"),
                tfm.FalsePositives(from_logits, name="FalsePositives"),
                tfm.TrueNegatives(from_logits, name="TrueNegatives"),
                tfm.FalseNegatives(from_logits, name="FalseNegatives"),
                tfm.Precision(from_logits, name="Precision"),
                tfm.Recall(from_logits, name="Recall"),
                tfm.F1Score(from_logits=from_logits, name="F1"),
            ],
            jit_compile=True,
        )

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_AUCPR", patience=2, mode="max", restore_best_weights=True
        )
        prune_cb = TFKerasPruningCallback(trial, "val_AUCPR")
        history = model.fit(
            ds_train,
            epochs=config["epochs"],
            validation_data=ds_val,
            verbose=1,
            callbacks=[early_stopping, prune_cb],
        )
        best_aucpr = max(history.history.get("val_AUCPR", [0]))
        log_trial(trial, best_aucpr)
        return best_aucpr
    except tf.errors.ResourceExhaustedError as e:
        # Handle OOM error: log it and return a "failed" result to Optuna (nan)
        print(
            f"Out of memory error occurred in trial {trial.number}. Skipping this trial."
        )
        log_trial(trial, float("nan"))  # Optionally log 'nan' to indicate failure
        return float("nan")  # Return NaN to indicate the trial failed


def log_trial(trial, trial_results):
    trial_data = {
        "trial_id": trial.number,
        "params": trial.params,
        "value": trial_results,
    }

    trial_log_path = os.path.join(STUDY_DIR, "all_trials.json")

    # Load existing log or initialize empty list
    if os.path.exists(trial_log_path):
        with open(trial_log_path, "r") as f:
            trial_log = json.load(f)
    else:
        trial_log = []

    trial_log.append(trial_data)

    # Write updated log
    with open(trial_log_path, "w") as f:
        json.dump(trial_log, f, indent=4)


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=SEED),
        pruner=optuna.pruners.HyperbandPruner(),
    )
    study.optimize(objective, n_trials=150)

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
