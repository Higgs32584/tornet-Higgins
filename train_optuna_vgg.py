import json
import logging
import os
import sys
from datetime import datetime
from typing import List, Tuple

import keras
import numpy as np
import optuna
import tensorflow as tf
from keras.optimizers.schedules import CosineDecayRestarts

from tornet.data.constants import ALL_VARIABLES, CHANNEL_MIN_MAX
from tornet.data.loader import get_dataloader
from tornet.data.preprocess import get_shape
from tornet.metrics.keras import metrics as tfm
from tornet.models.keras.cnn_baseline import normalize, vgg_block
from tornet.models.keras.layers import FillNaNs
from tornet.models.keras.losses import mae_loss
from tornet.utils.general import make_exp_dir

logging.basicConfig(level=logging.INFO)
SEED = 42

# Set random seeds for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Environment Variables
DATA_ROOT = "/home/ubuntu/tfds"
TFDS_DATA_DIR = DATA_ROOT
os.environ["TORNET_ROOT"] = DATA_ROOT
os.environ["TFDS_DATA_DIR"] = TFDS_DATA_DIR

# Enable GPU Memory Growth
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

strategy = tf.distribute.MultiWorkerMirroredStrategy()
tf.config.optimizer.set_jit(True)

# Create a unique directory for this study
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
STUDY_DIR = os.path.join("optuna_studies", f"study_{timestamp}")
os.makedirs(STUDY_DIR, exist_ok=True)

DEFAULT_CONFIG = {
    "epochs": 50,
    "batch_size": 128,
    "model": "vgg_2",
    "start_filters": 48,
    "learning_rate": 1e-3,
    "l2_reg": 1e-5,
    "weight_decay": 1e-4,
    "train_years": [2016],
    "val_years": [2017],
    "input_variables": ["DBZ", "VEL", "KDP", "ZDR", "RHOHV", "WIDTH"],
    "exp_name": "tornado_baseline",
    "exp_dir": STUDY_DIR,  # Save study results here
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
    "weights": {
        "wN": 1.0,
        "w0": 1.0,
        "w1": 1.0,
        "w2": 1.0,
        "wW": 1.0,
    },
}


def get_optimized_batch_size(model_name, start_bs=128, min_bs=8, max_attempts=10):
    batch_size = start_bs
    attempt = 0
    while attempt < max_attempts:
        try:
            logging.info(f"Testing batch size {batch_size} for {model_name}")
            ds_train = get_dataloader(
                DEFAULT_CONFIG["dataloader"],
                DATA_ROOT,
                DEFAULT_CONFIG["train_years"],
                "train",
                batch_size,
                None,
                **DEFAULT_CONFIG["dataloader_kwargs"],
            )

            _ = next(iter(ds_train))  # Try fetching one batch
            logging.info(f"Batch size {batch_size} is feasible for {model_name}")
            return batch_size  # Return max feasible batch size
        except tf.errors.ResourceExhaustedError:
            logging.warning(f"OOM at batch size {batch_size}, reducing by half...")
            batch_size = max(batch_size // 2, min_bs)
            attempt += 1
    logging.error(
        f"Could not find feasible batch size for {model_name}. Using {batch_size}."
    )
    return batch_size


# Usage inside the objective function


def build_model(
    model,
    shape: Tuple[int] = (120, 240, 2),
    c_shape: Tuple[int] = (120, 240, 2),
    input_variables: List[str] = ALL_VARIABLES,
    start_filters=48,
    l2_reg=0.003,
    drop_rate=0.1,
):
    inputs = {v: keras.Input(shape, name=v) for v in input_variables}
    normalized_inputs = keras.layers.Concatenate(axis=-1)(
        [normalize(inputs[v], v) for v in input_variables]
    )
    normalized_inputs = FillNaNs(-3.0)(normalized_inputs)

    cin = keras.Input(c_shape, name="coordinates")
    inputs["coordinates"] = cin
    x, c = normalized_inputs, cin

    model_structure = {
        "vgg_1": [(start_filters, 2)],
        "vgg_2": [(2 * start_filters, 2), (4 * start_filters, 2)],
        "vgg_3": [
            (2 * start_filters, 2),
            (4 * start_filters, 3),
            (8 * start_filters, 3),
        ],
        "vgg_4": [
            (2 * start_filters, 2),
            (4 * start_filters, 3),
            (8 * start_filters, 3),
            (16 * start_filters, 3),
        ],
    }

    for filters, n_convs in model_structure[model]:
        x, c = vgg_block(
            x,
            c,
            filters=filters,
            ksize=3,
            l2_reg=l2_reg,
            n_convs=n_convs,
            drop_rate=drop_rate,
        )

    x = keras.layers.Conv2D(filters=256, kernel_size=1, activation="relu")(x)
    x = keras.layers.Conv2D(filters=1, kernel_size=1, name="heatmap")(x)
    output = keras.layers.GlobalMaxPooling2D()(x)

    return keras.Model(inputs=inputs, outputs=output)


def get_optimized_batch_size(model_name, start_bs=128, min_bs=8, max_attempts=5):
    """
    Dynamically determines the optimal batch size for a given model
    by testing memory feasibility.
    """
    batch_size = start_bs
    attempt = 0

    while attempt < max_attempts:
        try:
            logging.info(f"Testing batch size {batch_size} for {model_name}")

            # Load dataset with current batch size
            ds_train = get_dataloader(
                DEFAULT_CONFIG["dataloader"],
                DATA_ROOT,
                DEFAULT_CONFIG["train_years"],
                "train",
                batch_size,
                DEFAULT_CONFIG["weights"],
                **DEFAULT_CONFIG["dataloader_kwargs"],
            )

            # Try fetching one batch to check memory feasibility
            _ = next(iter(ds_train))

            logging.info(f"✅ Batch size {batch_size} is feasible for {model_name}")
            return batch_size  # Return the largest feasible batch size

        except tf.errors.ResourceExhaustedError:
            logging.warning(
                f"🚨 OOM at batch size {batch_size} for {model_name}, reducing by half..."
            )
            batch_size = max(batch_size // 2, min_bs)
            attempt += 1

    logging.error(
        f"⚠️ Could not find feasible batch size for {model_name}. Using {batch_size}."
    )
    return float(batch_size)


def objective(trial):
    config = DEFAULT_CONFIG.copy()

    config["learning_rate"] = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    config["l2_reg"] = trial.suggest_float("l2_reg", 1e-6, 1e-3, log=True)
    config["start_filters"] = trial.suggest_int("start_filters", 64, 128, step=16)
    config["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    config["dropout_rate"] = trial.suggest_float("dropout_rate", 0.05, 0.3)
    config["weights"] = {
        "wN": trial.suggest_float("wN", 0.5, 2.0, log=True),
        "w0": trial.suggest_float("w0", 0.5, 2.0, log=True),
        "w1": trial.suggest_float("w1", 0.5, 2.0, log=True),
        "w2": trial.suggest_float("w2", 1.0, 4.0, log=True),
        "wW": trial.suggest_float("wW", 0.1, 1.0, log=True),
    }
    config["first_decay_steps"] = (
        trial.suggest_int("first_decay_steps", 5, 20) * 1.0
    )  # Controls decay restart interval
    config["t_mul"] = trial.suggest_float(
        "t_mul", 1.0, 3.0, log=True
    )  # Growth factor of decay period
    config["m_mul"] = trial.suggest_float("m_mul", 0.5, 1.0)

    logging.info(f"Tuning with config: {config}")
    config["batch_size"] = 64
    # Load dataset
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
        config["weights"],
        **config["dataloader_kwargs"],
    )

    x, _, _ = next(iter(ds_train))
    in_shapes = (None, None, get_shape(x)[-1])
    c_shapes = (None, None, x["coordinates"].shape[-1])

    model = build_model(
        config["model"],
        shape=in_shapes,
        c_shape=c_shapes,
        input_variables=config["input_variables"],
        l2_reg=config["l2_reg"],
        start_filters=config["start_filters"],
    )
    loss = keras.losses.BinaryCrossentropy()
    lr = CosineDecayRestarts(
        initial_learning_rate=config["learning_rate"],
        first_decay_steps=config["first_decay_steps"],
        t_mul=config["t_mul"],
        m_mul=config["m_mul"],
    )
    opt = keras.optimizers.AdamW(learning_rate=lr, weight_decay=config["weight_decay"])
    metrics = [keras.metrics.AUC(name="AUCPR", curve="PR", num_thresholds=2000)]
    model.compile(loss=loss, optimizer=opt, metrics=metrics)

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
    }
    trial_log_path = os.path.join(STUDY_DIR, f"trial_{trial.number}.json")
    with open(trial_log_path, "w") as f:
        json.dump(trial_data, f, indent=4)


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=150, n_jobs=1)

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
