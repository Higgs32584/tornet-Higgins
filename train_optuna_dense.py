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
from optuna.distributions import FloatDistribution
from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState, create_trial
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    AveragePooling2D,
    Concatenate,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    MaxPool2D,
    Multiply,
    PReLU,
    ReLU,
    Reshape,
)

from tornet.data.constants import ALL_VARIABLES, CHANNEL_MIN_MAX
from tornet.data.loader import get_dataloader
from tornet.data.preprocess import get_shape
from tornet.metrics.keras import metrics as tfm
from tornet.models.keras.cnn_baseline import normalize, vgg_block
from tornet.models.keras.layers import CoordConv2D, FillNaNs
from tornet.models.keras.losses import mae_loss
from tornet.utils.general import make_exp_dir

logging.basicConfig(level=logging.INFO)
SEED = 42


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

tf.config.optimizer.set_jit(True)
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
tf.config.optimizer.set_jit(True)
# Create a unique directory for this study
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
STUDY_DIR = os.path.join("optuna_studies", f"study_{timestamp}")
os.makedirs(STUDY_DIR, exist_ok=True)

DEFAULT_CONFIG = {
    "epochs": 100,
    "input_variables": ["DBZ", "VEL", "KDP", "ZDR", "RHOHV", "WIDTH"],
    "train_years": [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "val_years": [2021, 2022],
    "batch_size": 128,
    "model": "wide_resnet",
    "start_filters": 96,
    "learning_rate": 0.00441,
    "l2_reg": 3.83e-7,
    "label_smoothing": 0.0845,
    "dropout_rate": 0.0997,
    "warmup_epochs": 4,
    "t_mul": 2.0,
    "m_mul": 0.809,
    "attention_dropout": 0.0,
    "weights": {"wN": 1.0, "w0": 1.0, "w1": 1.0, "w2": 2.0, "wW": 0.5},
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


def dense_layer(x, c, growth_rate, kernel_regularizer=None):
    out = BatchNormalization()(x)
    out = ReLU()(out)
    out, c = CoordConv2D(
        growth_rate,
        kernel_size=3,
        padding="same",
        activation=None,
        kernel_regularizer=kernel_regularizer,
    )([out, c])
    x = Concatenate()([x, out])
    return x, c


def dense_block(x, c, num_layers, growth_rate, kernel_regularizer=None):
    for _ in range(num_layers):
        x, c = dense_layer(x, c, growth_rate, kernel_regularizer=kernel_regularizer)
    return x, c


def transition_layer(x, c, compression=0.5, kernel_regularizer=None):
    filters = int(x.shape[-1] * compression)

    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(
        filters, kernel_size=1, use_bias=False, kernel_regularizer=kernel_regularizer
    )(
        x
    )  # ✅ apply reg here
    x = MaxPool2D(pool_size=2)(x)
    c = MaxPool2D(pool_size=2)(c)

    return x, c


from tensorflow.keras.regularizers import l2


def build_model(
    shape: Tuple[int] = (120, 240, 2),
    c_shape: Tuple[int] = (120, 240, 2),
    input_variables: List[str] = ALL_VARIABLES,
    start_filters: int = 48,
    attention_dropout: float = 0.2,
    growth_rate: int = 16,
    l2_reg: float = 1e-5,
    dense_layers: int = 4,
    dropout_rate: float = 0.1,
    compression: float = 0.75,
) -> keras.Model:
    background_flag = -3.0
    include_range_folded = True
    reg = l2(l2_reg)

    # Inputs
    inputs = {v: keras.Input(shape, name=v) for v in input_variables}
    n_sweeps = shape[2]
    coords = keras.Input(c_shape, name="coordinates")
    inputs["coordinates"] = coords

    x = keras.layers.Concatenate(axis=-1)(
        [normalize(inputs[v], v) for v in input_variables]
    )
    x = FillNaNs(background_flag)(x)

    if include_range_folded:
        range_folded = keras.Input(shape[:2] + (n_sweeps,), name="range_folded_mask")
        inputs["range_folded_mask"] = range_folded
        x = keras.layers.Concatenate(axis=-1)([x, range_folded])

    # Initial CoordConv
    x, c = CoordConv2D(
        start_filters,
        kernel_size=3,
        padding="same",
        activation=None,
        kernel_regularizer=reg,
    )([x, coords])

    # Dense Blocks + Transition
    x, c = dense_block(
        x, c, num_layers=dense_layers, growth_rate=growth_rate, kernel_regularizer=reg
    )
    x, c = transition_layer(x, c, compression=compression, kernel_regularizer=reg)
    x, c = dense_block(
        x, c, num_layers=dense_layers, growth_rate=growth_rate, kernel_regularizer=reg
    )

    # SE Block
    x = se_block(x)

    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(128, 3, padding="same", use_bias=False)(x)

    # Attention
    attn = Conv2D(1, 1, activation="sigmoid", use_bias=False, name="attention_map")(x)
    attn = Dropout(rate=attention_dropout, name="attention_dropout")(attn)
    x = Multiply()([x, attn])

    # Pooling and projection
    x_avg = GlobalAveragePooling2D()(x)
    x_max = GlobalMaxPooling2D()(x)
    x = keras.layers.Concatenate()([x_avg, x_max])

    x = Dense(64)(x)
    x = PReLU()(x)

    output = Dense(1, activation="sigmoid", dtype="float32")(x)

    return keras.Model(inputs=inputs, outputs=output)


def se_block(x, ratio=16, name=None):
    filters = x.shape[-1]
    se = GlobalAveragePooling2D(name=f"{name}_gap" if name else None)(x)
    se = Dense(
        filters // ratio, activation="relu", name=f"{name}_fc1" if name else None
    )(se)
    se = Dense(filters, activation="sigmoid", name=f"{name}_fc2" if name else None)(se)
    se = Reshape((1, 1, filters), name=f"{name}_reshape" if name else None)(se)
    x = Multiply(name=f"{name}_scale" if name else None)([x, se])
    return x


def objective(trial):
    try:
        config = DEFAULT_CONFIG.copy()
        config["start_filters"] = trial.suggest_categorical(
            "start_filters", [16, 32, 48, 64, 96]
        )
        config["growth_rate"] = trial.suggest_categorical(
            "growth_rate", [8, 16, 24, 32]
        )
        config["dense_layers"] = trial.suggest_int("dense_layers", 2, 6)
        config["label_smoothing"] = trial.suggest_float("label_smoothing", 0.0, 0.1)
        config["compression"] = trial.suggest_float("compression", 0.4, 0.8)
        config["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-6, 1e-3, log=True
        )

        # 🛡️ Regularization
        config["l2_reg"] = trial.suggest_float("l2_reg", 1e-10, 1e-4, log=True)
        config["dropout_rate"] = trial.suggest_float("dropout_rate", 0.0, 0.3)
        config["attention_dropout"] = trial.suggest_float("attention_dropout", 0.0, 0.4)

        logging.info(f"Tuning with config: {config}")

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
            shape=in_shapes,
            c_shape=c_shapes,
            input_variables=config["input_variables"],
            attention_dropout=config["attention_dropout"],
            l2_reg=config["l2_reg"],
            dense_layers=config["dense_layers"],
            start_filters=config["start_filters"],
            growth_rate=config["growth_rate"],
            dropout_rate=config["dropout_rate"],
        )

        loss = keras.losses.BinaryCrossentropy(
            label_smoothing=config["label_smoothing"]
        )
        steps_per_epoch = len(ds_train)
        print(steps_per_epoch)
        lr = WarmUpCosine(
            base_lr=config["learning_rate"],
            warmup_steps=config["warmup_epochs"] * steps_per_epoch,
            restart_steps=10 * steps_per_epoch,
            t_mul=config["t_mul"],
            m_mul=config["m_mul"],
            alpha=1e-6,
        )
        opt = keras.optimizers.Adam(learning_rate=lr)
        from_logits = False
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
        model.compile(loss=loss, optimizer=opt, metrics=metrics, jit_compile=True)

        # Train the model

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_AUCPR", patience=3, mode="max", restore_best_weights=True
        )
        pruning_callback = TFKerasPruningCallback(trial, monitor="val_AUCPR")

        history = model.fit(
            ds_train,
            epochs=config["epochs"],
            validation_data=ds_val,
            verbose=1,
            callbacks=[early_stopping, pruning_callback],
        )

        # Get best validation AUCPR
        best_aucpr = max(history.history.get("val_AUCPR", [0]))

        # Log trial results
        log_trial(trial, best_aucpr)
        tf.keras.backend.clear_session()
        import gc

        gc.collect()

        return best_aucpr
    except tf.errors.ResourceExhaustedError as e:
        print(f"Trial {trial.number} pruned due to resource exhaustion: {e}")
        tf.keras.backend.clear_session()
        import gc

        gc.collect()
        raise optuna.exceptions.TrialPruned()

    except Exception as e:
        # Optional: prune on any unexpected issue during model build
        print(f"Trial {trial.number} failed: {e}")
        tf.keras.backend.clear_session()
        import gc

        gc.collect()
        raise optuna.exceptions.TrialPruned()


all_trials_path = os.path.join(STUDY_DIR, "all_trials.json")
all_trials = []


def log_trial(trial, trial_results):
    """Logs all trials into a single JSON file."""
    trial_data = {
        "trial_id": trial.number,
        "params": trial.params,
        "value": trial_results,
    }

    # Load existing results if file exists
    if os.path.exists(all_trials_path):
        with open(all_trials_path, "r") as f:
            try:
                all_trials_data = json.load(f)
            except json.JSONDecodeError:
                all_trials_data = []
    else:
        all_trials_data = []

    all_trials_data.append(trial_data)

    with open(all_trials_path, "w") as f:
        json.dump(all_trials_data, f, indent=4)


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
        pruner=optuna.pruners.HyperbandPruner(),
    )
    study.optimize(objective, n_trials=100)

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
