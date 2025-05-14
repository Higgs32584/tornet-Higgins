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
    "epochs": 50,
    "input_variables": ["DBZ", "VEL", "KDP", "ZDR", "RHOHV", "WIDTH"],
    "train_years": [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "val_years": [2021, 2022],
    "batch_size": 128,
    "model": "wide_resnet",
    "start_filters": 48,
    "learning_rate": 1e-4,
    "decay_rate": 0.958,
    "l2_reg": 1e-5,
    "weights": {"wN": 1.0, "w0": 1.0, "w1": 1.0, "w2": 2.0, "wW": 0.5},
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


def se_block(x, ratio=16):
    filters = x.shape[-1]
    se = GlobalAveragePooling2D()(x)
    se = Dense(filters // ratio, activation="relu")(se)
    se = Dense(filters, activation="sigmoid")(se)
    return Multiply()([x, se])


def build_model(
    shape: Tuple[int] = (120, 240, 2),
    c_shape: Tuple[int] = (120, 240, 2),
    attention_dropout: float = 0.2,  # <--- New
    input_variables: List[str] = ALL_VARIABLES,
    start_filters: int = 64,
    l2_reg: float = 0.001,
    dropout_rate: float = 0.1,
) -> keras.Model:
    background_flag = -3.0  # Constant
    include_range_folded = True  # Always True
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

    x, c = wide_resnet_block(
        x=x,
        c=coords,
        stride=1,
        filters=start_filters,
        l2_reg=l2_reg,
        drop_rate=dropout_rate,
    )
    x = se_block(x)

    x = Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    attn = Conv2D(1, 1, activation="sigmoid", use_bias=False, name="attention_map")(x)
    attn = Dropout(rate=attention_dropout, name="attention_dropout")(attn)
    x = Multiply()([x, attn])

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


def wide_resnet_block(
    x, c, filters, stride, l2_reg=1e-4, drop_rate=0.0, project_shortcut=False
):
    shortcut_x, shortcut_c = x, c
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x, c = CoordConv2D(
        filters,
        kernel_size=3,
        strides=stride,
        padding="same",
        activation=None,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
    )([x, c])

    if drop_rate > 0:
        x = Dropout(drop_rate)(x)

    x = BatchNormalization()(x)
    x = ReLU()(x)
    x, c = CoordConv2D(
        filters,
        kernel_size=3,
        strides=1,
        padding="same",
        activation=None,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
    )([x, c])

    if project_shortcut or shortcut_x.shape[-1] != filters or stride != 1:
        shortcut_x, c = CoordConv2D(
            filters,
            kernel_size=1,
            strides=stride,
            padding="same",
            activation=None,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
        )([shortcut_x, shortcut_c])

    x = Add()([x, shortcut_x])
    return x, c


def objective(trial):
    config = DEFAULT_CONFIG.copy()
    config["learning_rate"] = trial.suggest_float("learning_rate", 5e-4, 5e-3, log=True)
    config["l2_reg"] = trial.suggest_float("l2_reg", 1e-10, 1e-5, log=True)
    config["label_smoothing"] = trial.suggest_float("label_smoothing", 0.0, 0.1)
    config["dropout_rate"] = trial.suggest_float("dropout_rate", 0.08, 0.15)
    config["warmup_epochs"] = trial.suggest_int("warmup_epochs", 1, 5)
    config["t_mul"] = trial.suggest_categorical("t_mul", [1.0, 2.0, 2.5])
    config["m_mul"] = trial.suggest_float("m_mul", 0.8, 1.0)
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
        start_filters=config["start_filters"],
        dropout_rate=config["dropout_rate"],
    )

    loss = keras.losses.BinaryCrossentropy(label_smoothing=config["label_smoothing"])
    steps_per_epoch = len(ds_train)
    print(steps_per_epoch)
    total_epochs = config["epochs"]
    total_steps = steps_per_epoch * total_epochs
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


from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.trial import TrialState, create_trial

if __name__ == "__main__":
    manual_trials = [
        {
            "learning_rate": 0.001,
            "l2_reg": 1e-12,
            "label_smoothing": 0.000,
            "dropout_rate": 0.100,
            "warmup_epochs": 3,
            "attention_dropout": 0.2,
            "t_mul": 2.0,
            "m_mul": 0.9,
            "value": 0.600987,
        },
        {
            "learning_rate": 0.005,
            "l2_reg": 1e-5,
            "label_smoothing": 0.050,
            "dropout_rate": 0.100,
            "warmup_epochs": 3,
            "attention_dropout": 0.2,
            "t_mul": 2.0,
            "m_mul": 0.9,
            "value": 0.609940,
        },
        {
            "learning_rate": 0.000697,
            "l2_reg": 1.44e-08,
            "label_smoothing": 0.005,
            "dropout_rate": 0.103,
            "warmup_epochs": 3,
            "attention_dropout": 0.2,
            "t_mul": 2.0,
            "m_mul": 0.9,
            "value": 0.600814,
        },
        {
            "learning_rate": 0.000963,
            "l2_reg": 5.38e-12,
            "label_smoothing": 0.089,
            "dropout_rate": 0.107,
            "warmup_epochs": 3,
            "t_mul": 2.0,
            "attention_dropout": 0.2,
            "m_mul": 0.9,
            "value": 0.604373,
        },
        {
            "learning_rate": 0.00487,
            "l2_reg": 1.4e-6,
            "label_smoothing": 0.0807,
            "dropout_rate": 0.122,
            "warmup_epochs": 4,
            "t_mul": 2.0,
            "m_mul": 0.905,
            "attention_dropout": 0.341,
            "value": 0.6105,
        },
        {
            "learning_rate": 0.00441,
            "l2_reg": 3.83e-7,
            "label_smoothing": 0.0845,
            "dropout_rate": 0.0997,
            "warmup_epochs": 4,
            "t_mul": 2.0,
            "m_mul": 0.809,
            "attention_dropout": 0.322,
            "value": 0.6278,
        },
    ]
    distributions = {
        "learning_rate": FloatDistribution(5e-4, 5e-3, log=True),
        "l2_reg": FloatDistribution(1e-12, 1e-5, log=True),
        "dropout_rate": FloatDistribution(0.08, 0.15),
        "label_smoothing": FloatDistribution(0.0, 0.1),
        "warmup_epochs": IntDistribution(1, 5),
        "t_mul": CategoricalDistribution([1.0, 2.0, 2.5]),
        "m_mul": FloatDistribution(0.8, 1.0),
        "attention_dropout": FloatDistribution(0.0, 0.4),
    }

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=SEED, n_startup_trials=0, multivariate=True
        ),
        pruner=optuna.pruners.HyperbandPruner(),
    )
    for trial in manual_trials:
        trial_data = create_trial(
            params={
                k: trial[k]
                for k in [
                    "learning_rate",
                    "l2_reg",
                    "dropout_rate",
                    "label_smoothing",
                    "warmup_epochs",
                    "t_mul",
                    "m_mul",
                    "attention_dropout",
                ]
            },
            distributions=distributions,
            value=trial["value"],
            state=TrialState.COMPLETE,
        )
        study.add_trial(trial_data)

    study.optimize(objective, n_trials=50)

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
