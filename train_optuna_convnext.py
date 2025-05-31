import json
import logging
import os
import random
import shutil
import sys
from typing import List, Tuple
import datetime
import numpy as np
from datetime import datetime
from tensorflow.keras import backend as K
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from optuna.integration import TFKerasPruningCallback
from tensorflow.keras.layers import (
    GlobalMaxPooling2D,
    Conv2D,
    Dense,
    Dropout,
    Reshape,
    Multiply,
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
# SEED = 98
# os.environ["PYTHONHASHSEED"] = str(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)
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


def convnext_block(
    x, activation="gelu", drop_rate=0.0, layer_scale_init_value=1e-6, name=None
):
    input_channels = x.shape[-1]
    shortcut = x
    x = DepthwiseConv2D(kernel_size=7, padding="same", name=f"{name}_dwconv")(x)
    x = LayerNormalization(epsilon=1e-6, name=f"{name}_ln")(x)
    x = Dense(4 * input_channels, name=f"{name}_pw1")(x)
    x = Activation(activation, name=f"{name}_{activation}")(x)
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


def build_model(
    shape: Tuple[int] = (120, 240, 2),
    c_shape: Tuple[int] = (120, 240, 2),
    activation="gelu",
    input_variables: List[str] = ALL_VARIABLES,
    use_attn: bool = False,
    dense_filters: int = 64,
    start_filters: int = 64,
    dropout_rate: float = 0.1,
) -> keras.Model:
    background_flag = -3.0
    include_range_folded = True
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
    x = Conv2D(start_filters, kernel_size=3, strides=1, name="stem_conv")(x)
    x = LayerNormalization(epsilon=1e-6, name="stem_ln")(x)
    x = convnext_block(
        x, drop_rate=dropout_rate, name="convnext_block", activation=activation
    )
    if use_attn:
        x = se_block(x)
    x_avg = GlobalAveragePooling2D()(x)
    x_max = GlobalMaxPooling2D()(x)
    x = keras.layers.Concatenate()([x_avg, x_max])
    x = LayerNormalization(epsilon=1e-6, name="head_ln")(x)
    x = Dense(dense_filters, activation=activation, name="head_dense")(x)
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


# Enable GPU Memory Growth
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


DEFAULT_CONFIG = {
    "epochs": 50,
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
    "weights": {"wN": 1.0, "w0": 1.0, "w1": 1.0, "w2": 2.0, "wW": 0.5},
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


def objective(trial):
    try:
        config = DEFAULT_CONFIG.copy()
        config["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-6, 1e-3, log=True
        )
        config["dropout_rate"] = trial.suggest_float("dropout_rate", 0.0, 0.3)
        config["start_filters"] = trial.suggest_categorical(
            "start_filters", [24, 48, 96, 192]
        )
        config["dense_filters"] = trial.suggest_categorical(
            "dense_filters", [32, 64, 128, 256]
        )
        config["activation"] = trial.suggest_categorical(
            "activation", ["relu", "gelu", "swish"]
        )

        config["label_smoothing"] = trial.suggest_float("label_smoothing", 0.0, 0.1)
        config["warmup_epochs"] = trial.suggest_int("warmup_epochs", 1, 5)
        config["t_mul"] = trial.suggest_categorical("t_mul", [1.0, 2.0, 2.5])
        config["use_attn"] = trial.suggest_categorical("use_attn", [True, False])
        config["m_mul"] = trial.suggest_float("m_mul", 0.8, 1.0)

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
            activation=config["activation"],
            dense_filters=config["dense_filters"],
            use_attn=config["use_attn"],
            input_variables=config["input_variables"],
            start_filters=config["start_filters"],
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
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_AUCPR", patience=5, mode="max", restore_best_weights=True
        )
        pruning_callback = TFKerasPruningCallback(trial, monitor="val_AUCPR")

        history = model.fit(
            ds_train,
            epochs=config["epochs"],
            validation_data=ds_val,
            verbose=1,
            callbacks=[early_stopping, pruning_callback],
        )
        best_aucpr = max(history.history.get("val_AUCPR", [0]))
        log_trial(trial, best_aucpr)
        tf.keras.backend.clear_session()
        import gc

        gc.collect()
        K.clear_session()

        return best_aucpr
    except tf.errors.ResourceExhaustedError as e:
        print(f"Trial {trial.number} pruned due to resource exhaustion: {e}")
        tf.keras.backend.clear_session()
        K.clear_session()
        import gc

        gc.collect()
        raise optuna.exceptions.TrialPruned()

    except Exception as e:
        # Optional: prune on any unexpected issue during model build
        print(f"Trial {trial.number} failed: {e}")
        tf.keras.backend.clear_session()
        import gc

        K.clear_session()
        gc.collect()
        raise optuna.exceptions.TrialPruned()


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
STUDY_DIR = os.path.join("optuna_studies", f"study_{timestamp}")
os.makedirs(STUDY_DIR, exist_ok=True)

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


import optuna
from optuna.trial import TrialState, create_trial

if __name__ == "__main__":

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(multivariate=True),
        pruner=optuna.pruners.HyperbandPruner(),
    )
    import optuna
    from optuna.trial import TrialState
    import datetime

    # Assume you already have a study
    study = optuna.create_study(direction="maximize")
    existing_trials = [
        {
            "trial_id": 0,
            "params": {
                "learning_rate": 0.0003,
                "dropout_rate": 0.15,
                "start_filters": 48,
                "dense_filters": 64,
                "activation": "gelu",
                "label_smoothing": 0.1,
                "warmup_epochs": 3,
                "t_mul": 2.0,
                "use_attn": True,
                "m_mul": 0.9,
            },
            "value": 0.5138369798660278,
        },
        {
            "trial_id": 1,
            "params": {
                "learning_rate": 1.492355140925348e-05,
                "dropout_rate": 0.20044736842533462,
                "start_filters": 24,
                "dense_filters": 128,
                "activation": "gelu",
                "label_smoothing": 0.07340961990537682,
                "warmup_epochs": 3,
                "t_mul": 2.5,
                "use_attn": False,
                "m_mul": 0.9665257780055175,
            },
            "value": 0.3623429834842682,
        },
        {
            "trial_id": 3,
            "params": {
                "learning_rate": 1.286038977952338e-05,
                "dropout_rate": 0.016257018332804993,
                "start_filters": 24,
                "dense_filters": 256,
                "activation": "swish",
                "label_smoothing": 0.062031871838148706,
                "warmup_epochs": 3,
                "t_mul": 2.0,
                "use_attn": True,
                "m_mul": 0.8629068247752655,
            },
            "value": 0.34078744053840637,
        },
    ]
    from optuna.distributions import (
        FloatDistribution,
        IntDistribution,
        CategoricalDistribution,
    )

    distributions = {
        "learning_rate": FloatDistribution(1e-6, 1e-3, log=True),
        "dropout_rate": FloatDistribution(0.0, 0.3),
        "start_filters": CategoricalDistribution([24, 48, 96, 192]),
        "dense_filters": CategoricalDistribution([32, 64, 128, 256]),
        "activation": CategoricalDistribution(["relu", "gelu", "swish"]),
        "label_smoothing": FloatDistribution(0.0, 0.1),
        "warmup_epochs": IntDistribution(1, 5),
        "t_mul": CategoricalDistribution([1.0, 2.0, 2.5]),
        "use_attn": CategoricalDistribution([True, False]),
        "m_mul": FloatDistribution(0.8, 1.0),
    }

    for trial in existing_trials:
        trial_obj = create_trial(
            value=trial["value"],
            state=TrialState.COMPLETE,
            params=trial["params"],
            distributions=distributions,
            user_attrs={},
            system_attrs={},
            intermediate_values={},
        )
        study.add_trial(trial_obj)

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
