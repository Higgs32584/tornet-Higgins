import sys
import os
import json
import shutil
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    BatchNormalization,
    Add,
    MaxPool2D,
    Dropout,
    GlobalAveragePooling2D,
    Dense,
    Multiply,
    Conv2D,
    GlobalMaxPooling2D,
    Reshape,
    ReLU,
)
import optuna
from optuna.integration import TFKerasPruningCallback
from datetime import datetime
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
STUDY_DIR = os.path.join("optuna_studies", f"study_{timestamp}")
os.makedirs(STUDY_DIR, exist_ok=True)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from typing import List, Tuple
from tornet.data.loader import get_dataloader
from tornet.data import preprocess as pp
from tornet.data.preprocess import get_shape
from tornet.data.constants import ALL_VARIABLES, CHANNEL_MIN_MAX
from tornet.metrics.keras import metrics as tfm
from tornet.utils.general import make_exp_dir
from custom_func import FalseAlarmRate, ThreatScore
import tensorflow_datasets as tfds
from tornet.models.keras.layers import CoordConv2D
import tornet.data.tfds.tornet.tornet_dataset_builder
import random

logging.basicConfig(level=logging.ERROR)
SEED = 42  
# Set random seeds for reproducibility
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# Environment Variables
DATA_ROOT = '/home/ubuntu/tfds'
TORNET_ROOT=DATA_ROOT
TFDS_DATA_DIR = DATA_ROOT
EXP_DIR = "."
os.environ['TORNET_ROOT'] = DATA_ROOT
os.environ['TFDS_DATA_DIR'] = TFDS_DATA_DIR
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
tf.config.optimizer.set_jit(True)


logging.info(f'TORNET_ROOT={DATA_ROOT}')

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


def build_model(model='v7', shape: Tuple[int] = (120, 240, 2),
                c_shape: Tuple[int] = (120, 240, 2),
                input_variables: List[str] = ALL_VARIABLES,
                start_filters: int = 96,  # Bumped up from 64 for added capacity
                nconvs: int = 2,
                l2_reg: float = 0.001,
                background_flag: float = -3.0,
                include_range_folded: bool = True,
                dropout_rate: float = 0.1, **config):
    
    inputs = {v: keras.Input(shape, name=v) for v in input_variables}
    n_sweeps = shape[2]

    # Normalize and concatenate
    x = keras.layers.Concatenate(axis=-1, name='Concatenate1')(
        [normalize(inputs[v], v) for v in input_variables]
    )
    x = FillNaNs(background_flag)(x)

    if include_range_folded:
        range_folded = keras.Input(shape[:2] + (n_sweeps,), name='range_folded_mask')
        inputs['range_folded_mask'] = range_folded
        x = keras.layers.Concatenate(axis=-1, name='Concatenate2')([x, range_folded])

    cin = keras.Input(c_shape, name='coordinates')
    inputs['coordinates'] = cin

    # === Dual path structure ===

    # ResNet-style path
    path_a, c_out = wide_resnet_block(x, cin, filters=start_filters, widen_factor=1,
                                      l2_reg=l2_reg, nconvs=nconvs, drop_rate=dropout_rate)

    # Lightweight single conv path
    path_b, _ = CoordConv2D(filters=start_filters, kernel_size=3, padding="same",
                            kernel_regularizer=keras.regularizers.l2(l2_reg),
                            activation="relu")([x, cin])
    path_b = MaxPool2D(pool_size=2, strides=2, padding='same')(path_b)

    # Fuse paths
    x = keras.layers.Concatenate(axis=-1, name='fuse_paths')([path_a, path_b])

    # Post-fusion conv block
    x = Conv2D(start_filters, 3, padding='same', activation='relu',
               kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Squeeze-and-Excite block
    x = se_block(x, ratio=16, name="se_fused")

    # Optional spatial attention
    attention_map = Conv2D(1, 1, activation='sigmoid', name='spatial_attention_map')(x)
    attention_map = Dropout(rate=dropout_rate, name='spatial_attention_dropout')(attention_map)
    x = Multiply(name='apply_spatial_attention')([x, attention_map])

    # Head conv
    x = Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(rate=dropout_rate)(x)

    # Global avg + max pooling
    x_avg = GlobalAveragePooling2D()(x)
    x_max = GlobalMaxPooling2D()(x)
    x = keras.layers.Concatenate(name="global_pool_concat")([x_avg, x_max])

    # Dense head
    x = Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = Dropout(dropout_rate // 2)(x)

    output = Dense(1, activation='sigmoid', dtype='float32')(x)

    return keras.Model(inputs=inputs, outputs=output, name='v7_model')



def se_block(x, ratio=16, name=None):
    filters = x.shape[-1]
    
    # Squeeze
    se = GlobalAveragePooling2D(name=f"{name}_gap" if name else None)(x)

    # Excite
    se = Dense(filters // ratio, activation="relu", name=f"{name}_fc1" if name else None)(se)
    se = Dense(filters, activation="sigmoid", name=f"{name}_fc2" if name else None)(se)

    # Explicit broadcast shape for Multiply
    se = Reshape((1, 1, filters), name=f"{name}_reshape" if name else None)(se)

    # Multiply
    x = Multiply(name=f"{name}_scale" if name else None)([x, se])
    return x


def wide_resnet_block(x, c, filters=64, widen_factor=2, l2_reg=1e-6, drop_rate=0.1,nconvs=2):
    """Wide ResNet Block with CoordConv2D"""
    shortcut_x, shortcut_c = x, c  # Skip connection

    # 3x3 CoordConv2D (Wider filters)
    for i in range(nconvs):
        x = BatchNormalization()(x)
        x= ReLU()(x)
        x, c = CoordConv2D(filters=filters * widen_factor, kernel_size=3, padding="same",
                        kernel_regularizer=keras.regularizers.l2(l2_reg),
                        activation=None)([x, c])
    # Skip Connection
    shortcut_x, shortcut_c = CoordConv2D(filters=filters * widen_factor, kernel_size=1, padding="same",
                                         kernel_regularizer=keras.regularizers.l2(l2_reg),
                                         activation=None)([shortcut_x, shortcut_c])
    

    # Add Residual Connection
    x = ReLU()(x)
    x = Add()([x, shortcut_x])

    # Pooling and dropout
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    c = MaxPool2D(pool_size=2, strides=2, padding='same')(c)

    if drop_rate > 0:
        x = Dropout(rate=drop_rate)(x)
    
    return x, c



@keras.utils.register_keras_serializable()
class FastNormalize(keras.layers.Layer):
    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        self.mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        self.std = tf.convert_to_tensor(std, dtype=tf.float32)
        self._mean_list = mean.numpy().tolist() if hasattr(mean, 'numpy') else list(mean)
        self._std_list = std.numpy().tolist() if hasattr(std, 'numpy') else list(std)

    def call(self, x):
        return tf.math.subtract(x, self.mean) / (self.std + 1e-6)

    def get_config(self):
        config = super().get_config()
        config.update({
            "mean": self._mean_list,
            "std": self._std_list,
        })
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
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


DEFAULT_CONFIG = {
    "epochs": 100,
    "input_variables": ["DBZ", "VEL", "KDP", "ZDR", "RHOHV", "WIDTH"],
    "train_years": [2013, 2014, 2015],
    "val_years": [2021, 2022],
    "batch_size": 128,
    "model": "wide_resnet",
    "start_filters": 96,
    "learning_rate": 0.000368,
    "decay_steps": 1386,
    "decay_rate": 0.952,
    "l2_reg": 4.33e-06,
    "weights":{
    "wN": 0.85,
    "w0": 0.97,
    "w1": 1.13,
    "w2": 1.6,
    "wW": 1.3,
    },
    "nconvs": 2,
    "dropout_rate": 0.143,
    "loss": "combo",  # << Changed from "cce"
    "head": "maxpool",
    "exp_name": "tornado_v7_boosted",
    "exp_dir": ".",
    "dataloader": "tensorflow-tfds",
    "label_smoothing":0.065,
    "dataloader_kwargs": {
        "select_keys": ["DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH", "range_folded_mask", "coordinates"]
    }
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

    config.update({
        # Model capacity & regularization
        "dropout_rate": round(trial.suggest_float("dropout_rate", 0.05, 0.15),5),
        "learning_rate": round(trial.suggest_float("learning_rate", 1e-6, 1e-5, log=True),5),
        "l2_reg": round(trial.suggest_float("l2_reg", 1e-6, 1e-5, log=True),5),
        "decay_steps": trial.suggest_int("decay_steps", 800, 1200),
        "weights":{
                "wN": round(trial.suggest_float("wN", 0.59, 1.1),5),
                "w0": round(trial.suggest_float("w0", 0.68, 1.26),5),
                "w1": round(trial.suggest_float("w1", 0.79, 1.47),5),
                "w2": round(trial.suggest_float("w2", 1.12, 2.08),5),
                "wW": round(trial.suggest_float("wW", 0.8, 1.5),5),
        },

        "label_smoothing": round(trial.suggest_float("label_smoothing", 0.05, 0.20),5),
    })
    ds_train = get_dataloader(config['dataloader'], DATA_ROOT, config['train_years'], "train",
                                config['batch_size'], config['weights'], **config['dataloader_kwargs'])
    ds_val = get_dataloader(config['dataloader'], DATA_ROOT, config['val_years'], "train",
                            config['batch_size'], {'wN': 1.0, 'w0': 1.0, 'w1': 1.0, 'w2': 1.0, 'wW': 1.0},
                            **config['dataloader_kwargs'])

    x, _, _ = next(iter(ds_train))
    in_shapes = (None, None, get_shape(x)[-1])
    c_shapes = (None, None, x["coordinates"].shape[-1])
    model = build_model(
        shape=in_shapes,
        c_shape=c_shapes,
        **config
    )

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(config['learning_rate'], config['decay_steps'], config['decay_rate'])
    opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    loss = keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=config["label_smoothing"])

    try:
        model.compile(
            loss=loss,
            optimizer=opt,
            metrics=[
            keras.metrics.AUC(from_logits=False, curve='PR', name='AUCPR', num_thresholds=1000),
            tfm.F1Score(from_logits=False, name='F1'),
            ],
            jit_compile=True
        )

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_AUCPR', patience=2, mode='max', restore_best_weights=True)
        prune_cb = TFKerasPruningCallback(trial, 'val_AUCPR')
        history = model.fit(ds_train, epochs=config['epochs'], validation_data=ds_val, verbose=1,callbacks=[early_stopping, prune_cb])
        best_aucpr = max(history.history.get('val_AUCPR', [0]))
        log_trial(trial, best_aucpr)
        return best_aucpr
    except tf.errors.ResourceExhaustedError as e:
        # Handle OOM error: log it and return a "failed" result to Optuna (nan)
        print(f"Out of memory error occurred in trial {trial.number}. Skipping this trial.")
        log_trial(trial, float('nan'))  # Optionally log 'nan' to indicate failure
        return float('nan')  # Return NaN to indicate the trial failed

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

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(multivariate=True,seed=SEED),
                                 pruner=optuna.pruners.HyperbandPruner())
    study.optimize(objective, n_trials=150)

    best_trial_data = {
        "best_trial_id": study.best_trial.number,
        "best_params": study.best_trial.params,
        "best_value": study.best_trial.value,
        "DEFAULT_CONFIG": DEFAULT_CONFIG.copy()
    }

    best_params_path = os.path.join(STUDY_DIR, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(best_trial_data, f, indent=4)

    print(f"Best trial saved to: {best_params_path}")
    print("Best trial:", study.best_trial)