import sys
import os
import json
import shutil
import logging
import numpy as np
import tensorflow as tf
import optuna
from tensorflow import keras
from tensorflow.keras.layers import (
    BatchNormalization, Add, MaxPool2D, Dropout, GlobalAveragePooling2D,
    Dense, Multiply, Conv2D, GlobalMaxPooling2D, Reshape, ReLU, Conv1D, Flatten
)
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from typing import List, Tuple
from tornet.data.loader import get_dataloader
from tornet.data import preprocess as pp
from tornet.data.preprocess import get_shape
from tornet.data.constants import ALL_VARIABLES, CHANNEL_MIN_MAX
from tornet.metrics.keras import metrics as tfm
from tornet.utils.general import make_exp_dir
from custom_func import FalseAlarmRate, ThreatScore
from tornet.models.keras.layers import CoordConv2D
import random
from optuna.integration import TFKerasPruningCallback

# Set up logging and environment
logging.basicConfig(level=logging.ERROR)
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TORNET_ROOT'] = '/home/ubuntu/tfds'
os.environ['TFDS_DATA_DIR'] = '/home/ubuntu/tfds'
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
DATA_ROOT = '/home/ubuntu/tfds'

# GPU config
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

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

@keras.utils.register_keras_serializable()
class FastNormalize(keras.layers.Layer):
    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        self.mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        self.std = tf.convert_to_tensor(std, dtype=tf.float32)
        self._mean_list = list(mean)
        self._std_list = list(std)

    def call(self, x):
        return tf.math.subtract(x, self.mean) / (self.std + 1e-6)

    def get_config(self):
        config = super().get_config()
        config.update({"mean": self._mean_list, "std": self._std_list})
        return config

def normalize(x, name: str):
    min_val, max_val = CHANNEL_MIN_MAX[name]
    mean = np.float32((max_val + min_val) / 2)
    std = np.float32((max_val - min_val) / 2)
    n_sweeps = x.shape[-1]
    mean = tf.constant([mean] * n_sweeps, dtype=tf.float32)
    std = tf.constant([std] * n_sweeps, dtype=tf.float32)
    return FastNormalize(mean, std)(x)

@keras.utils.register_keras_serializable()
class ExpandDimsTwice(keras.layers.Layer):
    def call(self, inputs):
        return tf.expand_dims(tf.expand_dims(inputs, axis=1), axis=1)

@keras.utils.register_keras_serializable()
class StackAvgMax(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.stack(inputs, axis=1)

def se_block(x, ratio=16, name=None):
    filters = x.shape[-1]
    se = GlobalAveragePooling2D(name=f"{name}_gap" if name else None)(x)
    se = Dense(filters // ratio, activation="relu", name=f"{name}_fc1" if name else None)(se)
    se = Dense(filters, activation="sigmoid", name=f"{name}_fc2" if name else None)(se)
    se = Reshape((1, 1, filters), name=f"{name}_reshape" if name else None)(se)
    return Multiply(name=f"{name}_scale" if name else None)([x, se])

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


def build_model(model='wide_resnet', shape=(120,240,2), c_shape=(120,240,2),
                input_variables=ALL_VARIABLES, start_filters=64, nconvs=2,
                l2_reg=0.001, background_flag=-3.0, include_range_folded=True, dropout_rate=0.1):
    inputs = {v: keras.Input(shape, name=v) for v in input_variables}
    normalized_inputs = keras.layers.Concatenate(axis=-1, name='Concatenate1')([normalize(inputs[v], v) for v in input_variables])
    normalized_inputs = FillNaNs(background_flag)(normalized_inputs)

    if include_range_folded:
        range_folded = keras.Input(shape[:2] + (shape[2],), name='range_folded_mask')
        inputs['range_folded_mask'] = range_folded
        normalized_inputs = keras.layers.Concatenate(axis=-1, name='Concatenate2')([normalized_inputs, range_folded])

    cin = keras.Input(c_shape, name='coordinates')
    inputs['coordinates'] = cin

    gate_feat, c = CoordConv2D(filters=start_filters * 2, kernel_size=3, padding="same",
                               kernel_regularizer=keras.regularizers.l2(l2_reg),activation=None)([normalized_inputs, cin])
    gate_pool = GlobalMaxPooling2D()(gate_feat)
    gate_score = Dense(1, activation='sigmoid', name='gate_score')(gate_pool)
    gate_score = ExpandDimsTwice(name='expand_dims_twice')(gate_score)

    easy_path, _ = wide_resnet_block(normalized_inputs, cin, filters=start_filters, widen_factor=2,
                                     l2_reg=l2_reg, nconvs=nconvs, drop_rate=dropout_rate)
    hard_path, _ = wide_resnet_block(normalized_inputs, cin, filters=start_filters, widen_factor=2,
                                     l2_reg=l2_reg, nconvs=nconvs*2, drop_rate=dropout_rate)
    x = gate_score * hard_path + (1.0 - gate_score) * easy_path
    x = se_block(x)
    x = Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    attention_map = Conv2D(1, 1, activation='sigmoid', name='attention_map')(x)
    attention_map = Dropout(rate=0.2, name='attention_dropout')(attention_map)
    x_weighted = Multiply()([x, attention_map])
    x_avg = GlobalAveragePooling2D()(x_weighted)
    x_max = GlobalMaxPooling2D()(x_weighted)
    x_stack = StackAvgMax(name="stack_avg_max")([x_avg, x_max])
    x_fused = Conv1D(64, 1, activation='relu')(x_stack)
    x_fused = Flatten()(x_fused)
    x_dense = Dense(64, activation='relu')(x_fused)
    output = Dense(1, activation='sigmoid', dtype='float32')(x_dense)
    return keras.Model(inputs=inputs, outputs=output)

DEFAULT_CONFIG = {
    "epochs": 100,
    "input_variables": ["DBZ", "VEL", "KDP", "ZDR", "RHOHV", "WIDTH"],
    "train_years": [2013, 2014, 2015, 2016],
    "val_years": [2017],
    "batch_size": 128,
    "model": "wide_resnet",
    "start_filters": 48,
    "learning_rate": 1e-4,
    "decay_steps": 1386,
    "decay_rate": 0.958,
    "l2_reg": 1e-5,
    "wN": 1.0, "w0": 1.0, "w1": 1.0, "w2": 2.0, "wW": 0.5,
    "nconvs": 2,
    "dropout_rate": 0.1,
    "loss": "cce", "head": "maxpool",
    "exp_name": "tornado_baseline",
    "exp_dir": ".", "dataloader": "tensorflow-tfds",
    "dataloader_kwargs": {
        "select_keys": ["DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH", "range_folded_mask", "coordinates"]
    }
}

def main(config, trial=None):
    dataloader_kwargs = config['dataloader_kwargs']
    dataloader_kwargs.update({'select_keys': config['input_variables'] + ['range_folded_mask', 'coordinates']})
    ds_train = get_dataloader(config['dataloader'], DATA_ROOT, config['train_years'], "train", config['batch_size'],
                              {'wN': config['wN'], 'w0': config['w0'], 'w1': config['w1'], 'w2': config['w2'], 'wW': config['wW']},
                              **dataloader_kwargs)
    ds_val = get_dataloader(config['dataloader'], DATA_ROOT, config['val_years'], "train", config['batch_size'],
                            {'wN':1.0,'w0':1.0,'w1':1.0,'w2':1.0,'wW':1.0}, **dataloader_kwargs)
    x, _, _ = next(iter(ds_train))
    in_shapes = (120, 240, get_shape(x)[-1])
    c_shapes = (120, 240, x["coordinates"].shape[-1])
    nn = build_model(model=config['model'], shape=in_shapes, c_shape=c_shapes,
                     input_variables=config['input_variables'],
                     start_filters=config['start_filters'],
                     l2_reg=config['l2_reg'],
                     dropout_rate=config['dropout_rate'],
                     nconvs=config['nconvs'])
    lr_schedule = ExponentialDecay(config['learning_rate'], config['decay_steps'], config['decay_rate'])
    opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    loss = keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.1)
    metrics = [
        keras.metrics.AUC(from_logits=False, curve='PR', name='AUCPR', num_thresholds=1000),
        tfm.BinaryAccuracy(from_logits=False), tfm.TruePositives(from_logits=False),
        tfm.FalsePositives(from_logits=False), tfm.TrueNegatives(from_logits=False),
        tfm.FalseNegatives(from_logits=False), tfm.Precision(from_logits=False),
        tfm.Recall(from_logits=False), tfm.F1Score(from_logits=False)]
    nn.compile(loss=loss, optimizer=opt, metrics=metrics, jit_compile=True)
    expdir = make_exp_dir(exp_dir=config['exp_dir'], prefix=config['exp_name'])
    callbacks = [
        keras.callbacks.CSVLogger(os.path.join(expdir, 'log.csv')),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.EarlyStopping(monitor='val_AUCPR', patience=3, mode='max', restore_best_weights=True)
    ]
    if trial is not None:
        callbacks.append(TFKerasPruningCallback(trial, "val_AUCPR"))
    history = nn.fit(ds_train, validation_data=ds_val, epochs=config['epochs'], callbacks=callbacks, verbose=1)
    return {'AUCPR': max(history.history['val_AUCPR'])}

def objective(trial):
    config = DEFAULT_CONFIG.copy()
    config.update({
        'start_filters': trial.suggest_categorical("start_filters", [32, 48, 64]),
        'learning_rate': trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        'dropout_rate': trial.suggest_float("dropout_rate", 0.0, 0.5),
        'l2_reg': trial.suggest_float("l2_reg", 1e-6, 1e-3, log=True),
        'nconvs': trial.suggest_int("nconvs", 1, 2),
        'exp_name': f"optuna_trial_{trial.number}",
        'exp_dir': "./optuna_runs"
    })
    return main(config, trial)['AUCPR']

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "tune":
        study = optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(multivariate=True))
        study.optimize(objective, n_trials=100)
        print("\nBest Trial:")
        print(f"  AUCPR: {study.best_value}")
        for key, val in study.best_trial.params.items():
            print(f"    {key}: {val}")
    else:
        config = DEFAULT_CONFIG.copy()
        if len(sys.argv) > 1:
            config.update(json.load(open(sys.argv[1])))
        main(config)
