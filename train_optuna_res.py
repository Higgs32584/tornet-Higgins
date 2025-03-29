import sys
import os
import json
import logging
import optuna
import keras
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import List, Tuple
from tornet.data.loader import get_dataloader
from tornet.data.preprocess import get_shape
from tornet.data.constants import ALL_VARIABLES, CHANNEL_MIN_MAX
from tornet.models.keras.losses import mae_loss
from tornet.metrics.keras import metrics as tfm
from tornet.utils.general import make_exp_dir
from tornet.models.keras.layers import FillNaNs
from tornet.models.keras.cnn_baseline import vgg_block,normalize
from tornet.models.keras.layers import CoordConv2D, FillNaNs
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    Add,
    Activation,
    MaxPool2D,
    Dropout,
    Lambda,
    Multiply,
    GlobalAveragePooling2D
)

from keras.optimizers.schedules import CosineDecayRestarts

logging.basicConfig(level=logging.INFO)
SEED = 42  

# Set random seeds for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Environment Variables
DATA_ROOT = '/home/ubuntu/tfds'
TFDS_DATA_DIR = DATA_ROOT
os.environ['TORNET_ROOT'] = DATA_ROOT
os.environ['TFDS_DATA_DIR'] = TFDS_DATA_DIR

# Enable GPU Memory Growth
gpus = tf.config.list_physical_devices('GPU')
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
    "model": "wide_resnet",
    "start_filters": 112,
    "learning_rate": 1.329291894316217e-05,
    "l2_reg":  0.0007114476009343421,
    "weight_decay": 6.251373574521755e-05,
    "first_decay_steps":20,
    "train_years": [2016],
    "val_years": [2017],
    "input_variables": ["DBZ", "VEL", "KDP", "ZDR", "RHOHV", "WIDTH"],
    "exp_name": "tornado_baseline",
    "exp_dir": STUDY_DIR,  # Save study results here
    "dataloader": "tensorflow-tfds",
    "dataloader_kwargs": {
        "select_keys": ["DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH", "range_folded_mask", "coordinates"]
    },
    "dropout_rate": 0.08900466011060913,
    "weights":{
        "wN": 0.0996807409488512,
        "w0": 33.0,
        "w1": 63,
        "w2": 5,
        "wW": 51
    }

}
def se_block(x, ratio=16):
    filters = x.shape[-1]
    se = GlobalAveragePooling2D()(x)
    se = Dense(filters // ratio, activation="relu")(se)
    se = Dense(filters, activation="sigmoid")(se)
    return Multiply()([x, se])


def build_model(shape:Tuple[int]=(120,240,2),
                c_shape:Tuple[int]=(120,240,2),
                input_variables:List[str]=ALL_VARIABLES,
                start_filters:int=64,
                l2_reg:float=0.001,
                background_flag:float=-3.0,
                include_range_folded:bool=True,
                model='vgg',dropout_rate=0.0):
    # Create input layers for each input_variables
    inputs = {}
    for v in input_variables:
        inputs[v]=keras.Input(shape,name=v)
    n_sweeps=shape[2]
    
    # Normalize inputs and concate along channel dim
    normalized_inputs=keras.layers.Concatenate(axis=-1,name='Concatenate1')(
        [normalize(inputs[v],v) for v in input_variables]
        )

    # Replace nan pixel with background flag
    normalized_inputs = FillNaNs(background_flag)(normalized_inputs)

    # Add channel for range folded gates
    if include_range_folded:
        range_folded = keras.Input(shape[:2]+(n_sweeps,),name='range_folded_mask')
        inputs['range_folded_mask']=range_folded
        normalized_inputs = keras.layers.Concatenate(axis=-1,name='Concatenate2')(
               [normalized_inputs,range_folded])
        
    # Input coordinate information
    cin=keras.Input(c_shape,name='coordinates')
    inputs['coordinates']=cin
    
    x,c = normalized_inputs,cin
    
    
    if model == 'wide_resnet':
        x, c = wide_resnet_block(x, c, filters=start_filters, widen_factor=2, l2_reg=l2_reg, drop_rate=dropout_rate)
        x, c = wide_resnet_block(x, c, filters=start_filters*2, widen_factor=2, l2_reg=l2_reg, drop_rate=dropout_rate)


    x = keras.layers.Conv2D(filters=512, kernel_size=1,
                          kernel_regularizer=keras.regularizers.l2(l2_reg),
                          activation='relu')(x)
    x = keras.layers.Conv2D(filters=256, kernel_size=1,
                          kernel_regularizer=keras.regularizers.l2(l2_reg),
                          activation='relu')(x)
    x = keras.layers.Conv2D(filters=1, kernel_size=1,name='heatmap')(x)
        # Max in scene
    output = keras.layers.GlobalMaxPooling2D()(x)
        
    return keras.Model(inputs=inputs,outputs=output)


    
def wide_resnet_block(x, c, filters=64, widen_factor=2, l2_reg=1e-6, drop_rate=0.0):
    """Wide ResNet Block with CoordConv2D"""
    shortcut_x, shortcut_c = x, c  # Skip connection

    # 3x3 CoordConv2D (Wider filters)
    x, c = CoordConv2D(filters=filters * widen_factor, kernel_size=3, padding="same",
                       kernel_regularizer=keras.regularizers.l2(l2_reg),
                       activation=None)([x, c])
    x = BatchNormalization()(x)

    # 3x3 CoordConv2D (Wider filters)
    x, c = CoordConv2D(filters=filters * widen_factor, kernel_size=3, padding="same",
                       kernel_regularizer=keras.regularizers.l2(l2_reg),
                       activation=None)([x, c])
    x = BatchNormalization()(x)

    # Skip Connection
    shortcut_x, shortcut_c = CoordConv2D(filters=filters * widen_factor, kernel_size=1, padding="same",
                                         kernel_regularizer=keras.regularizers.l2(l2_reg),
                                         activation=None)([shortcut_x, shortcut_c])
    
    x = se_block(x)

    # Add Residual Connection
    x = Add()([x, shortcut_x])
    x = Activation('relu')(x)

    # Pooling and dropout
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    c = MaxPool2D(pool_size=2, strides=2, padding='same')(c)

    if drop_rate > 0:
        x = Dropout(rate=drop_rate)(x)
    
    return x, c

def objective(trial):
    config = DEFAULT_CONFIG.copy()
    config["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
    config["l2_reg"] = trial.suggest_float("l2_reg", 1e-6, 1e-3, log=True)
    config["start_filters"] = trial.suggest_int("start_filters", 8, 64, step=8)
    config["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    config["dropout_rate"] = trial.suggest_float("dropout_rate", 0.00, 0.15)
    config["t_mul"] = trial.suggest_float("t_mul", 0.5, 3.00, log=True)
    config["m_mul"] = trial.suggest_float("m_mul", 0.50, 2.00, log=True)
    config["weights"] = {
        'wN': trial.suggest_float('wN', 0.01, 1.00, log=True),
        'w0': trial.suggest_int('w0', 1.0, 10.0, log=True)*1.0,
        'w1': trial.suggest_int('w1', 1.0, 10.0, log=True)*1.0,
        'w2': trial.suggest_int('w2', 1.0, 10.0, log=True)*1.0,
        'wW': trial.suggest_int('wW', 1.0, 10.0, log=True)*1.0
    }
    config["first_decay_steps"] = trial.suggest_int("first_decay_steps", 10, 200)*1.0  # Controls decay restart interval
    logging.info(f"Tuning with config: {config}")
    config["batch_size"] = 64
    # Load dataset

    ds_train = get_dataloader(config['dataloader'], DATA_ROOT, config['train_years'], "train",
                              config['batch_size'], config['weights'], **config['dataloader_kwargs'])
    ds_val = get_dataloader(config['dataloader'], DATA_ROOT, config['val_years'], "train",
                            config['batch_size'], {'wN': 1.0,'w0': 1.0,'w1': 1.0,'w2': 1.0,'wW': 1.0}, **config['dataloader_kwargs'])

    x, _, _ = next(iter(ds_train))
    in_shapes = (None, None, get_shape(x)[-1])
    c_shapes = (None, None, x["coordinates"].shape[-1])

    model = build_model(
        model=config["model"], shape=in_shapes, c_shape=c_shapes, 
        input_variables=config['input_variables'], 
        l2_reg=config["l2_reg"], start_filters=config["start_filters"],dropout_rate=config["dropout_rate"]
    )
    loss = keras.losses.BinaryCrossentropy()
    lr = CosineDecayRestarts(initial_learning_rate=config['learning_rate'], first_decay_steps=config["first_decay_steps"],  t_mul=config["t_mul"], m_mul=config["m_mul"])
    opt = keras.optimizers.AdamW(learning_rate=lr, weight_decay=config['weight_decay'])
    from_logits=False
    metrics = [keras.metrics.AUC(from_logits=from_logits,curve='PR',name='AUCPR',num_thresholds=2000)]
    model.compile(loss=loss, optimizer=opt, metrics=metrics,jit_compile=True)

    # Train the model

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_AUCPR', patience=2, mode='max', restore_best_weights=True)
    history = model.fit(ds_train, epochs=config['epochs'], validation_data=ds_val, verbose=1, callbacks=[early_stopping])

    # Get best validation AUCPR
    best_aucpr = max(history.history.get('val_AUCPR', [0]))

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
        "DEFAULT_CONFIG": DEFAULT_CONFIG
    }
    trial_log_path = os.path.join(STUDY_DIR, f"trial_{trial.number}.json")
    with open(trial_log_path, "w") as f:
        json.dump(trial_data, f, indent=4)

if __name__ == '__main__':
    with strategy.scope():
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED),pruner=optuna.pruners.HyperbandPruner())
        study.optimize(objective, n_trials=200)

    # Save best trial parameters
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
