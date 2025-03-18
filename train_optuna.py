import sys
import os
import json
import logging
import optuna
import keras
import numpy as np
import tensorflow as tf
from typing import List, Tuple
from tornet.data.loader import get_dataloader
from tornet.data.preprocess import get_shape
from tornet.data.constants import ALL_VARIABLES, CHANNEL_MIN_MAX
from tornet.models.keras.losses import mae_loss
from tornet.metrics.keras import metrics as tfm
from tornet.utils.general import make_exp_dir
from tornet.models.keras.layers import CoordConv2D, FillNaNs

logging.basicConfig(level=logging.ERROR)
SEED = 42  # Choose any integer

# Set the random seed for Python's built-in random module

# Set the random seed for NumPy
np.random.seed(SEED)

# Set the random seed for TensorFlow
tf.random.set_seed(SEED)
# Environment Variables
DATA_ROOT = '/home/ubuntu/tfds'
TFDS_DATA_DIR = DATA_ROOT
os.environ['TORNET_ROOT'] = DATA_ROOT
os.environ['TFDS_DATA_DIR'] = TFDS_DATA_DIR

# Enable GPU Memory Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

strategy = tf.distribute.MultiWorkerMirroredStrategy()
tf.config.optimizer.set_jit(True)  # Enable XLA (Accelerated Linear Algebra)

DEFAULT_CONFIG = {
    "epochs": 50,
    "batch_size": 128,
    "model": "vgg",
    "start_filters": 48,
    "learning_rate": 1e-3,
    "l2_reg": 1e-5,
    "train_years": [2015],
    "val_years": [2016],
    "input_variables": ["DBZ", "VEL", "KDP", "ZDR", "RHOHV", "WIDTH"],
    "exp_name": "tornado_baseline",
    "exp_dir": ".",
    "dataloader": "tensorflow-tfds",
    "dataloader_kwargs": {
        "select_keys": ["DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH", "range_folded_mask", "coordinates"]
    },
    "weights": {
        "wN": 1.0,
        "w0": 10.0,
        "w1": 10.0,
        "w2": 10.0,
        "wW": 10.0
    }
}
def vgg_block(x,c, filters=64, ksize=3, n_convs=2, l2_reg=1e-6, drop_rate=0.0,se_block=False):

    for _ in range(n_convs):
        x,c = CoordConv2D(filters=filters,
                          kernel_size=ksize,
                          kernel_regularizer=keras.regularizers.l2(l2_reg),
                          padding='same',
                          activation='relu',)([x,c])
    if se_block:
        x = se_block(x, filters)
    
    x = keras.layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
    c = keras.layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(c)
    if drop_rate>0:
        x = keras.layers.Dropout(rate=drop_rate)(x)
    return x,c

def normalize(x,
              name:str):
    """
    Channel-wise normalization using known CHANNEL_MIN_MAX
    """
    min_max = np.array(CHANNEL_MIN_MAX[name]) # [2,]
    n_sweeps=x.shape[-1]
    
    # choose mean,var to get approximate [-1,1] scaling
    var=((min_max[1]-min_max[0])/2)**2 # scalar
    var=np.array(n_sweeps*[var,])    # [n_sweeps,]
    
    offset=(min_max[0]+min_max[1])/2    # scalar
    offset=np.array(n_sweeps*[offset,]) # [n_sweeps,]

    return keras.layers.Normalization(mean=offset,
                                         variance=var,
                                         name='Normalize_%s' % name)(x)

def build_model(trial, shape: Tuple[int] = (120, 240, 2), c_shape: Tuple[int] = (120, 240, 2), input_variables: List[str] = ALL_VARIABLES,start_filters=48,l2_reg=.003):

    inputs = {v: keras.Input(shape, name=v) for v in input_variables}
    normalized_inputs = keras.layers.Concatenate(axis=-1)([normalize(inputs[v], v) for v in input_variables])
    normalized_inputs = FillNaNs(-3.0)(normalized_inputs)
    
    cin = keras.Input(c_shape, name='coordinates')
    inputs['coordinates'] = cin
    x, c = normalized_inputs, cin
    
    x, c = vgg_block(x, c, filters=start_filters, ksize=3, l2_reg=l2_reg, n_convs=2, drop_rate=0.1)
    x, c = vgg_block(x, c, filters=2 * start_filters, ksize=3, l2_reg=l2_reg, n_convs=2, drop_rate=0.1)
    
    x = keras.layers.Conv2D(filters=256, kernel_size=1, activation='relu')(x)
    x = keras.layers.Conv2D(filters=1, kernel_size=1, name='heatmap')(x)
    output = keras.layers.GlobalMaxPooling2D()(x)
    
    return keras.Model(inputs=inputs, outputs=output)

def objective(trial):
    config = DEFAULT_CONFIG.copy()
    config["learning_rate"] = trial.suggest_loguniform("learning_rate", 1e-5, 5e-5)
    config["l2_reg"] = trial.suggest_loguniform("l2_reg", 5e-4, 5e-3)
    config["weight_decay"] = trial.suggest_loguniform("weight_decay", 1e-5, 5e-5)
    config["start_filters"] = trial.suggest_int("start_filters", 48, 80, step=8)
    ds_train = get_dataloader(config['dataloader'], DATA_ROOT, config['train_years'], "train",
                              config['batch_size'], config['weights'], **config['dataloader_kwargs'])
    ds_val = get_dataloader(config['dataloader'], DATA_ROOT, config['val_years'], "train",
                            config['batch_size'], config['weights'], **config['dataloader_kwargs'])

    x, _, _ = next(iter(ds_train))
    in_shapes = (None, None, get_shape(x)[-1])
    c_shapes = (None, None, x["coordinates"].shape[-1])

    model = build_model(trial, shape=in_shapes, c_shape=c_shapes, input_variables=config['input_variables'],l2_reg=config["l2_reg"],start_filters=config["start_filters"])

    loss = keras.losses.BinaryCrossentropy()
    lr = keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=config['learning_rate'], first_decay_steps=10, t_mul=2.0, m_mul=0.9)
    opt = keras.optimizers.AdamW(learning_rate=lr, weight_decay=config['weight_decay'])
    metrics = [keras.metrics.AUC(name='AUCPR', curve='PR', num_thresholds=2000)]
    model.compile(loss=loss, optimizer=opt, metrics=metrics)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_AUCPR', patience=2, mode='max', restore_best_weights=True)

    history = model.fit(ds_train, epochs=config['epochs'], validation_data=ds_val, verbose=1, callbacks=[early_stopping])

    val_aucpr = max(history.history.get('val_AUCPR', [0]))
    
    trial.report(val_aucpr, len(history.history['val_AUCPR']))
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return val_aucpr
def log_trial_results(study, trial):
    import datetime
    import json
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"optuna_results_{timestamp}.json"

    trial_data = {
        "trial_id": trial.number,
        "params": trial.params,
        "value": trial.value
    }
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    with open(log_filename, "a") as f:
        json.dump(trial_data, f, indent=4)
        f.write("\n")

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=10,callbacks=[log_trial_results])
    print("Best trial:", study.best_trial)
