import sys
import os
import json
import shutil
import logging
import numpy as np
import tensorflow as tf
import keras
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
    ReLU
)
import tensorflow as tf
import multiprocessing

num_cores = multiprocessing.cpu_count()

tf.config.threading.set_inter_op_parallelism_threads(num_cores)
tf.config.threading.set_intra_op_parallelism_threads(num_cores)
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
from tensorflow import keras

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


DEFAULT_CONFIG={"epochs":100, 
                "input_variables": ["DBZ", "VEL", "KDP", "ZDR","RHOHV","WIDTH"], 
                "train_years": [2013,2014,2015,2016,2017,2018,2019,2020], 
                "val_years": [2021,2022], "batch_size": 128
                , "model": "wide_resnet", 
                "start_filters": 48, 
                "learning_rate": 0.0001596, 
                "first_decay_steps": 2221, 
                "weight_decay": 2.63e-7,
                "dropout_rate": 0.122, 
                "l2_reg": 1.59e-7,
                "wN": 0.05069,
                    "w0": 0.55924,
                    "w1": 3.49593,
                    "w2": 11.26479,
                    "wW": 0.20101,
                  "label_smooth": 0.1, 
                "loss": "cce", "head": "maxpool", "exp_name": "tornado_baseline", "exp_dir": ".",
                  "dataloader": "tensorflow-tfds", 
                  "dataloader_kwargs": {"select_keys": ["DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH", "range_folded_mask", "coordinates","rng_lower","rng_upper","az_lower","az_upper"]}}

def main(config):
    # Gather all hyperparams
    epochs=config.get('epochs')
    batch_size=config.get('batch_size')
    start_filters=config.get('start_filters')
    dropout_rate=config.get('dropout_rate')
    first_decay_steps=config.get('first_decay_steps')
    lr=config.get('learning_rate')
    l2_reg=config.get('l2_reg')
    wN=config.get('wN')
    w0=config.get('w0')
    w1=config.get('w1')
    w2=config.get('w2')
    wW=config.get('wW')
    input_variables=config.get('input_variables')
    exp_name=config.get('exp_name')
    model=config.get('model')
    exp_dir=config.get('exp_dir')
    train_years=config.get('train_years')
    val_years=config.get('val_years')
    dataloader=config.get('dataloader')
    dataloader_kwargs = config.get('dataloader_kwargs')
    
    logging.info(f'Using {dataloader} dataloader')
    logging.info(f'Running with config: {config}')
    weights={'wN':wN,'w0':w0,'w1':w1,'w2':w2,'wW':wW}

    # Data Loaders
    dataloader_kwargs.update({'select_keys': input_variables + ['range_folded_mask', 'coordinates']})
    # Apply to Train and Validation Data
    ds_val = get_dataloader(dataloader, DATA_ROOT, val_years, "train", 128, {'wN':1.0,'w0':1.0,'w1':1.0,'w2':1.0,'wW':1.0}, **dataloader_kwargs)

    from sklearn.metrics import precision_score, recall_score, f1_score

    def threat_score(y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fn + fp + 1e-6)

    nn =tf.keras.models.load_model("/home/ubuntu/tornet-Higgins/best_models_so_far/tornadoDetector_013.keras",compile=False,safe_mode=False)
    print(nn.input)
    y_true = []
    y_pred_prob = []

    for x_batch, y_batch,_ in ds_val:
        y_true.append(y_batch)
        y_pred_prob.append((nn.predict(x_batch)))  # Faster than predict()

    
    import numpy as np
    y_true = np.concatenate(y_true).squeeze()
    y_pred_prob = np.concatenate(y_pred_prob).squeeze()

    thresholds = np.linspace(0, 1, 100)
    precision = []
    recall = []
    f1 = []
    ts = []

    for t in thresholds:
        y_pred = (y_pred_prob >= t).astype(int)
        precision.append(precision_score(y_true, y_pred, zero_division=0))
        recall.append(recall_score(y_true, y_pred, zero_division=0))
        f1.append(f1_score(y_true, y_pred, zero_division=0))
        ts.append(threat_score(y_true, y_pred))
    output_path = "/home/ubuntu/tornet-Higgins/threshold_metrics.txt"
    with open(output_path, "w") as f:
        f.write("Threshold\tPrecision\tRecall\tF1\tThreatScore\n")
        for i, t in enumerate(thresholds):
            f.write(f"{t:.4f}\t{precision[i]:.4f}\t{recall[i]:.4f}\t{f1[i]:.4f}\t{ts[i]:.4f}\n")

    


if __name__ == '__main__':
    config = DEFAULT_CONFIG
    if len(sys.argv) > 1:
            config.update(json.load(open(sys.argv[1], 'r')))
    main(config)