import sys
import os
import json
import shutil
import logging
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras.layers import (
    BatchNormalization,
    Add,
    Activation,
    MaxPool2D,
    Dropout,
    GlobalAveragePooling2D,
    Dense,
    Multiply,
    Conv2D,
    GlobalMaxPooling2D,
    Reshape

)
from typing import List, Tuple
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tornet.data.loader import get_dataloader
from tornet.data.preprocess import get_shape
from tornet.data.constants import ALL_VARIABLES, CHANNEL_MIN_MAX
from tornet.metrics.keras import metrics as tfm
from tornet.utils.general import make_exp_dir, make_callback_dirs
from tornet.models.keras.layers import CoordConv2D, FillNaNs
import tornet.data.tfds.tornet.tornet_dataset_builder

logging.basicConfig(level=logging.ERROR)
SEED = 42  

# Set random seeds for reproducibility
import random
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# Environment Variables
DATA_ROOT = '/home/ubuntu/tfds'
TFDS_DATA_DIR = DATA_ROOT
os.environ['TORNET_ROOT'] = DATA_ROOT
os.environ['TFDS_DATA_DIR'] = TFDS_DATA_DIR
import tensorflow as tf

class TverskyLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6, from_logits=False, name="tversky_loss"):
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        # Apply sigmoid if using logits
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, self.smooth, 1.0 - self.smooth)

        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky_index

# Environment Variables
EXP_DIR = "."
DATA_ROOT = '/home/ubuntu/tfds'
TORNET_ROOT=DATA_ROOT
TFDS_DATA_DIR = '/home/ubuntu/tfds'
DATA_ROOT = "/home/ubuntu/tfds"
TFDS_DATA_DIR = "/home/ubuntu/tfds"
os.environ['TORNET_ROOT']= DATA_ROOT
os.environ['TFDS_DATA_DIR']=TFDS_DATA_DIR


logging.info(f'TORNET_ROOT={DATA_ROOT}')

# Enable GPU Memory Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

strategy = tf.distribute.MirroredStrategy()
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
import tensorflow as tf

tf.config.optimizer.set_jit(True)  # Enable XLA (Accelerated Linear Algebra)
logging.info(f"Number of devices: {strategy.num_replicas_in_sync}")
# Default Configuration
DEFAULT_CONFIG={"epochs":100, 
                "input_variables": ["DBZ", "VEL", "KDP", "ZDR","RHOHV","WIDTH"], 
                "train_years": [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020], 
                "val_years": [2021, 2022], "batch_size": 64
                , "model": "wide_resnet", 
                "start_filters": 48, 
                "learning_rate": 1e-5, 
                "decay_steps": 1386, 
                "decay_rate": 0.958,
                "dropout_rate":0.0, 
                "l2_reg": 1e-6, "wN": 0.00001, "w0": 1.0, "w1": 1.0, "w2": 1.0, "wW": 1.0, "label_smooth": 0.1, 
                "loss": "cce", "head": "maxpool", "exp_name": "tornado_baseline", "exp_dir": ".",
                  "dataloader": "tensorflow-tfds", 
                  "dataloader_kwargs": {"select_keys": ["DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH", "range_folded_mask", "coordinates"]}}

def main(config):
    # Gather all hyperparams
    epochs=config.get('epochs')
    batch_size=config.get('batch_size')
    start_filters=config.get('start_filters')
    dropout_rate=config.get('dropout_rate')

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
    import tensorflow_datasets as tfds
    import tornet.data.tfds.tornet.tornet_dataset_builder

    # Apply to Train and Validation Data
    ds_train = get_dataloader(dataloader, DATA_ROOT, train_years, "train", batch_size, weights, **dataloader_kwargs)
    ds_val = get_dataloader(dataloader, DATA_ROOT, val_years, "train", batch_size, {'wN':1.0,'w0':1.0,'w1':1.0,'w2':1.0,'wW':1.0}, **dataloader_kwargs)

    x, _, _ = next(iter(ds_train))
    
    in_shapes = (None, None, get_shape(x)[-1])
    c_shapes = (None, None, x["coordinates"].shape[-1])
        # Loss Function
    import tensorflow as tf
    from tensorflow.keras import backend as K
    # Optimizer with Learning Rate Decay
    from_logits=True
    from tensorflow.keras.losses import Tversky,BinaryCrossentropy,BinaryFocalCrossentropy

    loss = BinaryCrossentropy(from_logits=True,label_smoothing= 0.1)
    # Optimizer with Learnindg Rate Decay


    opt = AdamW(
        learning_rate=CosineDecayRestarts(
    initial_learning_rate=lr,
    first_decay_steps=100,
    t_mul=2.0,
    m_mul=0.9
),
        weight_decay=1e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )


    from_logits=True
    # Metrics (Optimize AUCPR)
    metrics = [keras.metrics.AUC(from_logits=from_logits,curve='PR',name='AUCPR',num_thresholds=1000), 
                tfm.BinaryAccuracy(from_logits,name='BinaryAccuracy'), 
                tfm.TruePositives(from_logits,name='TruePositives'),
                tfm.FalsePositives(from_logits,name='FalsePositives'), 
                tfm.TrueNegatives(from_logits,name='TrueNegatives'),
                tfm.FalseNegatives(from_logits,name='FalseNegatives'), 
                tfm.Precision(from_logits,name='Precision'), 
                tfm.Recall(from_logits,name='Recall'),
                tfm.F1Score(from_logits=from_logits,name='F1')]
    

    nn = keras.models.load_model('/home/ubuntu/tornet-Higgins/40 epoch march 22nd overnight/tornadoDetector_030.keras')
    
    nn.compile(optimizer=opt,loss=loss)
    expdir = make_exp_dir(exp_dir=exp_dir, prefix=exp_name)
    with open(os.path.join(expdir,'data.json'),'w') as f:
        json.dump(
            {'data_root':DATA_ROOT,
             'train_data':list(train_years), 
             'val_data':list(val_years)},f)
    with open(os.path.join(expdir,'params.json'),'w') as f:
        json.dump({'config':config},f)
    # Copy the training script
    shutil.copy(__file__, os.path.join(expdir,'train.py')) 
    # Callbacks (Now Monitors AUCPR)
    checkpoint_name = os.path.join(expdir, 'tornadoDetector_{epoch:03d}.keras')
    callbacks = [
        keras.callbacks.ModelCheckpoint(checkpoint_name, monitor='val_AUCPR', save_best_only=False),
        keras.callbacks.CSVLogger(os.path.join(expdir, 'history.csv')),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.EarlyStopping(monitor='val_AUCPR', patience=5, mode='max', restore_best_weights=True),
        keras.callbacks.EarlyStopping(monitor='val_F1', patience=5, mode='max', restore_best_weights=True)

    ]
    
    # TensorBoard Logging
    if keras.config.backend() == "tensorflow":
        callbacks.append(keras.callbacks.TensorBoard(log_dir=os.path.join(expdir, 'logs'), write_graph=False))
    
    # Train Model
    history = nn.fit(ds_train, epochs=epochs, validation_data=ds_val, callbacks=callbacks, verbose=1)
    
    # Get Best Metrics
    best_aucpr = max(history.history.get('val_AUCPR', [0]))
    return {'AUCPR': best_aucpr}

if __name__ == '__main__':
    config = DEFAULT_CONFIG
    if len(sys.argv) > 1:
            config.update(json.load(open(sys.argv[1], 'r')))
    with strategy.scope():
        main(config)