# ==== Imports ====
import os
import sys
import json
import logging
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

from tornet.data.loader import get_dataloader
from tornet.data.preprocess import get_shape
from tornet.data.constants import ALL_VARIABLES
from tornet.models.keras.losses import mae_loss
from tornet.models.keras.cnn_baseline import build_model
from tornet.metrics.keras import metrics as tfm
from tornet.utils.general import make_exp_dir, make_callback_dirs

# ==== Environment Configuration ====
EXP_DIR = os.environ.get('EXP_DIR', '.')
logging.basicConfig(level=logging.ERROR)

# Set random seeds for reproducibility
import random
SEED = 42
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

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth set for: {gpu}")
    except RuntimeError as e:
        print(e)

# ==== Default Config ====
DEFAULT_CONFIG = {
    # Data
    'input_variables': ALL_VARIABLES,
    'train_years': list(range(2013, 2020)),
    'val_years': list(range(2020, 2022)),
    'batch_size': 128,
    'dataloader': "tensorflow-tfds",
    'dataloader_kwargs': {},

    # Model architecture
    'model': 'vgg',
    'head': 'maxpool',
    'start_filters': 48,

    # Optimization
    'epochs': 100,
    'learning_rate': 1e-4,
    'decay_steps': 1386,
    'decay_rate': 0.958,
    'l2_reg': 1e-5,
    'weight_decay': 1e-4,

    # Loss function & metrics
    'loss': 'combo',
    'label_smooth': 0,
    'from_logits': True,
    'focal_gamma': 2.0,
    'focal_alpha': 0.85,
    'tversky_alpha': 0.3,
    'tversky_beta': 0.7,
    'combo_alpha': 0.7,

    # Class weights
    'wN': 1.0,
    'w0': 1.0,
    'w1': 1.0,
    'w2': 1.0,
    'wW': 1.0,

    # Experiment
    'exp_name': 'tornet_baseline',
    'exp_dir': os.environ.get('EXP_DIR', '.')
}

# ==== Loss Functions ====
def focal_loss(gamma=2.0, alpha=0.85):
    def loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -tf.reduce_mean(alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt))
    return loss_fn

def tversky_loss(alpha=0.3, beta=0.7, smooth=1e-6):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        return 1 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return loss_fn

def combo_loss(alpha=0.7):
    return lambda y_true, y_pred: (
        alpha * tversky_loss(alpha=0.5, beta=0.5)(y_true, y_pred)
        + (1 - alpha) * focal_loss(gamma=2.0, alpha=0.85)(y_true, y_pred)
    )

# ==== Main Training Function ====
def main(config, start_epoch):
    # Extract config values
    batch_size = config['batch_size']
    lr = config['learning_rate']
    decay_steps = config['decay_steps']
    exp_name = config['exp_name']
    exp_dir = config['exp_dir']
    train_years = config['train_years']
    val_years = config['val_years']
    input_variables = config['input_variables']
    dataloader = config['dataloader']
    dataloader_kwargs = config['dataloader_kwargs']

    weights = {k: config[k] for k in ['wN', 'w0', 'w1', 'w2', 'wW']}

    logging.info(f'Using {dataloader} dataloader')
    logging.info('Resuming training with config:')
    logging.info(config)

    dataloader_kwargs.update({'select_keys': input_variables + ['range_folded_mask', 'coordinates']})

    ds_train = get_dataloader(dataloader, TORNET_ROOT, train_years, "train", batch_size, weights, **dataloader_kwargs)
    ds_val = get_dataloader(dataloader, TORNET_ROOT, val_years, "train", batch_size, {k: 1.0 for k in weights}, **dataloader_kwargs)

    checkpoint_path = 'tornado_detector_baseline.keras'
    logging.info(f'Loading checkpoint: {checkpoint_path}')
    model = load_model(checkpoint_path)

    # Optimizer with learning rate schedule
    lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=lr,
        first_decay_steps=decay_steps,
        t_mul=2.0,
        m_mul=0.9
    )

    optimizer = keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )

    from_logits = False
    loss = combo_loss()
    metrics = [
        keras.metrics.AUC(from_logits=from_logits, curve='PR', name='AUCPR', num_thresholds=2000),
        tfm.BinaryAccuracy(from_logits, name='BinaryAccuracy'),
        tfm.TruePositives(from_logits, name='TruePositives'),
        tfm.FalsePositives(from_logits, name='FalsePositives'),
        tfm.TrueNegatives(from_logits, name='TrueNegatives'),
        tfm.FalseNegatives(from_logits, name='FalseNegatives'),
        tfm.Precision(from_logits, name='Precision'),
        tfm.Recall(from_logits, name='Recall'),
        tfm.F1Score(from_logits=from_logits, name='F1')
    ]

    model.compile(loss=loss, metrics=metrics, optimizer=optimizer, weighted_metrics=[],jit_compile=True)

    # Setup Callbacks
    expdir = make_exp_dir(exp_dir=exp_dir, prefix=exp_name)
    shutil.copy(__file__, os.path.join(expdir, 'train_script_backup.py'))
    tboard_dir, checkpoints_dir = make_callback_dirs(expdir)
    checkpoint_name = os.path.join(checkpoints_dir, 'tornadoDetector_{epoch:03d}.keras')

    callbacks = [
        keras.callbacks.ModelCheckpoint(checkpoint_name, monitor='val_AUC', save_best_only=False),
        keras.callbacks.CSVLogger(os.path.join(expdir, 'history.csv'), append=True),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.EarlyStopping(monitor='val_AUCPR', patience=5, mode='max', restore_best_weights=True)
    ]

    # Resume training
    history = model.fit(
        ds_train,
        epochs=config['epochs'] - start_epoch,
        initial_epoch=start_epoch,
        validation_data=ds_val,
        callbacks=callbacks,
        verbose=1
    )

    return history

# ==== Entry Point ====
if __name__ == '__main__':
    config = DEFAULT_CONFIG
    if len(sys.argv) > 1:
        config.update(json.load(open(sys.argv[1], 'r')))
    main(config, start_epoch=0)
