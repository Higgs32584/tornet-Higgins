import os
import json
import numpy as np
import keras
import sys
import shutil
import logging
from tensorflow.keras.models import load_model
from tornet.data.loader import get_dataloader
from tornet.data.preprocess import get_shape
from tornet.data.constants import ALL_VARIABLES
from tornet.models.keras.losses import mae_loss
from tornet.models.keras.cnn_baseline import build_model
from tornet.metrics.keras import metrics as tfm
from tornet.utils.general import make_exp_dir, make_callback_dirs
import tensorflow as tf

EXP_DIR = os.environ.get('EXP_DIR', '.')
TORNET_ROOT = os.environ['TORNET_ROOT']
TFDS_DATA_DIR=os.environ['TFDS_DATA_DIR']
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth set for: {gpu}")
    except RuntimeError as e:
        print(e)


import tensorflow as tf
#os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"] = "false"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
DEFAULT_CONFIG={
    'epochs':100,
    'input_variables':ALL_VARIABLES,
    'train_years':list(range(2013,2020)),
    'val_years':list(range(2020,2022)),
    'batch_size':64,
    'model':'vgg',
    'start_filters':48,
    'learning_rate':1e-3,
    'decay_steps':1386,
    'decay_rate':0.958,
    'l2_reg':1e-5,
    'wN':1.0,
    'w0':1.0,
    'w1':1.0,
    'w2':2.0,
    'wW':0.5,
    'label_smooth':0,
    'loss':'cce',
    'head':'maxpool',
    'exp_name':'tornet_baseline',
    'exp_dir':EXP_DIR,                                                                                                                                          'dataloader':"tensorflow-tfds",
    'dataloader_kwargs': {}
}





def main(config,start_epoch):
    batch_size = config.get('batch_size')
    start_filters = config.get('start_filters')
    learning_rate = config.get('learning_rate')
    decay_steps = config.get('decay_steps')
    decay_rate = config.get('decay_rate')
    l2_reg = config.get('l2_reg')
    wN, w0, w1, w2, wW = config.get('wN'), config.get('w0'), config.get('w1'), config.get('w2'), config.get('wW')
    head = config.get('head')
    label_smooth = config.get('label_smooth')
    loss_fn = config.get('loss')
    input_variables = config.get('input_variables')
    exp_name = config.get('exp_name')
    exp_dir = config.get('exp_dir')
    train_years = config.get('train_years')
    val_years = config.get('val_years')
    dataloader = config.get('dataloader')
    dataloader_kwargs = config.get('dataloader_kwargs')
    logging.info(f'Using {dataloader} dataloader')
    logging.info('Resuming training with config:')
    logging.info(config)

    weights = {'wN': wN, 'w0': w0, 'w1': w1, 'w2': w2, 'wW': wW}


    dataloader_kwargs.update({'select_keys': input_variables + ['range_folded_mask', 'coordinates']})
    ds_train = get_dataloader(dataloader, TORNET_ROOT, train_years, "train", batch_size, weights, **dataloader_kwargs)
    ds_val = get_dataloader(dataloader, TORNET_ROOT, val_years, "train", batch_size, weights, **dataloader_kwargs)
    checkpoint_path = 'tornado_norm/checkpoints/tornadoDetector_001.keras'
    logging.info(f'Loading checkpoint: {checkpoint_path}')
    model = load_model(checkpoint_path)

    # Setup learning rate schedule
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        learning_rate, decay_steps, decay_rate, staircase=False, name="exp_decay"
    )

    # Loss function setup
    from_logits = True
    if loss_fn.lower() == 'cce':
        loss = keras.losses.BinaryCrossentropy(from_logits=from_logits, label_smoothing=label_smooth)
    elif loss_fn.lower() == 'hinge':
        loss = keras.losses.Hinge()
    elif loss_fn.lower() == 'mae':
        loss = lambda yt, yp: mae_loss(yt, yp)
    else:
        raise RuntimeError(f'Unknown loss {loss_fn}')

    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    metrics = [
        keras.metrics.AUC(from_logits=from_logits, name='AUC', num_thresholds=2000),
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
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer, weighted_metrics=[])

    # Callbacks
    expdir = make_exp_dir(exp_dir=exp_dir, prefix=exp_name)
    tboard_dir, checkpoints_dir = make_callback_dirs(expdir)
    checkpoint_name = os.path.join(checkpoints_dir, 'tornadoDetector' + '_{epoch:03d}.keras')
    


    callbacks = [
        keras.callbacks.ModelCheckpoint(checkpoint_name, monitor='val_AUC', save_best_only=False),
        keras.callbacks.CSVLogger(os.path.join(expdir, 'history.csv'), append=True),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.EarlyStopping(monitor='val_AUC',patience=5,mode='max',restore_best_weights=True)
        ]
    
    # Resume training
    history = model.fit(
        ds_train,
        epochs=config.get('epochs') - start_epoch,
        initial_epoch=start_epoch,
        validation_data=ds_val,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

if __name__=='__main__':
    config=DEFAULT_CONFIG
    # Load param file if given
    if len(sys.argv)>1:
        config.update(json.load(open(sys.argv[1],'r')))
    main(config,15)
