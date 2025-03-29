import sys
import os
import json
import logging
import optuna
from tensorflow import keras
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
    Flatten,
    Lambda,
    Multiply,
    GlobalAveragePooling2D,
    UpSampling2D,
    Conv2D,
)
# Loss Function
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam,AdamW

import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
# Environment Variables
DATA_ROOT = '/home/ubuntu/tfds'
TFDS_DATA_DIR = DATA_ROOT
os.environ['TORNET_ROOT'] = DATA_ROOT
os.environ['TFDS_DATA_DIR'] = TFDS_DATA_DIR
#os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
DEFAULT_CONFIG={"epochs":5, 
                "input_variables": ["DBZ", "VEL", "KDP", "ZDR","RHOHV","WIDTH"], 
                "train_years": [2013], 
                "val_years": [2013], "batch_size": 128
                , "model": "wide_resnet", 
                "start_filters": 32, 
                "learning_rate": 1e-1, 
                "decay_steps": 6114, 
                "decay_rate": 0.958,
                "dropout_rate":0.0, 
                "l2_reg": 0.0, 
                "wN": 1.0, "w0": 1.0, "w1": 1.0, "w2": 1.0, "wW": 1.0, "label_smooth": 0.1, 
                "loss": "cce", "head": "maxpool", "exp_name": "tornado_baseline", "exp_dir": ".",
                  "dataloader": "tensorflow-tfds", 
                  "dataloader_kwargs": {"select_keys": ["DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH", "range_folded_mask", "coordinates"]}}

from tensorflow.keras.optimizers import AdamW

DATA_ROOT = '/home/ubuntu/tfds'
TFDS_DATA_DIR = DATA_ROOT
os.environ['TORNET_ROOT'] = DATA_ROOT
os.environ['TFDS_DATA_DIR'] = TFDS_DATA_DIR
#strategy = tf.distribute.MirroredStrategy()
#tf.config.optimizer.set_jit(True) 
# Enable GPU Memory Growth



from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    Add,
    Activation,
    MaxPool2D,
    GlobalMaxPooling2D,
    Dropout,
    Lambda,
    Multiply,
    GlobalAveragePooling2D,
    Concatenate
)
import tensorflow as tf
from tensorflow.keras.layers import Dense, Multiply, Add, Conv2D, UpSampling2D, AveragePooling2D
from tensorflow.keras.optimizers import SGD
def top_k_pooling(x, k=5):
    """
    Top-k average pooling across spatial dimensions for each channel.
    Args:
        x: Tensor of shape (B, H, W, C)
    Returns:
        Tensor of shape (B, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    channels = x.shape[-1]

    # Flatten spatial dims: (B, H*W, C)
    x_flat = tf.reshape(x, [batch_size, height * width, channels])
    # Transpose to (B, C, H*W)
    x_flat = tf.transpose(x_flat, perm=[0, 2, 1])
    # Top-k over spatial dimension (H*W)
    topk = tf.nn.top_k(x_flat, k=k, sorted=False).values  # shape: (B, C, k)
    # Mean over top-k: (B, C)
    return tf.reduce_mean(topk, axis=-1)




def build_model(model:str='wide_resnet',shape:Tuple[int]=(120,240,2),
                c_shape:Tuple[int]=(120,240,2),
                input_variables:List[str]=ALL_VARIABLES,
                start_filters:int=64,
                l2_reg:float=0.001,
                background_flag:float=-3.0,
                include_range_folded:bool=True,dropout_rate=0.1,head:str='maxpool'):
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
        x, c = wide_resnet_block(x, c, filters=start_filters, widen_factor=2, l2_reg=l2_reg,nconvs=3, drop_rate=dropout_rate)
        x, c = wide_resnet_block(x, c, filters=start_filters*2, widen_factor=2, l2_reg=l2_reg,nconvs=3, drop_rate=dropout_rate)
        x, c = wide_resnet_block(x, c, filters=start_filters*4, widen_factor=2, l2_reg=l2_reg,nconvs=3, drop_rate=dropout_rate)
        x=se_block(x)
    if head=='mlp':
        x = GlobalMaxPooling2D()(x)
        x = keras.layers.Dense(units = 128, activation ='relu')(x) 
        x = keras.layers.Dense(units = 56, activation ='relu')(x) 
        output = keras.layers.Dense(1)(x)
    elif head=='maxpool':
        # Per gridcell
        x = keras.layers.Conv2D(filters=512, kernel_size=1,
                          kernel_regularizer=keras.regularizers.l2(l2_reg),
                          activation='relu')(x)
        x = keras.layers.Conv2D(filters=256, kernel_size=1,
                          kernel_regularizer=keras.regularizers.l2(l2_reg),
                          activation='relu')(x)
        # x = Lambda(lambda t: top_k_pooling(t, k=5), output_shape=(256,), name='TopKPooling')(x)
        x = keras.layers.Conv2D(filters=1, kernel_size=1,name='heatmap')(x)
        # Max in scene
        output = GlobalMaxPooling2D()(x)

    elif head=='maxpool_sigmoid':
        # Per gridcell
        x = keras.layers.Conv2D(filters=512, kernel_size=1,
                          kernel_regularizer=keras.regularizers.l2(l2_reg),
                          activation='relu')(x)
        x = keras.layers.Conv2D(filters=256, kernel_size=1,
                          kernel_regularizer=keras.regularizers.l2(l2_reg),
                          activation='relu')(x)
        x = keras.layers.Conv2D(filters=1, kernel_size=1,name='heatmap')(x)
        # Max in scene
        output = keras.layers.GlobalMaxPooling2D()(x)
        output = keras.layers.Activation('sigmoid')(output)
    elif head=='spatial':
        x = Conv2D(1, kernel_size=1, activation='relu')(x)  # reduce to 1 channel
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
    elif head == 'spatial_flattening':
        gap = GlobalAveragePooling2D()(x)
        gmp = GlobalMaxPooling2D()(x)
        x = Concatenate()([gap, gmp])
        x = Dense(128, activation='relu')(x)
        output = Dense(1)(x)
    return keras.Model(inputs=inputs,outputs=output)

def se_block(x, ratio=2):
    filters = int(x.shape[-1])
    se = GlobalAveragePooling2D()(x)
    se = Dense(filters // ratio, activation="relu")(se)
    se = Dense(filters, activation="sigmoid")(se)
    return Multiply()([x, se])

def wide_resnet_block(x, c, filters=64, widen_factor=2, l2_reg=1e-6, drop_rate=0.0,nconvs=2):
    """Wide ResNet Block with CoordConv2D"""
    shortcut_x, shortcut_c = x, c  # Skip connection

    # 3x3 CoordConv2D (Wider filters)
    for i in range(nconvs):
        x, c = CoordConv2D(filters=filters * widen_factor, kernel_size=3, padding="same",
                        kernel_regularizer=keras.regularizers.l2(l2_reg),
                        activation=None)([x, c])
        x = BatchNormalization(momentum=0.95)(x)
    # Skip Connection
    shortcut_x, shortcut_c = CoordConv2D(filters=filters * widen_factor, kernel_size=1, padding="same",
                                         kernel_regularizer=keras.regularizers.l2(l2_reg),
                                         activation=None)([shortcut_x, shortcut_c])
    
    #x = se_block(x)

    # Add Residual Connection
    x = Activation('relu')(x)
    x = Add()([x, shortcut_x])

    # Pooling and dropout
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    c = MaxPool2D(pool_size=2, strides=2, padding='same')(c)

    if drop_rate > 0:
        x = Dropout(rate=drop_rate)(x)
    
    return x, c


def main(config):
    
    # Gather all hyperparams
    epochs=config.get('epochs')
    batch_size=config.get('batch_size')
    start_filters=config.get('start_filters')
    decay_steps=config.get('decay_steps')
    decay_rate=config.get('decay_rate')
    l2_reg=config.get('l2_reg')
    lr=config.get('learning_rate')
    wN=config.get('wN')
    w0=config.get('w0')
    w1=config.get('w1')
    w2=config.get('w2')
    wW=config.get('wW')
    head=config.get('head')
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
    ds_train = get_dataloader(config['dataloader'], DATA_ROOT, config['train_years'], "train",
                              config['batch_size'], weights, **config['dataloader_kwargs'])
    ds_val = get_dataloader(config['dataloader'], DATA_ROOT, config['train_years'], "train",
                              config['batch_size'], {"wN": 1.0,"w0": 1.0,"w1": 1.0,"w2": 1.0,"wW": 1.0}, **config['dataloader_kwargs'])
    x, _, _ = next(iter(ds_train))
    in_shapes = (None, None, get_shape(x)[-1])
    c_shapes = (None, None, x["coordinates"].shape[-1])

    nn = build_model(shape=in_shapes, c_shape=c_shapes, start_filters=start_filters, 
                        l2_reg=config['l2_reg'], input_variables=input_variables, model=model,head=head)
    print(nn.summary())
    from_logits=True
    from tensorflow.keras.losses import BinaryCrossentropy
    from_logits = True

    # Optimizer with Learnindg Rate Decay
    #loss = focal_loss(gamma=0.25, alpha=0.75)
    loss= BinaryCrossentropy(from_logits=from_logits,label_smoothing=0.0)
    lr_schedule = CosineDecayRestarts(
        initial_learning_rate=lr,
        first_decay_steps=decay_steps,  # 1 epoch
        t_mul=2.0,               # each cycle doubles
        m_mul=0.9                # restart peak decays slightly
    )

    # opt = AdamW(
    #     learning_rate=lr_schedule,
    #     weight_decay=1e-4,
    #     beta_1=0.9,
    #     beta_2=0.999,
    #     epsilon=1e-7
    # )

    opt = SGD(learning_rate=lr,momentum=.9,nesterov=True)
    #opt=Adam(learning_rate=lr)
    metrics = [keras.metrics.AUC(from_logits=from_logits,curve='PR',name='AUCPR',num_thresholds=2000)]
    nn.compile(loss=loss, metrics=metrics, optimizer=opt,jit_compile=True)
    
    # Experiment Directory

    # Callbacks (Now Monitors AUCPR)
    callbacks = [
            keras.callbacks.TerminateOnNaN(),
            keras.callbacks.EarlyStopping(monitor='val_AUCPR', patience=3, mode='max', restore_best_weights=True),
        ]
    
        # Train Model
    history = nn.fit(ds_train, epochs=epochs,validation_data=ds_val, verbose=1)
    scores = nn.evaluate(ds_val)

    # Get Best Metrics
    best_aucpr = max(history.history.get('val_AUCPR', [0]))
    return {'AUCPR': best_aucpr}

if __name__ == '__main__':
    config = DEFAULT_CONFIG
    if len(sys.argv) > 1:
        config.update(json.load(open(sys.argv[1], 'r')))
    main(config)