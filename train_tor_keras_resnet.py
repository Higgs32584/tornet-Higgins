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
    Conv2D,
    LayerNormalization,
    GlobalAveragePooling2D,
    Dense,
    Multiply,
    Reshape,
    BatchNormalization,
    Add,
    Activation,
    MaxPool2D,
    Dropout
)
from typing import Dict, List, Tuple

from tornet.data.loader import get_dataloader
from tornet.data.preprocess import get_shape
from tornet.data.constants import ALL_VARIABLES, CHANNEL_MIN_MAX
from tornet.metrics.keras import metrics as tfm
from tornet.utils.general import make_exp_dir, make_callback_dirs
from tornet.models.keras.layers import CoordConv2D, FillNaNs
import tornet.data.tfds.tornet.tornet_dataset_builder

logging.basicConfig(level=logging.ERROR)

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


def build_model(shape:Tuple[int]=(120,240,2),
                c_shape:Tuple[int]=(120,240,2),
                input_variables:List[str]=ALL_VARIABLES,
                start_filters:int=64,
                l2_reg:float=0.001,
                background_flag:float=-3.0,
                include_range_folded:bool=True,
                head='maxpool',
                model='vgg',fusion_method="gated"):
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
    
    if model == 'resnet':
        x,c = resnet_block(x,c,filters=start_filters,ksize=3,l2_reg=l2_reg,drop_rate=0.1)
        x,c = resnet_block(x,c,filters=start_filters*2,ksize=3,l2_reg=l2_reg,drop_rate=0.1)
        x,c = resnet_block(x,c,filters=start_filters*4,ksize=3,l2_reg=l2_reg,drop_rate=0.1)
    
    elif model == 'wide_resnet':
        x, c = wide_resnet_block(x, c, filters=start_filters, widen_factor=2, l2_reg=l2_reg, drop_rate=0.1)
        x, c = wide_resnet_block(x, c, filters=start_filters*2, widen_factor=2, l2_reg=l2_reg, drop_rate=0.1)


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
                       activation='relu')([x, c])
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
    
    x = Add()([x, shortcut_x])  
    x = Activation('relu')(x)

    # Pooling and dropout
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    c = MaxPool2D(pool_size=2, strides=2, padding='same')(c)

    if drop_rate > 0:
        x = Dropout(rate=drop_rate)(x)
    
    return x, c


def fusion_block(x1, x2, method="concat"):
    """
    Feature Fusion Block - Combines two feature maps using different fusion strategies.
    :param x1: First feature map
    :param x2: Second feature map (typically a deeper layer upsampled)
    :param method: "concat", "gated", or "multi_cnn"
    """
    attention = Dense(64, activation='relu')(x1)
    attention = Dense(x1.shape[-1], activation='sigmoid')(attention)
    return Multiply()([x1, attention])  # Gated Fusion
    return x1  # Default to identity (no fusion)

    
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

def se_block(x, filters, ratio=8):
    """Squeeze-and-Excitation Block with 1x1 Convolutions"""
    se = GlobalAveragePooling2D()(x)
    se = Reshape((1, 1, filters))(se)
    se = Conv2D(filters // ratio, (1, 1), activation="swish", kernel_initializer="he_normal")(se)
    se = Conv2D(filters, (1, 1), activation="sigmoid", kernel_initializer="he_normal")(se)
    return Multiply()([x, se])

def resnet_block(x, c, filters=64, ksize=3, l2_reg=1e-6, drop_rate=0.0):
    shortcut_x, shortcut_c = x, c  # Skip connection
    shortcut_x, shortcut_c = CoordConv2D(filters=filters, kernel_size=ksize, padding="same",
                       kernel_regularizer=keras.regularizers.l2(l2_reg),
                       activation=None)([shortcut_x, shortcut_c])

    # First convolution
    x, c = CoordConv2D(filters=filters, kernel_size=ksize, padding="same",
                       kernel_regularizer=keras.regularizers.l2(l2_reg),
                       activation=None)([x, c])
    x= LayerNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    # Second convolution
    x, c = CoordConv2D(filters=filters, kernel_size=ksize, padding="same",
                       kernel_regularizer=keras.regularizers.l2(l2_reg),activation=None)([x, c])
    x= LayerNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x, c = CoordConv2D(filters=filters, kernel_size=ksize, padding="same",
                       kernel_regularizer=keras.regularizers.l2(l2_reg),activation='relu')([x, c])
    x = keras.layers.Add()([x, shortcut_x])  
    x = keras.layers.Activation('relu')(x)  # Activation after addition

    # Pooling and optional dropout
    x = keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    c = keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(c)
    if drop_rate > 0:
        x = keras.layers.Dropout(rate=drop_rate)(x)
    
    return x, c

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

# Enable GPU Memory Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

strategy = tf.distribute.MultiWorkerMirroredStrategy()

tf.config.optimizer.set_jit(True)  # Enable XLA (Accelerated Linear Algebra)
logging.info(f"Number of devices: {strategy.num_replicas_in_sync}")
from tensorflow.keras import mixed_precision
# Default Configuration
DEFAULT_CONFIG={"epochs":100, 
                "input_variables": ["DBZ", "VEL", "KDP", "ZDR","RHOHV","WIDTH"], 
                "train_years": [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020], 
                "val_years": [2021, 2022], "batch_size": 128
                , "model": "resnet", 
                "start_filters": 48, 
                "learning_rate": 1e-3, 
                "decay_steps": 1386, 
                "decay_rate": 0.958, 
                "l2_reg": 1e-5, "wN": 1.0, "w0": 1.0, "w1": 1.0, "w2": 1.0, "wW": 1.0, "label_smooth": 0.1, 
                "loss": "cce", "head": "maxpool", "exp_name": "tornado_baseline", "exp_dir": ".",
                  "dataloader": "tensorflow-tfds", 
                  "dataloader_kwargs": {"select_keys": ["DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH", "range_folded_mask", "coordinates"]}}

def main(config):
    # Gather all hyperparams
    epochs=config.get('epochs')
    batch_size=config.get('batch_size')
    start_filters=config.get('start_filters')
    decay_steps=config.get('decay_steps')
    decay_rate=config.get('decay_rate')
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
    ds_val = get_dataloader(dataloader, DATA_ROOT, val_years, "train", batch_size, weights, **dataloader_kwargs)
    
    x, _, _ = next(iter(ds_train))
    in_shapes = (None, None, get_shape(x)[-1])
    c_shapes = (None, None, x["coordinates"].shape[-1])
    nn = build_model_exp(shape=in_shapes, c_shape=c_shapes, start_filters=start_filters, 
                         l2_reg=config['l2_reg'], input_variables=input_variables, model=model)
    print(nn.summary())
    
    # Loss Function
    import tensorflow as tf
    from tensorflow.keras import backend as K
    loss = keras.losses.BinaryCrossentropy( from_logits=False)
    # Optimizer with Learning Rate Decay
    lr = keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=config['learning_rate'], first_decay_steps=config['decay_steps'], t_mul=2.0, m_mul=0.9)
    opt = keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
    from_logits=False
    # Metrics (Optimize AUCPR)
    metrics = [keras.metrics.AUC(from_logits=from_logits,curve='PR',name='AUCPR',num_thresholds=2000),
                tfm.TruePositives(from_logits,name='TruePositives'),
                tfm.FalsePositives(from_logits,name='FalsePositives'), 
                tfm.FalseNegatives(from_logits,name='FalseNegatives'), 
                tfm.Precision(from_logits,name='Precision'), 
                tfm.Recall(from_logits,name='Recall'),
                tfm.F1Score(from_logits=from_logits,name='F1')]
    
    nn.compile(loss=loss, metrics=metrics, optimizer=opt)
    
    # Experiment Directory
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
        keras.callbacks.EarlyStopping(monitor='val_AUCPR', patience=3, mode='max', restore_best_weights=True),
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