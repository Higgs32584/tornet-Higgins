import sys
import os
import numpy as np
import json
import shutil
import keras
import logging
import tensorflow as tf
from tornet.data.loader import get_dataloader
from tornet.data.preprocess import get_shape
from tornet.data.constants import ALL_VARIABLES
from tornet.models.keras.losses import mae_loss
from tornet.metrics.keras import metrics as tfm
from tornet.utils.general import make_exp_dir, make_callback_dirs
import tensorflow_datasets as tfds
import tornet.data.tfds.tornet.tornet_dataset_builder
logging.basicConfig(level=logging.ERROR)

from typing import Dict, List, Tuple
import numpy as np
import keras
from tornet.models.keras.layers import CoordConv2D, FillNaNs
from tornet.data.constants import CHANNEL_MIN_MAX, ALL_VARIABLES





def build_model_exp(shape:Tuple[int]=(120,240,2),
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
    
    if model == 'vgg_1':
        x, c = vgg_block(x, c, filters=start_filters, ksize=3, l2_reg=l2_reg, n_convs=2, drop_rate=0.1)   # (60,120)
    elif model == 'vgg_2':
        x, c = vgg_block(x, c, filters=2*start_filters, ksize=3, l2_reg=l2_reg, n_convs=2, drop_rate=0.1)  # (30,60)
        x, c = vgg_block(x, c, filters=4*start_filters, ksize=3, l2_reg=l2_reg, n_convs=2, drop_rate=0.1)  # (15,30)
    elif model == 'vgg_3':
        x, c = vgg_block(x, c, filters=2*start_filters, ksize=3, l2_reg=l2_reg, n_convs=2, drop_rate=0.1)  # (30,60)
        x, c = vgg_block(x, c, filters=4*start_filters, ksize=3, l2_reg=l2_reg, n_convs=3, drop_rate=0.1)  # (15,30)
        x, c = vgg_block(x, c, filters=8*start_filters, ksize=3, l2_reg=l2_reg, n_convs=3, drop_rate=0.1)  # (7,15)
    elif model == 'vgg_4':
        x, c = vgg_block(x, c, filters=2*start_filters, ksize=3, l2_reg=l2_reg, n_convs=2, drop_rate=0.1)  # (30,60)
        x, c = vgg_block(x, c, filters=4*start_filters, ksize=3, l2_reg=l2_reg, n_convs=3, drop_rate=0.1)  # (15,30)
        x, c = vgg_block(x, c, filters=8*start_filters, ksize=3, l2_reg=l2_reg, n_convs=3, drop_rate=0.1)  # (7,15)
        x, c = vgg_block(x, c, filters=16*start_filters, ksize=3, l2_reg=l2_reg, n_convs=3, drop_rate=0.1)  # (7,15)





    if head=='maxpool':
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
        
    return keras.Model(inputs=inputs,outputs=output)




from tensorflow.keras.layers import Conv2D, Add,GroupNormalization
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Multiply, Reshape,Flatten


def fusion_block(x1, x2):
    """
    Feature Fusion Block - Combines two feature maps using different fusion strategies.
    :param x1: First feature map
    :param x2: Second feature map (typically a deeper layer upsampled)
    :param method: "concat", "gated", or "multi_cnn"
    """
    attention = Dense(64, activation='relu')(x1)
    attention = Dense(x1.shape[-1], activation='sigmoid')(attention)
    return Multiply()([x1, attention])  # Gated Fusion

    
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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Dropout, LeakyReLU, Concatenate, MaxPool2D, LayerNormalization, ReLU





import keras
from keras.layers import DepthwiseConv2D, Conv2D, MaxPool2D, Dropout, Activation, Concatenate
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


logging.info(f'TORNET_ROOT={DATA_ROOT}')

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
                , "model": "vgg_1", 
                "start_filters": 48, 
                "learning_rate": 1e-3, 
                "decay_steps": 1386, 
                "decay_rate": 0.958, 
                "l2_reg": 1e-5, "wN": 1.0, "w0": 10.0, "w1": 10.0, "w2": 10.0, "wW": 10.0, "label_smooth": 0.1, 
                "loss": "cce", "head": "maxpool", "exp_name": "tornado_baseline", "exp_dir": ".",
                  "dataloader": "tensorflow-tfds", 
                  "dataloader_kwargs": {"select_keys": ["DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH", "range_folded_mask", "coordinates"]}}

def main(config):
    # Gather all hyperparams
    epochs=config.get('epochs')
    batch_size=config.get('batch_size')
    start_filters=config.get('start_filters')
    learning_rate = config.get('learning_rate') * (batch_size/128)
    decay_steps=config.get('decay_steps')
    decay_rate=config.get('decay_rate')
    l2_reg=config.get('l2_reg')
    wN=config.get('wN')
    w0=config.get('w0')
    w1=config.get('w1')
    w2=config.get('w2')
    wW=config.get('wW')
    head=config.get('head')
    label_smooth=config.get('label_smooth')
    loss_fn = config.get('loss')
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
    nn = build_model_exp(shape=in_shapes, c_shape=c_shapes, start_filters=config['start_filters'], 
                         l2_reg=config['l2_reg'], input_variables=config['input_variables'], model=config['model'],fusion_method='multi_cnn')
    print(nn.summary())
    
    # Loss Function
    import tensorflow as tf
    from tensorflow.keras import backend as K
    loss = keras.losses.BinaryCrossentropy( from_logits=False)
    # Optimizer with Learning Rate Decay
    lr = keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=1e-4, first_decay_steps=10, t_mul=2.0, m_mul=0.9)
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