import os
import sys
import json
import shutil
import random
import logging
from typing import List, Tuple

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
    Reshape,
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts

from tornet.data.loader import get_dataloader
from tornet.data.preprocess import get_shape
from tornet.data.constants import ALL_VARIABLES, CHANNEL_MIN_MAX
from tornet.metrics.keras import metrics as tfm
from tornet.utils.general import make_exp_dir, make_callback_dirs
from tornet.models.keras.layers import CoordConv2D, FillNaNs
import tornet.data.tfds.tornet.tornet_dataset_builder

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

# Environment Variables
EXP_DIR = "."
DATA_ROOT = '/home/ubuntu/tfds'
TORNET_ROOT=DATA_ROOT
TFDS_DATA_DIR = '/home/ubuntu/tfds'
DATA_ROOT = "/home/ubuntu/tfds"
TFDS_DATA_DIR = "/home/ubuntu/tfds"
os.environ['TORNET_ROOT']= DATA_ROOT
os.environ['DATA_ROOT']= DATA_ROOT

os.environ['TFDS_DATA_DIR']=TFDS_DATA_DIR
import logging
logging.basicConfig(level=logging.INFO)
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
TFDS_DATA_DIR = DATA_ROOT
EXP_DIR = "."
DATA_ROOT = '/home/ubuntu/tfds/'
TFDS_DATA_DIR = '/home/ubuntu/tfds/'
os.environ['TORNET_ROOT']= DATA_ROOT
os.environ['DATA_ROOT']= DATA_ROOT

os.environ['TFDS_DATA_DIR']=TFDS_DATA_DIR
EXP_DIR=os.environ.get('EXP_DIR','.')
DATA_ROOT=os.environ['TORNET_ROOT']
logging.info('TORNET_ROOT='+DATA_ROOT)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)



DEFAULT_CONFIG={
    'epochs':100,
    'input_variables':ALL_VARIABLES,
    'train_years':list(range(2013,2021)),
    'val_years':list(range(2021,2023)),
    'batch_size':128,
    'model':'vgg',
    'start_filters':48,
    'learning_rate':1e-4,
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
    'exp_dir':EXP_DIR,
    'dataloader':"tensorflow-tfds",
    'dataloader_kwargs': {}
}
def vgg_block(x,c, filters=64, ksize=3, n_convs=2, l2_reg=1e-6, drop_rate=0.0):

    for _ in range(n_convs):
        x,c = CoordConv2D(filters=filters,
                          kernel_size=ksize,
                          kernel_regularizer=keras.regularizers.l2(l2_reg),
                          padding='same',
                          activation=None)([x,c])
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
def wide_resnet_block(x, c, filters=64, widen_factor=2, l2_reg=1e-6, drop_rate=0.0,nconvs=2):
    """Wide ResNet Block with CoordConv2D"""
    shortcut_x, shortcut_c = x, c  # Skip connection

    # 3x3 CoordConv2D (Wider filters)
    for i in range(nconvs):
        x, c = CoordConv2D(filters=filters * widen_factor, kernel_size=3, padding="same",
                        kernel_regularizer=keras.regularizers.l2(l2_reg),
                        activation=None)([x, c])
        x = BatchNormalization()(x)
    # Skip Connection
    shortcut_x, shortcut_c = CoordConv2D(filters=filters * widen_factor, kernel_size=1, padding="same",
                                         kernel_regularizer=keras.regularizers.l2(l2_reg),
                                         activation=None)([shortcut_x, shortcut_c])
    

    # Add Residual Connection
    x = Activation('relu')(x)
    x = Add()([x, shortcut_x])

    # Pooling and dropout
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    c = MaxPool2D(pool_size=2, strides=2, padding='same')(c)

    if drop_rate > 0:
        x = Dropout(rate=drop_rate)(x)
    
    return x, c




def build_model(shape:Tuple[int]=(120,240,2),
                c_shape:Tuple[int]=(120,240,2),
                input_variables:List[str]=ALL_VARIABLES,
                start_filters:int=64,
                l2_reg:float=0.001,
                background_flag:float=-3.0,
                include_range_folded:bool=True,
                head='maxpool'):
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

    x, c = wide_resnet_block(x, c, filters=start_filters, widen_factor=2, l2_reg=l2_reg,nconvs=2, drop_rate=0.1)
    x, c = wide_resnet_block(x, c, filters=start_filters*2, widen_factor=2, l2_reg=l2_reg,nconvs=2, drop_rate=0.1)
    x, c = wide_resnet_block(x, c, filters=start_filters*4, widen_factor=2, l2_reg=l2_reg,nconvs=2, drop_rate=0.1)
    if head=='mlp':
        # MLP head
        x = keras.layers.Flatten()(x) 
        x = keras.layers.Dense(units = 4096, activation ='relu')(x) 
        x = keras.layers.Dense(units = 2024, activation ='relu')(x) 
        output = keras.layers.Dense(1)(x)
    elif head=='maxpool':
        # Per gridcell
        x = keras.layers.Conv2D(filters=512, kernel_size=1,
                          kernel_regularizer=keras.regularizers.l2(l2_reg),
                          activation='relu')(x)
        x = keras.layers.Conv2D(filters=256, kernel_size=1,
                          kernel_regularizer=keras.regularizers.l2(l2_reg),
                          activation='relu')(x)
        x = keras.layers.Conv2D(filters=1, kernel_size=1,name='heatmap')(x)
        x = keras.layers.GlobalMaxPooling2D()(x)
        output = keras.layers.Activation("sigmoid")(x)
        
    return keras.Model(inputs=inputs,outputs=output)
   
tf.config.optimizer.set_jit(True)


def main(config):
    # Gather all hyperparams
    epochs=config.get('epochs')
    batch_size=config.get('batch_size')
    start_filters=config.get('start_filters')
    learning_rate=config.get('learning_rate')
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
    exp_dir=config.get('exp_dir')
    train_years=config.get('train_years')
    val_years=config.get('val_years')
    dataloader=config.get('dataloader')
    dataloader_kwargs = config.get('dataloader_kwargs')

    logging.info(f"Using {keras.config.backend()} backend")
    logging.info(f'Using {dataloader} dataloader')
    logging.info('Running with config:')
    logging.info(config)

    weights={'wN':wN,'w0':w0,'w1':w1,'w2':w2,'wW':wW}
    
    # Create data laoders
    dataloader_kwargs.update({'select_keys':input_variables+['range_folded_mask','coordinates']})
    ds_train = get_dataloader(dataloader, DATA_ROOT, train_years, "train", batch_size, weights, **dataloader_kwargs)
    ds_val = get_dataloader(dataloader, DATA_ROOT, val_years, "train", batch_size, {'wN':1.0,'w0':1.0,'w1':1.0,'w2':1.0,'wW':1.0}, **dataloader_kwargs)    
    
    x, _, _ = next(iter(ds_train))
    in_shapes = (None, None, get_shape(x)[-1])
    c_shapes = (None, None, x["coordinates"].shape[-1])
    
    nn = build_model(shape=in_shapes,
                     c_shape=c_shapes,
                     start_filters=start_filters,
                     l2_reg=l2_reg,
                     input_variables=input_variables,
                     head=head)
    print(nn.summary())
    
    # model setup
    lr=keras.optimizers.schedules.ExponentialDecay(
                learning_rate, decay_steps, decay_rate, staircase=False, name="exp_decay")
    
    from_logits=False
    if loss_fn.lower()=='cce':
        loss = keras.losses.BinaryCrossentropy( from_logits=from_logits, 
                                                    label_smoothing=label_smooth )
    elif loss_fn.lower()=='hinge':
        loss = keras.losses.Hinge() # automatically converts labels to -1,1
    else:
        raise RuntimeError('unknown loss %s' % loss_fn)


    opt  = keras.optimizers.Adam(learning_rate=lr)

    # Compute various metrics while training
    metrics = [keras.metrics.AUC(from_logits=from_logits,curve='PR',name='AUCPR',num_thresholds=2000), 
                tfm.BinaryAccuracy(from_logits,name='BinaryAccuracy'), 
                tfm.TruePositives(from_logits,name='TruePositives'),
                tfm.FalsePositives(from_logits,name='FalsePositives'), 
                tfm.TrueNegatives(from_logits,name='TrueNegatives'),
                tfm.FalseNegatives(from_logits,name='FalseNegatives'), 
                tfm.Precision(from_logits,name='Precision'), 
                tfm.Recall(from_logits,name='Recall'),
                tfm.F1Score(from_logits=from_logits,name='F1')]
    
    nn.compile(loss=loss,
                metrics=metrics,
                optimizer=opt,
                weighted_metrics=[],jit_compile=True)
    
    ## Setup experiment directory and model callbacks
    expdir = make_exp_dir(exp_dir=exp_dir,prefix=exp_name)
    logging.info('expdir='+expdir)

    # Copy the properties that were used
    with open(os.path.join(expdir,'data.json'),'w') as f:
        json.dump(
            {'data_root':DATA_ROOT,
             'train_data':list(train_years), 
             'val_data':list(val_years)},f)
    with open(os.path.join(expdir,'params.json'),'w') as f:
        json.dump({'config':config},f)
    # Copy the training script
    shutil.copy(__file__, os.path.join(expdir,'train.py')) 
    
    # Callbacks
    tboard_dir, checkpoints_dir=make_callback_dirs(expdir)
    checkpoint_name=os.path.join(checkpoints_dir, 'tornadoDetector'+'_{epoch:03d}.keras' )
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(checkpoint_name,monitor='val_loss',save_best_only=False),
        keras.callbacks.CSVLogger(os.path.join(expdir,'history.csv')),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.EarlyStopping(monitor='val_AUCPR', patience=5, mode='max', restore_best_weights=True),
    ]

    # TensorBoard callback requires tensorflow backend
    if keras.config.backend() == "tensorflow":
        callbacks.append(keras.callbacks.TensorBoard(log_dir=tboard_dir,write_graph=False))#,profile_batch=(5,15)),

    ## FIT
    history=nn.fit(ds_train,
                   epochs=epochs,
                    validation_data=ds_val,
                    callbacks=callbacks,
                    verbose=1) 
    
    # At the end,  report the best score observed over all epochs
    if len(history.history['val_AUC'])>0:
        best_auc = np.max(history.history['val_AUC'])
        best_aucpr = np.max(history.history['val_AUCPR'])
    else:
        best_auc,best_aucpr=0.5,0.0
    
    return {'AUC':best_auc,'AUCPR':best_aucpr}


if __name__=='__main__':
    config=DEFAULT_CONFIG
    # Load param file if given
    if len(sys.argv)>1:
        config.update(json.load(open(sys.argv[1],'r')))
    main(config)