"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.


The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

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
                model='vgg'):
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
    
    if model == 'vgg':
        x, c = vgg_block(x, c, filters=start_filters, ksize=3, l2_reg=l2_reg, n_convs=2, drop_rate=0.1)   # (60,120)
        # x, c = vgg_block(x, c, filters=2*start_filters, ksize=3, l2_reg=l2_reg, n_convs=2, drop_rate=0.1)  # (30,60)
        # x, c = vgg_block(x, c, filters=4*start_filters, ksize=3, l2_reg=l2_reg, n_convs=3, drop_rate=0.1)  # (15,30)
        # x, c = vgg_block(x, c, filters=8*start_filters, ksize=3, l2_reg=l2_reg, n_convs=3, drop_rate=0.1)  # (7,15)
        # x, c = vgg_block(x, c, filters=8*start_filters, ksize=3, l2_reg=l2_reg, n_convs=3)  # (3,7)

    elif model == 'inception':  
        filter_multipliers = [1, 2, 4]  # Define filter scaling factors
        
        for multiplier in filter_multipliers:
            x, c = inception_block(x, c, filters=start_filters * multiplier, l2_reg=l2_reg)

        # Optional deeper inception block (uncomment if needed)
        # x, c = inception_block(x, c, filters=start_filters * 8, l2_reg=l2_reg)




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
        # Max in scene
        output = keras.layers.GlobalMaxPooling2D()(x)

        
    return keras.Model(inputs=inputs,outputs=output)



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Multiply, Reshape

    
def vgg_block(x,c, filters=64, ksize=3, n_convs=2, l2_reg=1e-6, drop_rate=0.0):

    for _ in range(n_convs):
        x,c = CoordConv2D(filters=filters,
                          kernel_size=ksize,
                          kernel_regularizer=keras.regularizers.l2(l2_reg),
                          padding='same',
                          activation='relu',)([x,c])
    x = keras.layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
    c = keras.layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(c)
    if drop_rate>0:
        x = keras.layers.Dropout(rate=drop_rate)(x)
    return x,c

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Dropout, LeakyReLU, Concatenate, MaxPool2D

def inception_block(x, c, filters=128, l2_reg=1e-6, drop_rate=0.3):
    # Branch 1x1
    branch1, c = CoordConv2D(filters=filters, kernel_size=1, padding='same', 
                              kernel_regularizer=keras.regularizers.l2(l2_reg),
                              activation='relu')([x, c])
    branch1, c = CoordConv2D(filters=filters, kernel_size=1, padding='same', 
                              kernel_regularizer=keras.regularizers.l2(l2_reg),
                              activation='relu')([branch1, c])
    branch3, c = CoordConv2D(filters=filters, kernel_size=2, padding='same', 
                              kernel_regularizer=keras.regularizers.l2(l2_reg),
                              activation='relu')([x, c])
    branch3, c = CoordConv2D(filters=filters, kernel_size=2, padding='same', 
                              kernel_regularizer=keras.regularizers.l2(l2_reg),
                              activation='relu')([branch3, c])

    x = Concatenate()([branch1, branch3])
    x = BatchNormalization()(x)
    if drop_rate > 0:
        x = Dropout(drop_rate)(x)

    # Max Pooling
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    c = MaxPool2D(pool_size=2, strides=2, padding='same')(c)

    return x, c
def dense_block(x, c, filters=64, ksize=3, growth_rate=32, l2_reg=1e-6, drop_rate=0.1):
    concat_inputs = [x]  # Store all previous feature maps

    for _ in range(3):  # 3-layer Dense Block
        new_x, c = CoordConv2D(filters=growth_rate, kernel_size=ksize, padding='same',
                               kernel_regularizer=keras.regularizers.l2(l2_reg),
                               activation='relu')([x, c])
        new_x = BatchNormalization()(new_x)
        concat_inputs.append(new_x)  # Store for concatenation

        x = Concatenate()(concat_inputs)  # Merge all previous feature maps

    x = Dropout(drop_rate)(x) if drop_rate > 0 else x

    # Pooling
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    c = MaxPool2D(pool_size=2, strides=2, padding='same')(c)

    return x, c


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


