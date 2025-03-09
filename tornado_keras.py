import sys
# Uncomment if tornet isn't installed in your environment or in your path already
#sys.path.append('../')  

import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tornet.data.tf.loader import create_tf_dataset 
from tornet.data.constants import ALL_VARIABLES
from tornet.models.keras.cnn_baseline import build_model
import tensorflow_datasets as tfds
import tornet.data.tfds.tornet.tornet_dataset_builder # registers 'tornet'

data_type='train'
years = [2013,2014,2015,2016,2017,2018,2019,2020]
years_val=[2021,2022]
ds_train = tfds.load('tornet',split='+'.join(['%s-%d' % (data_type,y) for y in years]))
ds_val = tfds.load('tornet',split='+'.join(['%s-%d' % (data_type,y) for y in years_val]))

import tornet.data.preprocess as pp
from tornet.data import preprocess as tfpp
def preprocess_me(ds):
    ds = ds.map(lambda d: pp.add_coordinates(d,include_az=False,backend=tf))
    ds = ds.map(pp.remove_time_dim)
    #ds = ds.map(tfpp.split_x_y)
    ds = ds.prefetch(tf.data.AUTOTUNE)        
    ds = ds.batch(32)
    return ds
from tornet.data.constants import CHANNEL_MIN_MAX
ds_train=preprocess_me(ds_train)
ds_val = preprocess_me(ds_val)
print(len(next(iter(ds_train))))
# Access coordinates correctly
#c_shape = sample_batch["coordinates"].shape
nn = build_model()
nn.compile()


history=nn.fit(ds_train,
                   epochs=20,
                    validation_data=ds_val,
                    verbose=1) 


