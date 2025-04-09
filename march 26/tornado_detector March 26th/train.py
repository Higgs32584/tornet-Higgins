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
class FalseAlarmRate(tf.keras.metrics.Metric):
    def __init__(self, name="false_alarm_rate", **kwargs):
        super().__init__(name=name, **kwargs)
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.epsilon = 1e-7

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Binary predictions
        y_true = tf.cast(y_true, tf.float32)

        fp = tf.reduce_sum((1 - y_true) * y_pred)
        tp = tf.reduce_sum(y_true * y_pred)

        self.false_positives.assign_add(fp)
        self.true_positives.assign_add(tp)

    def result(self):
        return self.false_positives / (self.false_positives + self.true_positives + self.epsilon)

    def reset_states(self):
        self.false_positives.assign(0)
        self.true_positives.assign(0)

class ThreatScore(tf.keras.metrics.Metric):
    def __init__(self, name="threat_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")
        self.epsilon = 1e-7

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.fp.assign_add(tf.reduce_sum((1 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1 - y_pred)))

    def result(self):
        return self.tp / (self.tp + self.fp + self.fn + self.epsilon)

    def reset_states(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)


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
TORNET_ROOT=DATA_ROOT
TFDS_DATA_DIR = DATA_ROOT
EXP_DIR = "."
os.environ['TORNET_ROOT'] = DATA_ROOT
os.environ['TFDS_DATA_DIR'] = TFDS_DATA_DIR


logging.info(f'TORNET_ROOT={DATA_ROOT}')

def build_model(model='wide_resnet',shape:Tuple[int]=(120,240,2),
                c_shape:Tuple[int]=(120,240,2),
                input_variables:List[str]=ALL_VARIABLES,
                start_filters:int=64,
                l2_reg:float=0.001,
                background_flag:float=-3.0,
                include_range_folded:bool=True,dropout_rate=0.1):
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
    az_lower = keras.Input(shape=(1,), name='az_lower')
    az_upper = keras.Input(shape=(1,), name='az_upper')
    rng_lower = keras.Input(shape=(1,), name='rng_lower')
    rng_upper = keras.Input(shape=(1,), name='rng_upper')

    inputs['az_lower'] = az_lower
    inputs['az_upper'] = az_upper
    inputs['rng_lower'] = rng_lower
    inputs['rng_upper'] = rng_upper

    x,c = normalized_inputs,cin
    
    if model == 'wide_resnet':
        x, c = wide_resnet_block(x, c, filters=start_filters, widen_factor=2, l2_reg=l2_reg,nconvs=3, drop_rate=dropout_rate)
        x, c = wide_resnet_block(x, c, filters=start_filters*2, widen_factor=2, l2_reg=l2_reg,nconvs=3, drop_rate=dropout_rate)
        x, c = wide_resnet_block(x, c, filters=start_filters*4, widen_factor=2, l2_reg=l2_reg,nconvs=3, drop_rate=dropout_rate)
        x=se_block(x)
    def normalize_scalar(x, mean, std):
        return keras.layers.Lambda(lambda x: (x - mean) / std)(x)
    az_l_norm = normalize_scalar(az_lower, 151.0, 110.0)
    az_u_norm = normalize_scalar(az_upper, 211.0, 110.0)
    rng_l_norm = normalize_scalar(rng_lower, 76193.0, 49812.0)
    rng_u_norm = normalize_scalar(rng_upper, 136193.0, 49812.0)

    scalar_context = keras.layers.Concatenate(name='radar_scalar_inputs')(
        [az_l_norm, az_u_norm, rng_l_norm, rng_u_norm]
    )
    scalar_dense = Dense(32, activation='relu')(scalar_context)
    scalar_dense = Dense(x.shape[-1], activation='sigmoid', name='scalar_gate')(scalar_dense)
    scalar_gate = keras.layers.Reshape((1, 1, x.shape[-1]))(scalar_dense)
    x = Multiply(name='context_gated_features')([x, scalar_gate])

    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    attention_map = Conv2D(1, 1, activation='sigmoid', name='attention_map')(x)  # shape (B, H, W, 1)
    #attention_map = Dropout(rate=0.2, name='attention_dropout')(attention_map)
    x_weighted = Multiply()([x, attention_map])

    x_avg = GlobalAveragePooling2D()(x_weighted)
    x_max = GlobalMaxPooling2D()(x_weighted)

    # Combine everything
    x_concat = keras.layers.Concatenate(name='final_concat')([x_avg, x_max, scalar_dense])

    x_dense = Dense(64, activation='relu')(x_concat)
    output = Dense(1, activation='sigmoid', name='output')(x_dense)
    return keras.Model(inputs=inputs,outputs=output)

def se_block(x, ratio=16, name=None):
    filters = x.shape[-1]
    
    # Squeeze
    se = GlobalAveragePooling2D(name=f"{name}_gap" if name else None)(x)

    # Excite
    se = Dense(filters // ratio, activation="relu", name=f"{name}_fc1" if name else None)(se)
    se = Dense(filters, activation="sigmoid", name=f"{name}_fc2" if name else None)(se)

    # Explicit broadcast shape for Multiply
    se = Reshape((1, 1, filters), name=f"{name}_reshape" if name else None)(se)

    # Multiply
    x = Multiply(name=f"{name}_scale" if name else None)([x, se])
    return x


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

strategy = tf.distribute.MirroredStrategy()
#os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

tf.config.optimizer.set_jit(True)  # Enable XLA (Accelerated Linear Algebra)
logging.info(f"Number of devices: {strategy.num_replicas_in_sync}")
# Default Configuration
DEFAULT_CONFIG={"epochs":100, 
                "input_variables": ["DBZ", "VEL", "KDP", "ZDR","RHOHV","WIDTH"], 
                "train_years": [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020], 
                "val_years": [2021, 2022], "batch_size": 128
                , "model": "wide_resnet", 
                "start_filters": 48, 
                "learning_rate": 6e-4, 
                "decay_steps": 1386, 
                "decay_rate": 0.958,
                "dropout_rate":0.1, 
                "l2_reg": 1e-4, "wN": 0.25, "wW": 0.75, "w0": 3.0, "w1": 5.0, "w2": 8.0, "label_smooth": 0.1, 
                "loss": "cce", "head": "maxpool", "exp_name": "tornado_baseline", "exp_dir": ".",
                  "dataloader": "tensorflow-tfds", 
                  "dataloader_kwargs": {"select_keys": ["DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH", "range_folded_mask", "coordinates","rng_lower","rng_upper","az_lower","az_upper"]}}
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
    dataloader_kwargs.update({'select_keys': input_variables + [ "range_folded_mask", "coordinates","rng_lower","rng_upper","az_lower","az_upper"]})
    import tensorflow_datasets as tfds
    import tornet.data.tfds.tornet.tornet_dataset_builder

    # Apply to Train and Validation Data
    ds_train = get_dataloader(dataloader, DATA_ROOT, train_years, "train", batch_size, weights, **dataloader_kwargs)
    ds_val = get_dataloader(dataloader, DATA_ROOT, val_years, "train", batch_size, {'wN':1.0,'w0':1.0,'w1':1.0,'w2':1.0,'wW':1.0}, **dataloader_kwargs)

    x, _, _ = next(iter(ds_train))
    
    in_shapes = (None, None, get_shape(x)[-1])
    c_shapes = (None, None, x["coordinates"].shape[-1])
    nn = build_model(model=model,shape=in_shapes, c_shape=c_shapes, start_filters=start_filters, 
                         l2_reg=l2_reg, input_variables=input_variables,dropout_rate=dropout_rate)
    print(nn.summary())
    
    # Loss Function
    import tensorflow as tf
    from tensorflow.keras import backend as K
    # Optimizer with Learning Rate Decay
    from_logits=False

    from tensorflow.keras.losses import BinaryCrossentropy,Tversky
    def focal_loss(gamma=2.0, alpha=0.85):
        def loss_fn(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            return -tf.reduce_mean(alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt))
        return loss_fn
    def tversky_loss(alpha=0.3, beta=0.7, smooth=1e-6):
        """
        Tversky Loss: adjusts trade-off between FP and FN.
        alpha = weight for FP
        beta = weight for FN
        """
        def loss_fn(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            tp = tf.reduce_sum(y_true * y_pred)
            fp = tf.reduce_sum((1 - y_true) * y_pred)
            fn = tf.reduce_sum(y_true * (1 - y_pred))
            return 1 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        return loss_fn
    def combo_loss(alpha=0.7):
        return lambda y_true, y_pred: alpha * tversky_loss(alpha=0.5, beta=0.5)(y_true, y_pred) + \
                                    (1 - alpha) * focal_loss(gamma=2.0, alpha=0.85)(y_true, y_pred)
    loss = combo_loss()
    
    # Optimizer with Learnindg Rate Decay

    lr_schedule = CosineDecayRestarts(
        initial_learning_rate=lr,
        first_decay_steps=3060,  # 1 epoch
        t_mul=2.0,               # each cycle doubles
        m_mul=0.9                # restart peak decays slightly
    )

    opt = AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    # Metrics (Optimize AUCPR)
    metrics = [keras.metrics.AUC(from_logits=from_logits,curve='PR',name='AUCPR',num_thresholds=1000), 
                tfm.BinaryAccuracy(from_logits,name='BinaryAccuracy'), 
                tfm.TruePositives(from_logits,name='TruePositives'),
                tfm.FalsePositives(from_logits,name='FalsePositives'), 
                tfm.TrueNegatives(from_logits,name='TrueNegatives'),
                tfm.FalseNegatives(from_logits,name='FalseNegatives'), 
                tfm.Precision(from_logits,name='Precision'), 
                tfm.Recall(from_logits,name='Recall'),
                FalseAlarmRate(name='FalseAlarmRate'),
                tfm.F1Score(from_logits=from_logits,name='F1'),
                ThreatScore(name='ThreatScore')]
    
    nn.compile(loss=loss, metrics=metrics, optimizer=opt,jit_compile=True)
    
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
        keras.callbacks.EarlyStopping(monitor='val_AUCPR', patience=5, mode='max', restore_best_weights=True),
        keras.callbacks.EarlyStopping(monitor='val_F1', patience=5, mode='max', restore_best_weights=True),

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