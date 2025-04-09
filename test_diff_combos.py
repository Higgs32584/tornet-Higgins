import os
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, AdamW
from tensorflow.keras.optimizers.schedules import (
    CosineDecayRestarts,
    ExponentialDecay,
    PolynomialDecay,
    LearningRateSchedule
)

from custom_func import combo_loss, FalseAlarmRate, ThreatScore,tversky_loss,focal_loss
from tornet.data.loader import get_dataloader
from tornet.data.preprocess import get_shape
from tornet.data.constants import ALL_VARIABLES
from tornet.metrics.keras import metrics as tfm
DATA_ROOT = '/home/ubuntu/tfds'
TORNET_ROOT=DATA_ROOT
TFDS_DATA_DIR = DATA_ROOT
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
os.environ['TORNET_ROOT'] = DATA_ROOT
os.environ['TFDS_DATA_DIR'] = TFDS_DATA_DIR
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
# Reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
import random
random.seed(SEED)
# Paths
DATA_ROOT = '/home/ubuntu/tfds'
LOG_DIR = os.path.join("loss_opt_scheduler_runs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(LOG_DIR, exist_ok=True)

# Basic config
CONFIG = {
    "epochs": 15,
    "batch_size": 128,
    "train_years": [2014, 2015],
    "val_years": [2016],
    "input_variables": ["DBZ", "VEL", "KDP", "ZDR", "RHOHV", "WIDTH"],
    "weights": {"wN": 0.1, "w0": 1.0, "w1": 3.0, "w2": 10.0, "wW": 0.2},
    "dataloader": "tensorflow-tfds",
    "dataloader_kwargs": {
        "select_keys": ["DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH", "range_folded_mask", "coordinates"]
    },
    "model": "wide_resnet",
    "start_filters": 48,
    "dropout_rate": 0.2,
    "l2_reg": 1e-6,
    "widen_factor": 2,
    "nconvs": 2,
    "attention_type": "se",
    "multi_scale": "none",
    "loss_config": {
        "alpha": 0.75,
        "alpha_tversky": 0.4,
        "beta_tversky": 0.45,
        "focal_gamma": 1.5,
        "focal_alpha": 0.8,
    }
}


from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.optimizers.schedules import (
    CosineDecayRestarts, ExponentialDecay, PolynomialDecay
)
from tensorflow_addons.optimizers import Lookahead, RectifiedAdam
from adabelief_tf import AdaBeliefOptimizer
LOSSES = {
    "binary_crossentropy": keras.losses.BinaryCrossentropy(from_logits=False),
    "asymmetric_focal": lambda gamma_pos=2.0, gamma_neg=1.0, alpha=0.25: 
        lambda y_true, y_pred: -tf.reduce_mean(
            y_true * alpha * tf.pow(1 - y_pred, gamma_pos) * tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.0)) +
            (1 - y_true) * (1 - alpha) * tf.pow(y_pred, gamma_neg) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-7, 1.0))
        ),
    "log_cosh_dice": lambda smooth=1e-6: 
        lambda y_true, y_pred: tf.math.log(
            tf.cosh(1 - (
                (2 * tf.reduce_sum(y_true * y_pred) + smooth) /
                (tf.reduce_sum(y_true + y_pred) + smooth)
            ))
        ),

}

SCHEDULERS = {
    "cosine_decay": lambda lr: CosineDecayRestarts(initial_learning_rate=lr, first_decay_steps=2000),
    "exp_decay": lambda lr: ExponentialDecay(initial_learning_rate=lr, decay_steps=1000, decay_rate=0.96),
    "poly_decay": lambda lr: PolynomialDecay(initial_learning_rate=lr, decay_steps=2000, end_learning_rate=1e-5)
}

OPTIMIZERS = {
    "adam": lambda lr: Adam(learning_rate=lr),
    "sgd": lambda lr: SGD(learning_rate=lr, momentum=0.9),
    "adamw": lambda lr: AdamW(learning_rate=lr, weight_decay=1e-6),
    "rmsprop": lambda lr: RMSprop(learning_rate=lr)
}
import sys
import os
import json
import shutil
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    BatchNormalization,
    Add,
    MaxPool2D,
    Dropout,
    GlobalAveragePooling2D,
    Dense,
    Multiply,
    Conv2D,
    GlobalMaxPooling2D,
    Reshape,
    ReLU
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from typing import List, Tuple
from tornet.data.loader import get_dataloader
from tornet.data import preprocess as pp
from tornet.data.preprocess import get_shape
from tornet.data.constants import ALL_VARIABLES, CHANNEL_MIN_MAX
from tornet.metrics.keras import metrics as tfm
from tornet.utils.general import make_exp_dir
from custom_func import FalseAlarmRate, ThreatScore
import tensorflow_datasets as tfds
from tornet.models.keras.layers import CoordConv2D
import tornet.data.tfds.tornet.tornet_dataset_builder
import random

logging.basicConfig(level=logging.ERROR)
SEED = 42  
# Set random seeds for reproducibility
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
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"


logging.info(f'TORNET_ROOT={DATA_ROOT}')

@keras.utils.register_keras_serializable()
class FillNaNs(keras.layers.Layer):
    def __init__(self, fill_val, **kwargs):
        super().__init__(**kwargs)
        self.fill_val = tf.convert_to_tensor(fill_val, dtype=tf.float32)
    @tf.function(jit_compile=True)
    def call(self, x):
        return tf.where(tf.math.is_nan(x), self.fill_val, x)

    def get_config(self):
        return {**super().get_config(), "fill_val": self.fill_val.numpy().item()}
@keras.utils.register_keras_serializable()
class FastNormalize(keras.layers.Layer):
    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        self.mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        self.std = tf.convert_to_tensor(std, dtype=tf.float32)
        self._mean_list = mean.numpy().tolist() if hasattr(mean, 'numpy') else list(mean)
        self._std_list = std.numpy().tolist() if hasattr(std, 'numpy') else list(std)

    def call(self, x):
        return tf.math.subtract(x, self.mean) / (self.std + 1e-6)

    def get_config(self):
        config = super().get_config()
        config.update({
            "mean": self._mean_list,
            "std": self._std_list,
        })
        return config
    
    
def normalize(x, name: str):
    min_val, max_val = CHANNEL_MIN_MAX[name]
    mean = np.float32((max_val + min_val) / 2)
    std = np.float32((max_val - min_val) / 2)
    n_sweeps = x.shape[-1]

    # Use tf.constant directly for faster graph compilation
    mean = tf.constant([mean] * n_sweeps, dtype=tf.float32)
    std = tf.constant([std] * n_sweeps, dtype=tf.float32)

    return FastNormalize(mean, std)(x)
# Enable GPU Memory Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def build_model(model='wide_resnet',shape:Tuple[int]=(120,240,2),
                c_shape:Tuple[int]=(120,240,2),
                input_variables:List[str]=ALL_VARIABLES,
                start_filters:int=64,
                nconvs:int=2,
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
    
    x,c = normalized_inputs,cin
    
    if model == 'wide_resnet':
        x, c = wide_resnet_block(x, c, filters=start_filters, widen_factor=2, l2_reg=l2_reg,nconvs=nconvs, drop_rate=dropout_rate)
        x, c = wide_resnet_block(x, c, filters=start_filters*2, widen_factor=2, l2_reg=l2_reg,nconvs=nconvs, drop_rate=dropout_rate)
        x, c = wide_resnet_block(x, c, filters=start_filters*4, widen_factor=2, l2_reg=l2_reg,nconvs=nconvs, drop_rate=dropout_rate)
        x=se_block(x)
    x = Conv2D(128, 3, padding='same', use_bias=False)(x)  # <-- no bias
    x = BatchNormalization()(x)
    x = ReLU()(x)
    attention_map = Conv2D(1, 1, activation='sigmoid', name='attention_map',use_bias=False)(x)  # shape (B, H, W, 1)
    attention_map = Dropout(rate=0.2, name='attention_dropout')(attention_map)
    x_weighted = Multiply()([x, attention_map])

    x_avg = GlobalAveragePooling2D()(x_weighted)
    x_max = GlobalMaxPooling2D()(x_weighted)
    x_concat = keras.layers.Concatenate()([x_avg, x_max])


    x_dense = Dense(64, activation='relu')(x_concat)
    output = Dense(1, activation='sigmoid', dtype='float32')(x_dense)
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
    x = ReLU()(x)
    x = Add()([x, shortcut_x])

    # Pooling and dropout
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    c = MaxPool2D(pool_size=2, strides=2, padding='same')(c)

    if drop_rate > 0:
        x = Dropout(rate=drop_rate)(x)
    
    return x, c






def run_experiment(loss_name, optimizer_name, scheduler_name):
    # Setup dataloaders
    import tornet.data.tfds.tornet.tornet_dataset_builder
    import tensorflow_datasets as tfds

    ds_train = get_dataloader(CONFIG["dataloader"], DATA_ROOT, CONFIG["train_years"], "train",
                              CONFIG["batch_size"], CONFIG["weights"], **CONFIG["dataloader_kwargs"])
    ds_val = get_dataloader(CONFIG["dataloader"], DATA_ROOT, CONFIG["val_years"], "train",
                            CONFIG["batch_size"], {k: 1.0 for k in CONFIG["weights"]},
                            **CONFIG["dataloader_kwargs"])

    x_sample, _, _ = next(iter(ds_train))
    input_shape = (None, None, get_shape(x_sample)[-1])
    coord_shape = (None, None, x_sample["coordinates"].shape[-1])

    model = build_model(
        shape=input_shape,
        c_shape=coord_shape,
        input_variables=CONFIG["input_variables"],
        start_filters=CONFIG["start_filters"],
        l2_reg=CONFIG["l2_reg"],
    )

    # Learning rate setup
    base_lr = 1e-3
    scheduler = SCHEDULERS[scheduler_name](base_lr)
    optimizer = OPTIMIZERS[optimizer_name](scheduler)
    loss_fn = LOSSES[loss_name]
    from_logits=False
    model.compile(
        loss=loss_fn,
        optimizer=optimizer,
    metrics = [keras.metrics.AUC(from_logits=from_logits,curve='PR',name='AUCPR',num_thresholds=1000), 
                tfm.BinaryAccuracy(from_logits,name='BinaryAccuracy'), 
                tfm.TruePositives(from_logits,name='TruePositives'),
                tfm.FalsePositives(from_logits,name='FalsePositives'), 
                tfm.TrueNegatives(from_logits,name='TrueNegatives'),
                tfm.FalseNegatives(from_logits,name='FalseNegatives'), 
                tfm.Precision(from_logits,name='Precision'), 
                tfm.Recall(from_logits,name='Recall'),
                tfm.F1Score(from_logits=from_logits,name='F1')],
        jit_compile=True
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_AUCPR", patience=3, restore_best_weights=True)
    ]

    print(f"🔧 Training: Loss={loss_name}, Optimizer={optimizer_name}, Scheduler={scheduler_name}")
    history = model.fit(ds_train, validation_data=ds_val, epochs=CONFIG["epochs"], callbacks=callbacks, verbose=1)

    best_aucpr = max(history.history.get("val_AUCPR", [0]))

    result = {
        "loss": loss_name,
        "optimizer": optimizer_name,
        "scheduler": scheduler_name,
        "best_val_AUCPR": best_aucpr
    }

    with open(os.path.join(LOG_DIR, f"{loss_name}_{optimizer_name}_{scheduler_name}.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"✅ Done: Best AUCPR = {best_aucpr:.4f}")
    return result


if __name__ == "__main__":
    results = []
    for loss_name in LOSSES:
        for optimizer_name in OPTIMIZERS:
            for scheduler_name in SCHEDULERS:
                try:
                    res = run_experiment(loss_name, optimizer_name, scheduler_name)
                    results.append(res)
                except tf.errors.ResourceExhaustedError:
                    print(f"❌ Skipped: OOM - {loss_name}, {optimizer_name}, {scheduler_name}")
    
    summary_path = os.path.join(LOG_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📝 Summary written to {summary_path}")
