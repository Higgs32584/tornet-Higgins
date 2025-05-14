import os

os.environ["KERAS_BACKEND"] = (
    "tensorflow"  # set to 'tensorflow', 'torch' or 'jax' (installs required)
)

# Uncomment if tornet isn't installed in your environment or in your path already
# sys.path.append('../')
import glob
import sys

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tornet.data.constants import ALL_VARIABLES

# keras accepts most data loaders (tensorflow, torch).
# A pure keras data loader, with necessary preprocessing steps for the cnn baseline, is provided
from tornet.data.keras.loader import KerasDataLoader
from tornet.data.tf.loader import create_tf_dataset

data_root = "/fsx/ns1"
ds = KerasDataLoader(
    data_root=data_root,
    data_type="train",
    years=[2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    workers=4,
    batch_size=128,
    select_keys=ALL_VARIABLES + ["range_folded_mask", "coordinates"],
    use_multiprocessing=True,
)
print(dir(ds))
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Layer,
    Normalization,
    Reshape,
    multiply,
)
from tensorflow.keras.optimizers import Adam

# Updated Tornado Prediction CNN Model
from tornet.data.constants import CHANNEL_MIN_MAX
from tornet.models.keras.layers import CoordConv2D, FillNaNs


# Focal Loss for Class Imbalance
def focal_loss(gamma=2.0, alpha=0.25):
    import tensorflow as tf

    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        return tf.reduce_mean(alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy)

    return focal_loss_fixed


# Define input variables
input_vars = ALL_VARIABLES
import keras
import numpy as np
from keras.layers import (
    Concatenate,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPool2D,
    Normalization,
)
from keras.models import Model

from tornet.models.keras.layers import CoordConv2D, FillNaNs

# Input & Normalization
inputs = {v: Input(shape=(120, 240, 2), name=v) for v in ALL_VARIABLES}
norm_layers = [
    Normalization(
        mean=np.array(2 * [(CHANNEL_MIN_MAX[v][0] + CHANNEL_MIN_MAX[v][1]) / 2]),
        variance=np.array(
            2 * [((CHANNEL_MIN_MAX[v][1] - CHANNEL_MIN_MAX[v][0]) / 2) ** 2]
        ),
        name=f"Normalized_{v}",
    )
    for v in ALL_VARIABLES
]

x = Concatenate(axis=-1, name="Concatenate1")(
    [l(inputs[v]) for l, v in zip(norm_layers, ALL_VARIABLES)]
)
x = FillNaNs(fill_val=-3, name="ReplaceNan")(x)
include_range_folded = True  # Set this based on config
if include_range_folded:
    range_folded = Input(shape=(120, 240, 2), name="range_folded_mask")
    inputs["range_folded_mask"] = range_folded
    x = Concatenate(axis=-1, name="Concatenate2")([x, range_folded])


from tornet.models.keras.cnn_baseline import vgg_block

filters, dropouts = [32, 64, 128, 256], [0.2, 0.2, 0.3, 0.3]
for f, d in zip(filters, dropouts):
    x = CoordConv2D(
        f,
        (3, 3),
        strides=(2, 2),
        padding="same",
        kernel_regularizer=keras.regularizers.l2(1e-6),
        activation="relu",
    )(x)
    x = MaxPool2D(pool_size=2, strides=2, padding="same")(x)
    x = Dropout(d)(x)

x = keras.layers.Flatten()(x)
x = keras.layers.Dense(units=4096, activation="relu")(x)
x = keras.layers.Dense(units=2024, activation="relu")(x)
y = keras.layers.Dense(1)(x)
# Create Model
model = build_model(
    shape=in_shapes,
    c_shape=c_shapes,
    start_filters=start_filters,
    l2_reg=l2_reg,
    input_variables=input_variables,
    head=head,
)


model = Model(inputs=inputs, outputs=y, name="OptimizedTornadoDetector")
model.summary()
model.compile(optimizer=Adam(learning_rate=1e-3), loss=focal_loss(), metrics=["AUC"])
# ==================== 🌟 Imports 🌟 ==================== #
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.metrics import AUC, BinaryAccuracy, Precision, Recall
from tensorflow.keras.optimizers import Adam, AdamW

monitor = "pr_auc"  # Monitor Precision-Recall AUC for imbalanced data
# ==================== 🏃 Callbacks 🏃 ==================== #
# 🛑 Early stopping to prevent overfitting and restore the best weights
early_stopping = EarlyStopping(
    monitor=monitor,
    patience=4,  # Allow some epochs for PR AUC improvement
    mode="max",  # Because higher pr_auc is better
    restore_best_weights=True,  # Reload the best weights when stopping
)

# 📉 Dynamic LR scheduler to adjust learning rate when PR AUC stalls
reduce_lr = ReduceLROnPlateau(
    monitor=monitor,
    factor=0.5,  # Halve the LR if performance plateaus
    patience=2,  # Give more time before reducing LR
    min_lr=1e-6,  # Minimum LR to prevent over-reduction
    mode="max",
    verbose=1,
)


# ==================== 🎯 Custom Loss 🎯 ==================== #
# ⚖️ Weighted binary cross-entropy to handle class imbalance
def weighted_binary_crossentropy(pos_weight):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        weight = y_true * pos_weight + (
            1 - y_true
        )  # Heavier penalty for false negatives
        return tf.reduce_mean(bce * weight)

    return loss


import keras.backend as K
import tensorflow as tf

# ==================== 🚀 Optimizer & Metrics 🚀 ==================== #
# ⚡ AdamW optimizer with weight decay for regularization
opt = Adam(learning_rate=1e-3)

# 📊 Metrics to evaluate model performance beyond accuracy
metrics = [
    AUC(curve="PR", name="pr_auc"),  # Precision-Recall AUC for imbalanced data
    AUC(name="AUC"),
]


# ==================== 🔧 Compile Model 🔧 ==================== #
# 🛠️ Compile the model with custom loss and chosen metrics
model.compile(
    optimizer=opt,
    loss=weighted_binary_crossentropy(15),  # Adjust pos_weight as needed (e.g., 10, 20)
    metrics=metrics,
)

# ==================== 🏃‍♂️ Train Model 🏃‍♂️ ==================== #
# 🚀 Training the model with callbacks for early stopping and adaptive LR

import time

import numpy as np
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm

# Example validation steps
val_steps = 100  # Adjust based on your validation dataset size

# Updated model.fit with validation progress and metrics callback
history = model.fit(
    ds,
    validation_data=None,  # Validation is handled by the custom callback
    epochs=20,
    steps_per_epoch=1341,
    callbacks=[early_stopping, reduce_lr],
)

# ==================== 📈 Final Notes 📈 ==================== #
# - Consider tuning 'pos_weight' further for optimal PR AUC.
# - Experiment with 'BinaryFocalCrossentropy' if PR AUC remains low:
#     model.compile(optimizer=opt, loss=BinaryFocalCrossentropy(gamma=2.0), metrics=metrics)
# - Check PR AUC thresholds post-training for best classification cutoff.
# Build a test set
ds_test = KerasDataLoader(
    data_root=data_root,
    data_type="test",
    years=[2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    batch_size=128,
    workers=4,
    select_keys=ALL_VARIABLES,
    use_multiprocessing=True,
)
# Evaluate
import tornet.metrics.keras.metrics as km

metrics = [
    keras.metrics.AUC(curve="pr", name="AUC"),
    keras.metrics.Precision(),
    keras.metrics.Recall(),
]
from tensorflow.keras.optimizers import AdamW

model.compile(metrics=metrics)
# model.compile(optimizer=AdamW(learning_rate=.001),loss=BinaryFocalCrossentropy(),metrics=metrics)
# steps=10 for demo purposes
model.evaluate(ds_test, steps=105, return_dict=True)


import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Generates a timestamp
model_name = f"tornado_model_{timestamp}.keras"  # Unique filename
model.save(model_name)  # Save model
print(f"Model saved as {model_name}")
