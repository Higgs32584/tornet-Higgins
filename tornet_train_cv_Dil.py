import json
import logging
import os
import random
import shutil
import sys
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Multiply,
    PReLU,
    RandomRotation,
    ReLU,
    Reshape,
)
import tornet.data.tfds.tornet.tornet_dataset_builder
from tornet.data import preprocess as pp
from tornet.data.constants import ALL_VARIABLES, CHANNEL_MIN_MAX
from tornet.data.loader import get_dataloader
from tornet.data.preprocess import get_shape
from tornet.metrics.keras import metrics as tfm
from tornet.models.keras.layers import CoordConv2D
from tornet.utils.general import make_exp_dir

logging.basicConfig(level=logging.ERROR)
SEED = 1034
# Set random seeds for reproducibility
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# Environment Variables
DATA_ROOT = "/home/ubuntu/tfds"
TORNET_ROOT = DATA_ROOT
TFDS_DATA_DIR = DATA_ROOT
EXP_DIR = "."
os.environ["TORNET_ROOT"] = DATA_ROOT
os.environ["TFDS_DATA_DIR"] = TFDS_DATA_DIR
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
tf.config.optimizer.set_jit(True)

logging.info(f"TORNET_ROOT={DATA_ROOT}")


class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        base_lr,
        warmup_steps,
        restart_steps=5000,
        t_mul=2.0,
        m_mul=1.0,
        alpha=1e-6,
    ):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.alpha = alpha

        self.cosine_decay = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=base_lr,
            first_decay_steps=restart_steps,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=alpha,
        )

    def __call__(self, step):
        warmup_lr = self.base_lr * (
            tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32)
        )
        lr = tf.cond(
            step < self.warmup_steps,
            lambda: warmup_lr,
            lambda: self.cosine_decay(step - self.warmup_steps),
        )
        return lr


import tensorflow as tf
import numpy as np


class PrintBestAUCPR(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best_val_aucpr = 0.0

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        val_aucpr = logs.get("val_AUCPR")
        if val_aucpr is not None:
            if val_aucpr > self.best_val_aucpr:
                self.best_val_aucpr = val_aucpr
            print(
                f"Epoch {epoch + 1}: Best val_AUCPR so far = {self.best_val_aucpr:.5f}"
            )


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


def build_model(
    shape: Tuple[int] = (120, 240, 2),
    c_shape: Tuple[int] = (120, 240, 2),
    attention_dropout: float = 0.2,
    input_variables: List[str] = ALL_VARIABLES,
    dense_filters: int = 64,
    mid_filters: int = 64,
    start_filters: int = 64,
    l2_reg: float = 0.001,
    dropout_rate: float = 0.1,
) -> keras.Model:
    HE_INIT = "he_normal"
    GLOROT_INIT = "glorot_uniform"

    background_flag = -3.0
    include_range_folded = True
    inputs = {v: keras.Input(shape, name=v) for v in input_variables}

    n_sweeps = shape[2]
    x = keras.layers.Concatenate(axis=-1)(
        [normalize(inputs[v], v) for v in input_variables]
    )
    x = FillNaNs(background_flag)(x)
    if include_range_folded:
        range_folded = keras.Input(shape[:2] + (n_sweeps,), name="range_folded_mask")
        inputs["range_folded_mask"] = range_folded
        x = keras.layers.Concatenate(axis=-1)([x, range_folded])

    coords = keras.Input(c_shape, name="coordinates")
    inputs["coordinates"] = coords
    x = keras.layers.Concatenate(axis=-1)([x, coords])

    x = wide_resnet_block(
        x=x,
        stride=1,
        filters=start_filters,
        kernel_initializer=HE_INIT,
        l2_reg=l2_reg,
        drop_rate=dropout_rate,
    )
    x = se_block(x, kernel_initializer=GLOROT_INIT)
    x = multi_dilated_conv_block(
        x,
        filters=mid_filters,
        kernel_regularizer=l2_reg,
        kernel_initializer=HE_INIT,
    )
    x = BatchNormalization()(x)
    x = ReLU()(x)

    attn = Conv2D(
        1,
        1,
        activation="sigmoid",
        name="attention_map",
        kernel_initializer=GLOROT_INIT,
    )(x)
    attn = Dropout(rate=attention_dropout, name="attention_dropout")(attn)
    x = Multiply()([x, attn])

    x_avg = GlobalAveragePooling2D()(x)
    x_max = GlobalMaxPooling2D()(x)
    x = keras.layers.Concatenate()([x_avg, x_max])
    x = Dense(dense_filters, kernel_initializer=HE_INIT)(x)
    x = PReLU()(x)
    output = Dense(
        1, activation="sigmoid", dtype="float32", kernel_initializer=GLOROT_INIT
    )(x)

    return keras.Model(inputs=inputs, outputs=output)


def se_block(x, ratio=16, kernel_initializer="glorot_uniform", name=None):
    filters = x.shape[-1]
    se = GlobalAveragePooling2D(name=f"{name}_gap" if name else None)(x)
    se = Dense(
        filters // ratio,
        activation="relu",
        kernel_initializer=kernel_initializer,
        name=f"{name}_fc1" if name else None,
    )(se)
    se = Dense(
        filters,
        activation="sigmoid",
        kernel_initializer=kernel_initializer,
        name=f"{name}_fc2" if name else None,
    )(se)
    se = Reshape((1, 1, filters), name=f"{name}_reshape" if name else None)(se)
    x = Multiply(name=f"{name}_scale" if name else None)([x, se])
    return x


def multi_dilated_conv_block(
    x,
    filters,
    kernel_size=3,
    dilation_rates=[1, 2],
    kernel_regularizer=None,
    kernel_initializer="he_normal",
):
    convs = []
    for rate in dilation_rates:
        conv = BatchNormalization()(x)
        conv = ReLU()(conv)
        conv = Conv2D(
            filters,
            kernel_size=kernel_size,
            dilation_rate=rate,
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=keras.regularizers.l2(kernel_regularizer),
        )(conv)
        convs.append(conv)
    x = keras.layers.Concatenate()(convs)
    return x


def wide_resnet_block(
    x,
    filters,
    stride,
    l2_reg=1e-4,
    drop_rate=0.0,
    project_shortcut=False,
    kernel_initializer="he_normal",
):
    shortcut_x = x
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        kernel_initializer=kernel_initializer,
    )(x)
    if drop_rate > 0:
        x = Dropout(drop_rate)(x)

    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=stride,
        padding="same",
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        kernel_initializer=kernel_initializer,
    )(x)

    if project_shortcut or shortcut_x.shape[-1] != filters or stride != 1:
        shortcut_x = Conv2D(
            filters=filters,
            kernel_size=3,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
        )(shortcut_x)

    x = Add()([x, shortcut_x])
    return x


@keras.utils.register_keras_serializable()
class FastNormalize(keras.layers.Layer):
    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        self.mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        self.std = tf.convert_to_tensor(std, dtype=tf.float32)
        self._mean_list = (
            mean.numpy().tolist() if hasattr(mean, "numpy") else list(mean)
        )
        self._std_list = std.numpy().tolist() if hasattr(std, "numpy") else list(std)

    def call(self, x):
        return tf.math.subtract(x, self.mean) / (self.std + 1e-6)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mean": self._mean_list,
                "std": self._std_list,
            }
        )
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


gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


DEFAULT_CONFIG = {
    "epochs": 100,
    "input_variables": ["DBZ", "VEL", "KDP", "ZDR", "RHOHV", "WIDTH"],
    "train_years": [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "val_years": [2021, 2022],
    "batch_size": 64,
    "model": "wide_resnet",
    "learning_rate": 0.00441,
    "l2_reg": 3.83e-7,
    "label_smoothing": 0.0845,
    "dropout_rate": 0.0997,
    "warmup_epochs": 4,
    "t_mul": 2.0,
    "m_mul": 0.809,
    "attention_dropout": 0.322,
    "dense_filters": 64,
    "start_filters": 48,
    "mid_filters": 128,
    "wN": 1.0,
    "w0": 1.0,
    "w1": 1.0,
    "w2": 2.0,
    "wW": 0.5,
    "loss": "cce",
    "head": "maxpool",
    "exp_name": "tornado_baseline",
    "exp_dir": ".",
    "dataloader": "tensorflow-tfds",
    "dataloader_kwargs": {
        "select_keys": [
            "DBZ",
            "VEL",
            "KDP",
            "RHOHV",
            "ZDR",
            "WIDTH",
            "range_folded_mask",
            "coordinates",
        ]
    },
}


def main(config):
    # Gather all hyperparams
    epochs = config.get("epochs")
    batch_size = config.get("batch_size")
    start_filters = config.get("start_filters")
    dropout_rate = config.get("dropout_rate")
    lr = config.get("learning_rate")
    l2_reg = config.get("l2_reg")
    wN = config.get("wN")
    w0 = config.get("w0")
    w1 = config.get("w1")
    w2 = config.get("w2")
    wW = config.get("wW")
    input_variables = config.get("input_variables")
    train_years = config.get("train_years")
    val_years = config.get("val_years")
    dataloader = config.get("dataloader")
    dataloader_kwargs = config.get("dataloader_kwargs")

    logging.info(f"Using {dataloader} dataloader")
    logging.info(f"Running with config: {config}")
    weights = {"wN": wN, "w0": w0, "w1": w1, "w2": w2, "wW": wW}

    # Data Loaders
    dataloader_kwargs.update(
        {"select_keys": input_variables + ["range_folded_mask", "coordinates"]}
    )
    # Apply to Train and Validation Data
    ds_train = get_dataloader(
        dataloader,
        DATA_ROOT,
        train_years,
        "train",
        batch_size,
        weights,
        **dataloader_kwargs,
    )
    ds_val = get_dataloader(
        dataloader,
        DATA_ROOT,
        val_years,
        "train",
        batch_size,
        {"wN": 1.0, "w0": 1.0, "w1": 1.0, "w2": 1.0, "wW": 1.0},
        **dataloader_kwargs,
    )

    x, _, _ = next(iter(ds_train))

    in_shapes = (120, 240, get_shape(x)[-1])
    c_shapes = (120, 240, x["coordinates"].shape[-1])
    nn = build_model(
        attention_dropout=config["attention_dropout"],
        shape=in_shapes,
        c_shape=c_shapes,
        start_filters=start_filters,
        dense_filters=config["dense_filters"],
        mid_filters=config["mid_filters"],
        l2_reg=l2_reg,
        input_variables=input_variables,
        dropout_rate=dropout_rate,
    )
    print(nn.summary())
    from_logits = False
    steps_per_epoch = len(ds_train)

    lr = WarmUpCosine(
        base_lr=config["learning_rate"],
        warmup_steps=config["warmup_epochs"] * steps_per_epoch,
        restart_steps=10 * steps_per_epoch,
        t_mul=config["t_mul"],
        m_mul=config["m_mul"],
        alpha=1e-6,
    )

    loss = keras.losses.BinaryCrossentropy(
        from_logits=from_logits, label_smoothing=config["label_smoothing"]
    )
    opt = keras.optimizers.Adam(learning_rate=lr)
    metrics = [
        keras.metrics.AUC(
            from_logits=from_logits, curve="PR", name="AUCPR", num_thresholds=2000
        ),
        keras.metrics.AUC(from_logits=from_logits, name="AUC", num_thresholds=2000),
        tfm.BinaryAccuracy(from_logits, name="BinaryAccuracy"),
        tfm.TruePositives(from_logits, name="TruePositives"),
        tfm.FalsePositives(from_logits, name="FalsePositives"),
        tfm.TrueNegatives(from_logits, name="TrueNegatives"),
        tfm.FalseNegatives(from_logits, name="FalseNegatives"),
        tfm.Precision(from_logits, name="Precision"),
        tfm.Recall(from_logits, name="Recall"),
        tfm.F1Score(from_logits=from_logits, name="F1"),
    ]

    nn.compile(loss=loss, metrics=metrics, optimizer=opt, jit_compile=True)
    callbacks = [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.EarlyStopping(
            monitor="val_AUCPR", patience=3, mode="max", restore_best_weights=True
        ),
        PrintBestAUCPR(),
    ]

    history = nn.fit(
        ds_train, epochs=epochs, validation_data=ds_val, callbacks=callbacks, verbose=1
    )

    # Get Best Metrics
    best_aucpr = max(history.history.get("val_AUCPR", [0]))
    return {"AUCPR": best_aucpr}


if __name__ == "__main__":
    config = DEFAULT_CONFIG
    if len(sys.argv) > 1:
        config.update(json.load(open(sys.argv[1], "r")))

    # Define CV folds
    folds = [
        {
            "train_years": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
            "val_years": [2013, 2014],
        },
        {
            "train_years": [2013, 2014, 2017, 2018, 2019, 2020, 2021, 2022],
            "val_years": [2015, 2016],
        },
        {
            "train_years": [2013, 2014, 2015, 2016, 2019, 2020, 2021, 2022],
            "val_years": [2017, 2018],
        },
        {
            "train_years": [2013, 2014, 2015, 2016, 2017, 2018, 2021, 2022],
            "val_years": [2019, 2020],
        },
        {
            "train_years": [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
            "val_years": [2021, 2022],
        },
    ]

    results = []

    for i, fold in enumerate(folds):
        print(
            f"\n================ Fold {i + 1}: Train {fold['train_years']} | Test {fold['val_years']} ================\n"
        )
        fold_config = config.copy()
        fold_config["train_years"] = fold["train_years"]
        fold_config["val_years"] = fold["val_years"]
        fold_config["exp_name"] = f"{config['exp_name']}_fold{i+1}"
        fold_result = main(fold_config)
        results.append({"fold": i + 1, **fold_result})

    # Print Summary
    print("\n================ Cross-Validation Summary ================\n")
    for res in results:
        print(f"Fold {res['fold']} - AUCPR: {res['AUCPR']:.4f}")
    mean_aucpr = np.mean([res["AUCPR"] for res in results])
    print(f"\nMean AUCPR over all folds: {mean_aucpr:.4f}")
