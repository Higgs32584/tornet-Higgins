import json
import logging
import os
import random
import shutil
import sys
from typing import List, Tuple
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras.layers import (
    Add,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Conv2D,
    Conv2DTranspose,
    Activation,
    Dense,
    SpatialDropout2D,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Multiply,
    Concatenate,
    ReLU,
    Reshape,
)
import tensorflow as tf
import numpy as np
import tornet.data.tfds.tornet.tornet_dataset_builder
from tornet.data.constants import ALL_VARIABLES, CHANNEL_MIN_MAX
from tornet.data.loader import get_dataloader
from tornet.data.preprocess import get_shape
from tornet.metrics.keras import metrics as tfm
from tornet.utils.general import make_exp_dir
from tensorflow.keras.regularizers import l2

logging.basicConfig(level=logging.ERROR)
SEED = 230
# Set random seeds for reproducibility
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# Environment Variables
DATA_ROOT = os.getenv("DATA_ROOT", "/home/ubuntu/tfds")
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
        self.restart_steps = restart_steps
        self.t_mul = t_mul
        self.m_mul = m_mul
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

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "restart_steps": self.restart_steps,
            "t_mul": self.t_mul,
            "m_mul": self.m_mul,
            "alpha": self.alpha,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.utils.register_keras_serializable()
class FillNaNs(keras.layers.Layer):
    def __init__(self, fill_val, **kwargs):
        super().__init__(**kwargs)
        self.initial_fill_val = fill_val
        self.fill_val = tf.constant(
            fill_val, dtype=tf.float32
        )  # stored as float32 by default

    @tf.function(jit_compile=True)
    def call(self, x):
        fill_val = tf.cast(
            self.fill_val, x.dtype
        )  # Cast to match input dtype (e.g., float16)
        return tf.where(tf.math.is_nan(x), fill_val, x)

    def get_config(self):
        return {
            **super().get_config(),
            "fill_val": self.initial_fill_val,
        }


def conv_block(x, filters, kernel_size=3, padding="same", activation="relu"):
    x = Conv2D(filters, kernel_size, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(filters, kernel_size, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x


from tensorflow.keras.layers import ZeroPadding2D


def build_model(
    shape: Tuple[int] = (120, 240, 2),
    c_shape: Tuple[int] = (120, 240, 2),
    input_variables: List[str] = ALL_VARIABLES,
    filters=64,
) -> keras.Model:
    background_flag = -3.0
    include_range_folded = True
    inputs = {v: keras.Input(shape, name=v) for v in input_variables}

    # Merge normalized radar variables
    x = keras.layers.Concatenate(axis=-1)(
        [normalize(inputs[v], v) for v in input_variables]
    )
    x = FillNaNs(background_flag)(x)

    if include_range_folded:
        rf_input = keras.Input(shape[:2] + (shape[2],), name="range_folded_mask")
        inputs["range_folded_mask"] = rf_input
        x = Concatenate(axis=-1)([x, rf_input])

    coords = keras.Input(c_shape, name="coordinates")
    inputs["coordinates"] = coords
    x = Concatenate(axis=-1)([x, coords])
    x = ZeroPadding2D(padding=((4, 4), (8, 8)))(x)

    c1 = conv_block(x, filters)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, filters * 2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, filters * 4)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, filters * 8)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    bn = conv_block(p4, filters * 16)

    # Decoder
    u1 = Conv2DTranspose(filters * 8, (2, 2), strides=(2, 2), padding="same")(bn)
    u1 = Concatenate()([u1, c4])
    c5 = conv_block(u1, filters * 8)

    u2 = Conv2DTranspose(filters * 4, (2, 2), strides=(2, 2), padding="same")(c5)
    u2 = Concatenate()([u2, c3])
    c6 = conv_block(u2, filters * 4)

    u3 = Conv2DTranspose(filters * 2, (2, 2), strides=(2, 2), padding="same")(c6)
    u3 = Concatenate()([u3, c2])
    c7 = conv_block(u3, filters * 2)

    u4 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(c7)
    u4 = Concatenate()([u4, c1])
    c8 = conv_block(u4, filters)

    max = GlobalAveragePooling2D()(c8)  # shape becomes (None, filters)
    avg = GlobalMaxPooling2D()(c8)  # shape becomes (None, filters)
    x = Concatenate()([max, avg])  # shape becomes (None, filters * 2)
    x = Dropout(0.2)(x)  # optional
    output = Dense(1, activation="sigmoid")(x)  # shape becomes (None,)
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


class Float32BinaryCrossentropy(tf.keras.losses.Loss):
    def __init__(
        self,
        from_logits=False,
        label_smoothing=0.0,
        reduction="sum_over_batch_size",  # was "auto", now fixed
        name="float32_bce",
    ):
        super().__init__(reduction=reduction, name=name)
        self.inner_loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            reduction=reduction,
            name=name,
        )

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return self.inner_loss(y_true, y_pred)


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
        x = SpatialDropout2D(drop_rate)(x)

    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
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
        self.std = tf.maximum(std, 1e-6)  # Ensure std is never zero

        self._mean_list = (
            mean.numpy().tolist() if hasattr(mean, "numpy") else list(mean)
        )
        self._std_list = std.numpy().tolist() if hasattr(std, "numpy") else list(std)

    def call(self, x):
        mean = tf.cast(self.mean, x.dtype)
        std = tf.cast(self.std, x.dtype)
        return tf.math.subtract(x, mean) / (std + 1e-6)

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


gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")


DEFAULT_CONFIG = {
    "epochs": 100,
    "input_variables": ["DBZ", "VEL", "KDP", "ZDR", "RHOHV", "WIDTH"],
    "train_years": [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "val_years": [2021, 2022],
    "batch_size": 64,
    "model": "wide_resnet",
    "learning_rate": 0.003,
    "l2_reg": 3e-7,
    "label_smoothing": 0.01,
    "dropout_rate": 0.12,
    "warmup_epochs": 3,
    "t_mul": 2.0,
    "m_mul": 0.9,
    "cycle_restart": 10,
    "early_stopping_patience": 5,
    "dense_filters": 128,
    "start_filters": 16,
    "mid_filters": 64,
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


@keras.utils.register_keras_serializable()
class GeMPooling2D(tf.keras.layers.Layer):
    def __init__(self, p_init=3.0, trainable=True, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.p_init = p_init
        self.trainable_p = trainable
        self.eps = eps

    def build(self, input_shape):
        # Always keep variable in float32, but cast during call
        self.p = self.add_weight(
            name="gem_p",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.p_init),
            trainable=self.trainable_p,
            dtype=tf.float32,
        )

    def call(self, inputs):
        p = tf.cast(self.p, inputs.dtype)  # match dtype (e.g. float16)
        x = tf.clip_by_value(inputs, self.eps, tf.reduce_max(inputs))
        x = tf.pow(x, p)
        x = tf.reduce_mean(x, axis=[1, 2])
        x = tf.pow(x, 1.0 / p)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "p_init": self.p_init,
                "trainable": self.trainable_p,
                "eps": self.eps,
            }
        )
        return config


def multi_dilated_attention_head(x, dilation_rates=[1, 2, 4], name_prefix="attn"):
    branches = []
    for i, rate in enumerate(dilation_rates):
        attn = Conv2D(
            filters=1,
            kernel_size=1,
            dilation_rate=rate,
            padding="same",
            activation="sigmoid",
            name=f"{name_prefix}_d{rate}",
        )(x)
        branches.append(attn)

    x_avg = GlobalAveragePooling2D()(x)
    x_max = GlobalMaxPooling2D()(x)
    context = Concatenate()([x_avg, x_max])
    attention_gates = []
    for i in range(len(branches)):
        gate = Dense(1, activation="sigmoid", name=f"{name_prefix}_gate_d{i}")(context)
        gate = Reshape((1, 1, 1), name=f"{name_prefix}_reshape_d{i}")(gate)
        attention_gates.append(gate)

    # Apply per-branch gates to attention maps
    weighted_attn_maps = []
    for i, attn_map in enumerate(branches):
        weighted = Multiply(name=f"{name_prefix}_scale_d{i}")(
            [attn_map, attention_gates[i]]
        )
        weighted_attn_maps.append(weighted)

    # Fuse gated attention maps
    fused_attn = Add(name=f"{name_prefix}_fused")(weighted_attn_maps)

    return fused_attn


def multi_dilated_attention_block(
    x,
    filters,
    kernel_size=3,
    kernel_regularizer=None,
    dilation_rates=[1, 2, 4],
    name_prefix="mdab",
):
    branches = []
    for i, rate in enumerate(dilation_rates):
        b = Conv2D(
            filters,
            kernel_size,
            dilation_rate=rate,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(kernel_regularizer),
            use_bias=False,
            name=f"{name_prefix}_sepconv_d{rate}",
        )(x)
        b = BatchNormalization(name=f"{name_prefix}_bn_d{rate}")(b)
        b = ReLU(name=f"{name_prefix}_relu_d{rate}")(b)
        branches.append(b)

    avg_pool = GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
    max_pool = GlobalMaxPooling2D(name=f"{name_prefix}_gmp")(x)
    context = keras.layers.Concatenate(name=f"{name_prefix}_context")(
        [avg_pool, max_pool]
    )
    weighted_branches = []
    for i, branch in enumerate(branches):
        weight = Dense(
            filters, activation="sigmoid", name=f"{name_prefix}_dense_weight_d{i}"
        )(
            context
        )  # shape: (batch, filters)
        weight = Reshape((1, 1, filters), name=f"{name_prefix}_reshape_weight_d{i}")(
            weight
        )
        weighted = Multiply(name=f"{name_prefix}_gated_branch_d{i}")([branch, weight])
        weighted_branches.append(weighted)

    # Fuse weighted branches
    output = Add(name=f"{name_prefix}_fused")(weighted_branches)

    return output


def create_fold_expdir(base_expdir, fold):
    fold_expdir = os.path.join(base_expdir, f"fold_{fold}")
    if not os.path.exists(fold_expdir):
        os.makedirs(fold_expdir)
    return fold_expdir


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
    exp_dir = config.get("exp_dir")
    exp_name = config.get("exp_name")
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
    ds_train = get_dataloader(
        dataloader,
        DATA_ROOT,
        years=train_years,
        data_type="train",
        batch_size=batch_size,
        weights=weights,
        random_state=SEED,
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

    data = next(iter(ds_train))
    if isinstance(data, tuple) and len(data) == 3:
        x, _, _ = data
    else:
        raise ValueError(
            "Expected dataset to yield three outputs, but got: {}".format(data)
        )

    in_shapes = (120, 240, get_shape(x)[-1])
    c_shapes = (120, 240, x["coordinates"].shape[-1])
    nn = build_model(
        shape=in_shapes,
        c_shape=c_shapes,
        filters=start_filters,
        input_variables=input_variables,
    )
    from_logits = False
    steps_per_epoch = len(ds_train)

    lr = WarmUpCosine(
        base_lr=config["learning_rate"],
        warmup_steps=config["warmup_epochs"] * steps_per_epoch,
        restart_steps=config["cycle_restart"] * steps_per_epoch,
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
            from_logits=from_logits, curve="PR", name="aucpr", num_thresholds=2000
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
    print(nn.summary())
    # Experiment Directory
    fold = config["fold"]
    fold_expdir = make_exp_dir(exp_dir=exp_dir, prefix=f"{exp_name}_fold_{fold}_")
    print(fold_expdir)
    shutil.copy(__file__, os.path.join(fold_expdir, "train.py"))

    checkpoint_name = os.path.join(
        fold_expdir, "epoch_{epoch:03d}_valAUCPR_{val_aucpr:.4f}.keras"
    )
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_name,
            monitor="val_aucpr",  # Ensure this matches the metric name defined in metrics
            save_best_only=True,
            mode="max",
        ),
        keras.callbacks.CSVLogger(os.path.join(fold_expdir, "history.csv")),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.EarlyStopping(
            monitor="val_aucpr",
            patience=config["early_stopping_patience"],
            mode="max",
            restore_best_weights=True,
        ),
    ]

    history = nn.fit(
        ds_train, epochs=epochs, validation_data=ds_val, callbacks=callbacks, verbose=1
    )

    best_aucpr = max(history.history.get("val_aucpr", [0]))
    return {"aucpr": best_aucpr}


# Cross-validation loop
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
        fold_config["fold"] = i + 1  # Pass fold number here
        fold_result = main(fold_config)
        results.append({"fold": i + 1, **fold_result})
        from tensorflow.keras import backend as K

        K.clear_session()

    # Print Summary
    print("\n================ Cross-Validation Summary ================\n")
    for res in results:
        print(f"Fold {res['fold']} - AUCPR: {res['aucpr']:.4f}")
    mean_aucpr = np.mean([res["aucpr"] for res in results])
    print(f"\nMean AUCPR over all folds: {mean_aucpr:.4f}")
