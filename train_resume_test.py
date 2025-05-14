import json
import logging
import os
import shutil
import subprocess  # To check instance interruptions
import sys
import time

import keras
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import tornet.data.tfds.tornet.tornet_dataset_builder
from tornet.data.constants import ALL_VARIABLES
from tornet.data.loader import get_dataloader
from tornet.data.preprocess import get_shape
from tornet.metrics.keras import metrics as tfm
from tornet.models.keras.cnn_experiment import build_model_exp
from tornet.models.keras.losses import mae_loss
from tornet.utils.general import make_callback_dirs, make_exp_dir

logging.basicConfig(level=logging.INFO)

# Environment Variables
EXP_DIR = "s3://tornet-checkpoints"
DATA_ROOT = "/home/ubuntu/tfds"
TFDS_DATA_DIR = "/home/ubuntu/tfds"
os.environ["TORNET_ROOT"] = DATA_ROOT
os.environ["TFDS_DATA_DIR"] = TFDS_DATA_DIR
logging.info(f"TORNET_ROOT={DATA_ROOT}")

# Enable GPU Memory Growth
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

strategy = tf.distribute.MirroredStrategy()
tf.config.optimizer.set_jit(True)  # Enable XLA (Accelerated Linear Algebra)
logging.info(f"Number of devices: {strategy.num_replicas_in_sync}")

# Default Configuration
DEFAULT_CONFIG = {
    "epochs": 200,
    "input_variables": ALL_VARIABLES,
    "train_years": list(range(2013, 2020)),
    "val_years": list(range(2020, 2022)),
    "batch_size": 128,
    "model": "inception",
    "start_filters": 64,
    "learning_rate": 1e-3,
    "decay_steps": 2500,
    "decay_rate": 0.92,
    "l2_reg": 1e-5,
    "wN": 1.0,
    "w0": 1.0,
    "w1": 1.0,
    "w2": 2.0,
    "wW": 0.5,
    "label_smooth": 0.1,
    "loss": "cce",
    "head": "maxpool",
    "exp_name": "tornet_baseline",
    "exp_dir": EXP_DIR,
    "dataloader": "tensorflow-tfds",
    "dataloader_kwargs": {},
}


def check_spot_instance():
    """Checks if the Spot Instance is about to be terminated."""
    try:
        result = subprocess.run(
            [
                "curl",
                "-s",
                "http://169.254.169.254/latest/meta-data/spot/termination-time",
            ],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() != ""
    except Exception:
        return False  # Assume no termination if check fails


def main(config):
    """Main training function with spot instance resilience."""

    # Load Training Configuration
    epochs = config.get("epochs")
    batch_size = config.get("batch_size")
    learning_rate = config.get("learning_rate")
    decay_steps = config.get("decay_steps")
    decay_rate = config.get("decay_rate")
    input_variables = config.get("input_variables")
    exp_name = config.get("exp_name")
    exp_dir = config.get("exp_dir")
    dataloader = config.get("dataloader")
    dataloader_kwargs = config.get("dataloader_kwargs")

    logging.info(f"Using {dataloader} dataloader")
    logging.info(f"Running with config: {config}")

    # Data Loaders
    dataloader_kwargs.update(
        {"select_keys": input_variables + ["range_folded_mask", "coordinates"]}
    )
    ds_train = get_dataloader(
        dataloader,
        DATA_ROOT,
        config["train_years"],
        "train",
        batch_size,
        {},
        **dataloader_kwargs,
    )
    ds_val = get_dataloader(
        dataloader,
        DATA_ROOT,
        config["val_years"],
        "train",
        batch_size,
        {},
        **dataloader_kwargs,
    )

    # Model Input Shapes
    x, _, _ = next(iter(ds_train))
    in_shapes = (None, None, get_shape(x)[-1])
    c_shapes = (None, None, x["coordinates"].shape[-1])

    # Build Model
    nn = build_model_exp(
        shape=in_shapes,
        c_shape=c_shapes,
        start_filters=config["start_filters"],
        l2_reg=config["l2_reg"],
        input_variables=input_variables,
        model=config["model"],
    )
    print(nn.summary())

    # Loss & Optimizer
    loss = keras.losses.BinaryCrossentropy(label_smoothing=config["label_smooth"])
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate, decay_steps, decay_rate
    )
    opt = keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=1e-4, amsgrad=True
    )

    # Metrics
    metrics = [keras.metrics.AUC(name="AUCPR", curve="PR", num_thresholds=2000)]

    nn.compile(loss=loss, metrics=metrics, optimizer=opt)

    # Experiment Directory
    expdir = make_exp_dir(exp_dir=exp_dir, prefix=exp_name)
    logging.info(f"expdir={expdir}")

    # Checkpoint Handling for Spot Instance
    checkpoint_path = os.path.join(expdir, "model_checkpoint.keras")
    latest_checkpoint = tf.train.latest_checkpoint(expdir)

    if latest_checkpoint:
        logging.info(f"Resuming training from {latest_checkpoint}")
        nn.load_weights(latest_checkpoint)
    else:
        logging.info("No checkpoint found, starting fresh training.")

    # Callbacks
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_AUCPR",
        save_best_only=True,
        save_weights_only=True,
    )
    early_stop_cb = keras.callbacks.EarlyStopping(
        monitor="val_AUCPR", patience=5, restore_best_weights=True
    )

    # TensorBoard Logging
    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir=os.path.join(expdir, "logs"), write_graph=False
    )

    callbacks = [checkpoint_cb, early_stop_cb, tensorboard_cb]

    # Spot Instance Preemption Handling
    for epoch in range(epochs):
        if check_spot_instance():
            logging.warning(
                "Spot instance interruption detected! Saving checkpoint and exiting..."
            )
            nn.save_weights(checkpoint_path)
            break

        history = nn.fit(
            ds_train, validation_data=ds_val, callbacks=callbacks, epochs=1, verbose=1
        )

    best_aucpr = max(history.history.get("val_AUCPR", [0]))
    return {"AUCPR": best_aucpr}


if __name__ == "__main__":
    config = DEFAULT_CONFIG
    if len(sys.argv) > 1:
        config.update(json.load(open(sys.argv[1], "r")))

    try:
        main(config)
    except KeyboardInterrupt:
        logging.warning("Training interrupted! Saving checkpoint...")
