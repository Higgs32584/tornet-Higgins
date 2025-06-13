import optuna
import json
import os
import tensorflow as tf
from optuna.pruners import MedianPruner
from tornet.metrics.keras import metrics as tfm
from optuna.integration import TFKerasPruningCallback
from tensorflow.keras.optimizers import Adam
import keras
from tensorflow.keras.losses import BinaryCrossentropy
from fold1fold5deeper import (
    build_model,
    WarmUpCosine,
    DEFAULT_CONFIG,
    get_dataloader,
    SEED,
)
import json
import logging
import os
import random
import shutil
import sys
from typing import List, Tuple
import tensorflow_datasets as tfds
from tensorflow import keras
import tensorflow as tf
import numpy as np
import tornet.data.tfds.tornet.tornet_dataset_builder
from tornet.data.constants import ALL_VARIABLES, CHANNEL_MIN_MAX
from tornet.data.loader import get_dataloader
from tornet.data.preprocess import get_shape
from tornet.metrics.keras import metrics as tfm
from tornet.utils.general import make_exp_dir

optuna.logging.set_verbosity(optuna.logging.DEBUG)
logging.basicConfig(level=logging.INFO)


def save_best_params_callback(study, trial):
    if study.best_trial == trial:
        best_params = trial.params
        os.makedirs("tuner_results", exist_ok=True)
        output_path = os.path.join("tuner_results", "best_hyperparameters.json")
        with open(output_path, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"\n[✓] New best trial #{trial.number}")
        print(f"Best AUCPR so far: {trial.value}")
        print(f"Best parameters:\n{json.dumps(best_params, indent=2)}")


def objective(trial):
    config = DEFAULT_CONFIG.copy()
    config["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    config["dropout_rate"] = trial.suggest_float("dropout_rate", 0.05, 0.3, step=0.01)
    config["label_smoothing"] = trial.suggest_float(
        "label_smoothing", 0.00, 0.10, step=0.01
    )
    config["l2_reg"] = trial.suggest_float("l2_reg", 1e-8, 1e-3, log=True)
    print(f"\n🔍 Trial {trial.number} | Params: {json.dumps(trial.params, indent=2)}")
    # Load training data
    ds_train = get_dataloader(
        "tensorflow-tfds",
        DEFAULT_CONFIG["exp_dir"],
        years=DEFAULT_CONFIG["train_years"],
        data_type="train",
        batch_size=64,
        weights=DEFAULT_CONFIG["weights"],
        random_state=SEED,
        **DEFAULT_CONFIG["dataloader_kwargs"],
    )
    data = next(iter(ds_train))
    x, _, _ = data
    in_shape = (120, 240, get_shape(x)[-1])
    c_shapes = (120, 240, x["coordinates"].shape[-1])

    model = build_model(
        shape=in_shape,
        c_shape=c_shapes,
        input_variables=config["input_variables"],
        dropout_rate=config["dropout_rate"],
        l2_reg=config["l2_reg"],
        start_filters=config["start_filters"],
        mid_filters=config["start_filters"],
        dense_filters=config["dense_filters"],
    )

    steps_per_epoch = len(ds_train)
    lr_schedule = WarmUpCosine(
        base_lr=config["learning_rate"],
        warmup_steps=3 * steps_per_epoch,
        restart_steps=10 * steps_per_epoch,
    )
    from_logits = False  # Use logits for binary classification
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

    loss_fn = BinaryCrossentropy(
        from_logits=from_logits, label_smoothing=config["label_smoothing"]
    )
    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss=loss_fn,
        metrics=metrics,
        jit_compile=True,
    )

    # Load validation data
    val_ds = get_dataloader(
        "tensorflow-tfds",
        DEFAULT_CONFIG["exp_dir"],
        years=DEFAULT_CONFIG["val_years"],
        data_type="train",
        batch_size=64,
        weights={"wN": 1.0, "w0": 1.0, "w1": 1.0, "w2": 1.0, "wW": 1},
        random_state=SEED,
        **DEFAULT_CONFIG["dataloader_kwargs"],
    )
    callbacks = [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.EarlyStopping(
            monitor="val_aucpr",
            patience=config["early_stopping_patience"],
            mode="max",
            restore_best_weights=True,
        ),
        TFKerasPruningCallback(trial, "val_aucpr"),
    ]

    history = model.fit(
        ds_train,
        validation_data=val_ds,
        epochs=30,
        callbacks=callbacks,
    )
    val_auc = max(history.history["val_aucpr"])
    from tensorflow.keras import backend as K
    import gc

    K.clear_session()
    del model
    gc.collect()
    return val_auc


# Run Optuna
pruner = MedianPruner(n_warmup_steps=5)
study = optuna.create_study(
    direction="maximize", sampler=optuna.samplers.TPESampler(n_startup_trials=0)
)
best_params = {
    "learning_rate": 0.003,
    "l2_reg": 3e-7,
    "label_smoothing": 0.01,
    "dropout_rate": 0.12,
}
best_params_2 = {
    "learning_rate": 0.0010907301707496417,
    "dropout_rate": 0.22999999999999998,
    "label_smoothing": 0.1,
    "l2_reg": 6.243448217828412e-08,
}
study.enqueue_trial(best_params)
study.enqueue_trial(best_params_2)
study.optimize(objective, n_trials=50, callbacks=[save_best_params_callback])

# Save best hyperparameters
best_params = study.best_params
output_path = os.path.join("tuner_results", "best_hyperparameters.json")
os.makedirs("tuner_results", exist_ok=True)
with open(output_path, "w") as f:
    json.dump(best_params, f, indent=2)

print(f"\n[✓] Best hyperparameters saved to {output_path}")
