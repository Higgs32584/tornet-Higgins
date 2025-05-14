import json
import os
from datetime import datetime

import optuna
import tensorflow as tf

from train_5926 import main  # Assumes your main training function is here

DEFAULT_CONFIG = {
    "epochs": 100,
    "input_variables": ["DBZ", "VEL", "KDP", "ZDR", "RHOHV", "WIDTH"],
    "train_years": [2014, 2015, 2016],
    "val_years": [2017],
    "batch_size": 128,
    "model": "wide_resnet",
    "start_filters": 48,
    "learning_rate": 1e-4,
    "decay_steps": 1386,
    "decay_rate": 0.958,
    "l2_reg": 1e-5,
    "wN": 1.0,
    "w0": 1.0,
    "w1": 1.0,
    "w2": 2.0,
    "wW": 0.5,
    "nconvs": 2,
    "dropout_rate": 0.1,
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

# --- Logging directory setup ---
BASE_LOG_DIR = "/home/ubuntu/tornet-Higgins/optuna_studies"
TIMESTAMP = datetime.now().strftime("%y_%m_%d_%H_%M")
STUDY_DIR = os.path.join(BASE_LOG_DIR, f"study_{TIMESTAMP}")
os.makedirs(STUDY_DIR, exist_ok=True)
TRIAL_LOG_PATH = os.path.join(STUDY_DIR, "trials.json")


def log_trial_result(trial_number, params, value):
    result = {"trial_number": trial_number, "params": params, "value": value}
    if os.path.exists(TRIAL_LOG_PATH):
        with open(TRIAL_LOG_PATH, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(result)
    with open(TRIAL_LOG_PATH, "w") as f:
        json.dump(data, f, indent=2)


def objective(trial):
    config = DEFAULT_CONFIG.copy()
    config.update(
        {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
            "l2_reg": trial.suggest_float("l2_reg", 1e-8, 1e-3, log=True),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.05, 0.4),
            "start_filters": trial.suggest_categorical(
                "start_filters", [16, 32, 48, 64]
            ),
            "nconvs": trial.suggest_int("nconvs", 1, 3),
            "decay_steps": trial.suggest_int("decay_steps", 1000, 4000),
            "decay_rate": trial.suggest_float("decay_rate", 0.9, 0.99),
            "wN": trial.suggest_float("wN", 0.01, 1.0),
            "w0": trial.suggest_float("w0", 0.1, 2.0),
            "w1": trial.suggest_float("w1", 0.5, 5.0),
            "w2": trial.suggest_float("w2", 1.0, 15.0),
            "wW": trial.suggest_float("wW", 0.1, 1.0),
        }
    )

    try:
        results = main(config)
        aucpr = results.get("AUCPR", 0.0)
    except tf.errors.InvalidArgumentError as e:
        print(f"Trial {trial.number} failed: {e}")
        aucpr = 0.0

    # Log result to JSON
    log_trial_result(trial.number, trial.params, aucpr)

    return aucpr


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        study_name=f"tornado_aucpr_opt_{TIMESTAMP}",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(objective, n_trials=50)

    print("\n✅ Best Trial:")
    print(study.best_trial)

    print("\n✅ Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")
