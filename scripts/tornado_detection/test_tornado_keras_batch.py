import argparse
import logging
import os

import tensorflow as tf
from custom_func import FalseAlarmRate, ThreatScore
from tensorflow import keras

from tornet.data.loader import get_dataloader
from tornet.metrics.keras import metrics as tfm

logging.basicConfig(level=logging.INFO)

TFDS_DATA_DIR = "/home/ubuntu/tfds"
EXP_DIR = os.environ.get("EXP_DIR", ".")
TORNET_ROOT = TFDS_DATA_DIR
# TFDS_DATA_DIR=os.environ['TFDS_DATA_DIR']


# Environment Variables with Defaults (can be overridden externally)
DATA_ROOT = os.environ.get("DATA_ROOT", "/home/ubuntu/tfds")
TORNET_ROOT = os.environ.get("TORNET_ROOT", DATA_ROOT)
TFDS_DATA_DIR = os.environ.get("TFDS_DATA_DIR", DATA_ROOT)
EXP_DIR = os.environ.get("EXP_DIR", ".")

# Set variables if not already defined (avoid overwriting existing ones)
os.environ.setdefault("TORNET_ROOT", TORNET_ROOT)
os.environ.setdefault("TFDS_DATA_DIR", TFDS_DATA_DIR)
os.environ.setdefault("EXP_DIR", EXP_DIR)
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")
tf.config.optimizer.set_jit(True)

os.environ["TORNET_ROOT"] = DATA_ROOT
os.environ["TFDS_DATA_DIR"] = TFDS_DATA_DIR


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


@keras.utils.register_keras_serializable()
class ExpandDimsTwice(keras.layers.Layer):
    def call(self, inputs):
        return tf.expand_dims(tf.expand_dims(inputs, axis=1), axis=1)


@keras.utils.register_keras_serializable()
class StackAvgMax(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.stack(inputs, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_paths",
        nargs="+",
        help="List of pretrained models to test (.keras files)",
        required=True,
    )
    parser.add_argument(
        "--dataloader",
        help="Which data loader to use for loading test data",
        default="tensorflow-tfds",
        choices=["keras", "tensorflow", "tensorflow-tfds", "torch", "torch-tfds"],
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold to apply to ensemble predictions for binary classification",
    )
    args = parser.parse_args()

    dataloader = args.dataloader

    logging.info(f"Using {keras.config.backend()} backend")
    logging.info(f"Using {dataloader} dataloader")

    if ("tfds" in dataloader) and ("TFDS_DATA_DIR" in os.environ):
        logging.info("Using TFDS dataset location at " + os.environ["TFDS_DATA_DIR"])

    def uncertainty_fn(x):
        return tf.abs(x - 0.5)

    # Load all ensemble models
    model_paths = args.model_paths
    models = []
    for path in model_paths:
        model = tf.keras.models.load_model(
            path,
            safe_mode=False,
            compile=False,
            custom_objects={"<lambda>": uncertainty_fn},
        )
        models.append(model)
        logging.info(f"Loaded model from: {path}")

    # Set up data loader
    test_years = range(2013, 2023)
    ds_test = get_dataloader(
        dataloader,
        DATA_ROOT,
        test_years,
        "test",
        128,
        weights={"wN": 1.0, "w0": 1.0, "w1": 1.0, "w2": 1.0, "wW": 1.0},
        select_keys=list(models[0].input.keys()),
    )

    # Define metrics
    from_logits = False
    metrics = [
        keras.metrics.AUC(
            from_logits=from_logits, curve="PR", name="AUCPR", num_thresholds=2000
        ),
        tfm.BinaryAccuracy(from_logits, name="BinaryAccuracy"),
        tfm.TruePositives(from_logits, name="TruePositives"),
        tfm.FalsePositives(from_logits, name="FalsePositives"),
        tfm.TrueNegatives(from_logits, name="TrueNegatives"),
        tfm.FalseNegatives(from_logits, name="FalseNegatives"),
        tfm.Precision(from_logits, name="Precision"),
        tfm.Recall(from_logits, name="Recall"),
        FalseAlarmRate(name="FalseAlarmRate"),
        tfm.F1Score(from_logits=from_logits, name="F1"),
        ThreatScore(name="ThreatScore"),
    ]

    # Create a metric tracker
    for m in metrics:
        m.reset_state()
    import tqdm

    for batch in tqdm.tqdm(ds_test):

        inputs, labels, weights = batch
        # Collect predictions from all models
        preds = [model.predict_on_batch(inputs) for model in models]
        preds_stack = tf.stack(preds, axis=0)
        ensemble_preds = tf.reduce_mean(preds_stack, axis=0)
        binary_preds = tf.cast(ensemble_preds >= args.threshold, tf.float32)

        # Mask out samples where the weight is zero
        weights_total = (
            tf.reduce_sum(weights, axis=-1) if len(weights.shape) > 1 else weights
        )
        nonzero_mask = tf.not_equal(weights_total, 0)

        labels_masked = tf.boolean_mask(labels, nonzero_mask)
        ensemble_preds_masked = tf.boolean_mask(ensemble_preds, nonzero_mask)
        binary_preds_masked = tf.boolean_mask(binary_preds, nonzero_mask)

        for metric in metrics:
            if metric.name.lower() in ["aucpr", "auc"]:
                metric.update_state(labels_masked, ensemble_preds_masked)
            else:
                metric.update_state(labels_masked, binary_preds_masked)

    # Print results
    scores = {m.name: m.result().numpy() for m in metrics}
    num_params = sum([model.count_params() for model in models])
    logging.info(f"Threshold: {args.threshold:.4f}")
    logging.info(f"Model parameters: {num_params:,}")
    for name, score in scores.items():
        logging.info(f"{name}: {score:.4f}")


if __name__ == "__main__":
    main()
