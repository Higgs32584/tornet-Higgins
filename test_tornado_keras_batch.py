import argparse
import logging
import os

import tensorflow as tf
from tensorflow import keras

from tornet.data.loader import get_dataloader
from tornet.metrics.keras import metrics as tfm

logging.basicConfig(level=logging.INFO)
from custom_func import FalseAlarmRate, ThreatScore

TFDS_DATA_DIR = "/home/ubuntu/tfds"
EXP_DIR = os.environ.get("EXP_DIR", ".")
TORNET_ROOT = TFDS_DATA_DIR
# TFDS_DATA_DIR=os.environ['TFDS_DATA_DIR']
import tensorflow_datasets as tfds

import tornet.data.tfds.tornet.tornet_dataset_builder  # registers 'tornet'

EXP_DIR = "."
DATA_ROOT = "/home/ubuntu/tfds"
TORNET_ROOT = DATA_ROOT
DATA_ROOT = "/home/ubuntu/tfds"
TFDS_DATA_DIR = "/home/ubuntu/tfds"
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


@keras.utils.register_keras_serializable()
class SelectAttentionBranch(tf.keras.layers.Layer):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, x):
        # x has shape (batch, num_branches)
        return tf.expand_dims(x[:, self.index], axis=-1)  # shape: (batch, 1)


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

    # Load all ensemble models
    custom_objects = {
        "FillNaNs": FillNaNs,
        "FastNormalize": FastNormalize,
        "ExpandDimsTwice": ExpandDimsTwice,
        "StackAvgMax": StackAvgMax,
        "FalseAlarmRate": FalseAlarmRate,
        "ThreatScore": ThreatScore,
        # Optional: Add WarmUpCosine if used in learning rate schedule
    }
    model_paths = args.model_paths
    models = []
    for path in model_paths:
        model = tf.keras.models.load_model(
            path,
            safe_mode=False,
            compile=False,
            custom_objects=custom_objects,
        )
        models.append(model)
        logging.info(f"Loaded model from: {path}")

    # Set up data loader
    test_years = range(2013, 2023)
    ds_test = get_dataloader(
        dataloader,
        DATA_ROOT,
        test_years,
        random_state=42,
        data_type="test",
        batch_size=128,
        weights={"wN": 1.0, "w0": 1.0, "w1": 1.0, "w2": 1.0, "wW": 1.0},
        select_keys=list(models[0].input.keys()),
    )

    # Define metrics
    from_logits = False
    threshold = args.threshold
    metrics = [
        keras.metrics.AUC(from_logits=from_logits, name="AUC", num_thresholds=2000),
        keras.metrics.AUC(
            from_logits=from_logits, curve="PR", name="AUCPR", num_thresholds=2000
        ),
        tfm.BinaryAccuracy(
            from_logits=from_logits, threshold=threshold, name="BinaryAccuracy"
        ),
        tfm.TruePositives(
            from_logits=from_logits, thresholds=threshold, name="TruePositives"
        ),
        tfm.FalsePositives(
            from_logits=from_logits, thresholds=threshold, name="FalsePositives"
        ),
        tfm.TrueNegatives(
            from_logits=from_logits, thresholds=threshold, name="TrueNegatives"
        ),
        tfm.FalseNegatives(
            from_logits=from_logits, thresholds=threshold, name="FalseNegatives"
        ),
        tfm.Precision(from_logits=from_logits, thresholds=threshold, name="Precision"),
        tfm.Recall(from_logits=from_logits, thresholds=threshold, name="Recall"),
        tfm.F1Score(threshold=threshold, name="F1"),
        FalseAlarmRate(name="FalseAlarmRate", threshold=threshold),
        ThreatScore(name="ThreatScore", threshold=threshold),
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
