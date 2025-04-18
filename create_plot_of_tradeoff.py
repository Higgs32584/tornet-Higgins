import os
import argparse
import logging
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tornet.data.loader import get_dataloader
from sklearn.metrics import precision_recall_curve
import numpy as np
from custom_func import FalseAlarmRate, ThreatScore

# Set up environment
TFDS_DATA_DIR = "/home/ubuntu/tfds"
DATA_ROOT = TFDS_DATA_DIR
os.environ['TORNET_ROOT'] = DATA_ROOT
os.environ['TFDS_DATA_DIR'] = TFDS_DATA_DIR

logging.basicConfig(level=logging.INFO)

@keras.utils.register_keras_serializable()
class SpatialAttention(keras.layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.conv = keras.layers.Conv2D(1, kernel_size=kernel_size, padding='same', activation='sigmoid')
    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        attention = self.conv(tf.concat([avg_pool, max_pool], axis=-1))
        return keras.layers.Multiply()([x, attention])

@keras.utils.register_keras_serializable()
class ChannelAttention(keras.layers.Layer):
    def __init__(self, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(1, activation='sigmoid')
    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        attention = self.dense1(avg_pool)
        return keras.layers.Multiply()([x, attention])

@keras.utils.register_keras_serializable()
class FillNaNs(keras.layers.Layer):
    def __init__(self, fill_val, **kwargs):
        super().__init__(**kwargs)
        self.fill_val = tf.convert_to_tensor(fill_val, dtype=tf.float32)
    def call(self, x):
        return tf.where(tf.math.is_nan(x), self.fill_val, x)

@keras.utils.register_keras_serializable()
class FastNormalize(keras.layers.Layer):
    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        self.mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        self.std = tf.convert_to_tensor(std, dtype=tf.float32)
    def call(self, x):
        return (x - self.mean) / (self.std + 1e-6)

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
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing .keras models")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary predictions")
    args = parser.parse_args()

    model_dir = args.model_dir

    # Load dataset once
    logging.info("Loading test dataset...")
    dataset = get_dataloader(
        dataloader="tensorflow-tfds",
        data_root=DATA_ROOT,
        years=range(2020, 2023),
        data_type="train",
        batch_size=128,
        weights={'wN': 1.0, 'w0': 1.0, 'w1': 1.0, 'w2': 1.0, 'wW': 1.0},
        select_keys=None  # Will dynamically use first model's inputs
    )
    logging.info("Dataset ready.")

    # Collect ground truth once
    y_true_all = []
    inputs_cache = []
    for batch in dataset:
        inputs, labels, _ = batch
        y_true_all.append(labels.numpy())
        inputs_cache.append(inputs)

    y_true = np.concatenate(y_true_all, axis=0)

    plt.figure(figsize=(10, 7))

    # Loop through each model in directory
    for fname in os.listdir(model_dir):
        if not fname.endswith(".keras"):
            continue

        model_path = os.path.join(model_dir, fname)
        logging.info(f"Loading model: {fname}")
        model = keras.models.load_model(
            model_path, compile=False, safe_mode=False,
            custom_objects={"<lambda>": lambda x: tf.abs(x - 0.5)}
        )

        # Predict with the current model
        preds_all = []
        for inputs in inputs_cache:
            preds = model.predict_on_batch(inputs)
            preds_all.append(preds)
        y_scores = np.concatenate(preds_all, axis=0)

        # Precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true.ravel(), y_scores.ravel())
        aucpr = tf.keras.metrics.AUC(curve='PR')
        aucpr.update_state(y_true.ravel(), y_scores.ravel())
        auc_val = aucpr.result().numpy()

        # Plot
        plt.plot(recall, precision, label=f"{fname} (PR AUC={auc_val:.3f})")

        # Clean up
        del model
        tf.keras.backend.clear_session()

    # Final plot formatting
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR AUC Curves for Models in Directory")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("AUC_PR_CURVES.png")

if __name__ == "__main__":
    main()

