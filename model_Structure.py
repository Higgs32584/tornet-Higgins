import argparse

import tensorflow as tf
from tensorflow import keras


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
class FillNaNs(keras.layers.Layer):
    def __init__(self, fill_val, **kwargs):
        super().__init__(**kwargs)
        self.fill_val = tf.convert_to_tensor(fill_val, dtype=tf.float32)

    @tf.function(jit_compile=True)
    def call(self, x):
        return tf.where(tf.math.is_nan(x), self.fill_val, x)

    def get_config(self):
        return {**super().get_config(), "fill_val": self.fill_val.numpy().item()}


def load_and_print_model(model_path):
    """Loads a TensorFlow model from a given path and prints its structure."""
    try:
        # Load the model
        model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        print("Model loaded successfully!")

        # Print the model summary
        model.summary()

        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and print a TensorFlow model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the TensorFlow model (.keras or SavedModel directory)",
    )

    args = parser.parse_args()
    load_and_print_model(args.model_path)
