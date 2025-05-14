import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Average
from tensorflow.keras.models import Model


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
class ExpandDimsTwice(keras.layers.Layer):
    def call(self, inputs):
        return tf.expand_dims(tf.expand_dims(inputs, axis=1), axis=1)


@keras.utils.register_keras_serializable()
class StackAvgMax(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.stack(inputs, axis=1)


@keras.utils.register_keras_serializable()
class FastNormalize(tf.keras.layers.Layer):
    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        self._mean_list = (
            [float(v) for v in mean.numpy()] if hasattr(mean, "numpy") else list(mean)
        )
        self._std_list = (
            [float(v) for v in std.numpy()] if hasattr(std, "numpy") else list(std)
        )
        self.mean = tf.constant(self._mean_list, dtype=tf.float32)
        self.std = tf.constant(self._std_list, dtype=tf.float32)

    def call(self, x):
        return (x - self.mean) / (self.std + 1e-6)

    def get_config(self):
        return {**super().get_config(), "mean": self._mean_list, "std": self._std_list}


print("load model 1")
model1 = keras.models.load_model(
    "/home/ubuntu/tornet-Higgins/best_models_so_far/tornadoDetector_v5.keras",
    compile=False,
)
print("loaded model 1")
print("load model 2")
model2 = keras.models.load_model(
    "/home/ubuntu/tornet-Higgins/best_models_so_far/tornadoDetector_v6.keras",
    compile=False,
)
print("loaded model 2")

# Assuming both models share the same input
ensemble_output = Average()([model1.output, model2.output])
ensemble_model = Model(inputs=model1.input, outputs=ensemble_output)
ensemble_model.save("ensemble_model.keras")
