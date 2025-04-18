# build_ensemble.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Average
import sys

def log(msg):
    print(f"[DEBUG] {msg}", flush=True)  # flush=True forces immediate output

# ✅ Re-register custom layers
@keras.utils.register_keras_serializable()
class FillNaNs(tf.keras.layers.Layer):
    def __init__(self, fill_val, **kwargs):
        super().__init__(**kwargs)
        self.fill_val = tf.convert_to_tensor(fill_val, dtype=tf.float32)
    def call(self, x):
        return tf.where(tf.math.is_nan(x), self.fill_val, x)
    def get_config(self):
        return {**super().get_config(), "fill_val": float(self.fill_val.numpy())}

@keras.utils.register_keras_serializable()
class ExpandDimsTwice(tf.keras.layers.Layer):
    def call(self, x): return tf.expand_dims(tf.expand_dims(x, axis=1), axis=1)

@keras.utils.register_keras_serializable()
class StackAvgMax(tf.keras.layers.Layer):
    def call(self, x): return tf.stack(x, axis=1)

@keras.utils.register_keras_serializable()
class FastNormalize(tf.keras.layers.Layer):
    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        self._mean_list = [float(m) for m in mean.numpy()] if hasattr(mean, 'numpy') else list(mean)
        self._std_list = [float(s) for s in std.numpy()] if hasattr(std, 'numpy') else list(std)
        self.mean = tf.convert_to_tensor(self._mean_list, dtype=tf.float32)
        self.std = tf.convert_to_tensor(self._std_list, dtype=tf.float32)
    def call(self, x): return (x - self.mean) / (self.std + 1e-6)
    def get_config(self):
        return {**super().get_config(), "mean": self._mean_list, "std": self._std_list}

custom_objects = {
    "FillNaNs": FillNaNs,
    "ExpandDimsTwice": ExpandDimsTwice,
    "StackAvgMax": StackAvgMax,
    "FastNormalize": FastNormalize
}

# multiprocess_loader.py
import tensorflow as tf
from tensorflow import keras
import multiprocessing as mp
import os
import json


def load_model(path_json, path_weights, name, return_dict):
    with open(path_json) as f:
        model = keras.models.model_from_json(f.read(), custom_objects=custom_objects)
    model.load_weights(path_weights)
    return_dict[name] = model.get_weights()

if __name__ == "__main__":
    manager = mp.Manager()
    return_dict = manager.dict()

    p1 = mp.Process(target=load_model, args=("model1.json", "model1.weights.h5", "model1", return_dict))
    p2 = mp.Process(target=load_model, args=("model2.json", "model2.weights.h5", "model2", return_dict))

    print("[INFO] Starting subprocesses...")
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    print("[INFO] Both models loaded in separate processes.")

    # ✅ Reload the architecture in this process (e.g. model1.json)
    with open("model1.json") as f:
        model1 = keras.models.model_from_json(f.read(), custom_objects=custom_objects)
    model1.set_weights(return_dict["model1"])

    with open("model2.json") as f:
        model2 = keras.models.model_from_json(f.read(), custom_objects=custom_objects)
    model2.set_weights(return_dict["model2"])

    print("[INFO] Creating ensemble...")
    ensemble_output = keras.layers.Average(name="soft_vote")([model1.output, model2.output])
    ensemble_model = keras.models.Model(inputs=model1.input, outputs=ensemble_output)

    # Save ensemble (safe)
    with open("ensemble_model.json", "w") as f:
        f.write(ensemble_model.to_json())
    ensemble_model.save_weights("ensemble_model.weights.h5")
    print("✅ Ensemble saved successfully.")
