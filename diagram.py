import tensorflow as tf
from tensorflow.keras.utils import plot_model
import sys
import os
from tensorflow.keras.models import Model

def plot_keras_model(model_path, output_file="model.png"):
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        plot_model(
            model,
            to_file=output_file,
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True,
            dpi=100,
            rankdir="TB"                # Top-to-bottom layout

        )
        print(f"Model architecture saved to {output_file}")
    except Exception as e:
        print(f"Failed to plot model: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_keras_model.py path_to_model.keras [output_image.png]")
    else:
        model_path = sys.argv[1]
        output_image = sys.argv[2] if len(sys.argv) > 2 else "model.png"
        plot_keras_model(model_path, output_image)
