import keras

model=keras.models.load_model("/home/ubuntu/tornet-Higgins/tornado_detector_baseline.keras")
import tensorflow as tf




print(model.summary())
