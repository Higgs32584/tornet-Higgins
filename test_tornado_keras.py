from tensorflow import keras
import tensorflow as tf
from tornet.data.loader import get_dataloader
from tornet.metrics.keras import metrics as tfm
import os
import argparse
import logging
logging.basicConfig(level=logging.INFO)
from custom_func import FalseAlarmRate,ThreatScore

TFDS_DATA_DIR="/home/ubuntu/tfds"
EXP_DIR=os.environ.get('EXP_DIR','.')
TORNET_ROOT=TFDS_DATA_DIR
#TFDS_DATA_DIR=os.environ['TFDS_DATA_DIR']
import tensorflow_datasets as tfds
from tornet.models.keras.layers import CoordConv2D
import tornet.data.tfds.tornet.tornet_dataset_builder # registers 'tornet'
EXP_DIR = "."
DATA_ROOT = '/home/ubuntu/tfds'
TORNET_ROOT=DATA_ROOT
TFDS_DATA_DIR = '/home/ubuntu/tfds'
DATA_ROOT = "/home/ubuntu/tfds"
TFDS_DATA_DIR = "/home/ubuntu/tfds"
os.environ['TORNET_ROOT']= DATA_ROOT
os.environ['TFDS_DATA_DIR']=TFDS_DATA_DIR
@keras.utils.register_keras_serializable()
class SpatialAttention(keras.layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        # Define the Conv2D layer outside of call to avoid re-creating it on every call
        self.conv = keras.layers.Conv2D(1, kernel_size=self.kernel_size, strides=1, padding='same', activation='sigmoid')

    def call(self, x):
        # Perform average and max pooling along the channel axis
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)  # Use tf.reduce_mean
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)  # Use tf.reduce_max

        # Concatenate the pooled results along the channel axis
        concatenated = tf.concat([avg_pool, max_pool], axis=-1)  # Use tf.concat
        
        # Apply the convolution to generate the spatial attention map
        attention = self.conv(concatenated)

        # Multiply the input with the attention map
        return keras.layers.Multiply()([x, attention])

@keras.utils.register_keras_serializable()
class ChannelAttention(keras.layers.Layer):
    def __init__(self, ratio=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio
        # Define the Dense layers to apply channel attention
        self.dense1 = keras.layers.Dense(1, activation='sigmoid')

    def call(self, x):
        # Perform global average pooling to obtain channel-wise statistics
        avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        
        # Apply the dense layer to create attention weights for each channel
        attention = self.dense1(avg_pool)

        # Multiply the input by the attention weights
        return keras.layers.Multiply()([x, attention])
    
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
        self._mean_list = mean.numpy().tolist() if hasattr(mean, 'numpy') else list(mean)
        self._std_list = std.numpy().tolist() if hasattr(std, 'numpy') else list(std)

    def call(self, x):
        return tf.math.subtract(x, self.mean) / (self.std + 1e-6)

    def get_config(self):
        config = super().get_config()
        config.update({
            "mean": self._mean_list,
            "std": self._std_list,
        })
        return config
@keras.utils.register_keras_serializable()
class ExpandDimsTwice(keras.layers.Layer):
    def call(self, inputs):
        return tf.expand_dims(tf.expand_dims(inputs, axis=1), axis=1)
@keras.utils.register_keras_serializable()
class StackAvgMax(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.stack(inputs, axis=1)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer, Conv2D, ReLU
from keras.saving import register_keras_serializable
from keras.layers import Layer
import tensorflow as tf
        
from keras.saving import register_keras_serializable
from keras.layers import Layer, Conv2D
import tensorflow as tf

# @register_keras_serializable()
# class CoordConv2D(Layer):
#     def __init__(self, out_channels, kernel_size, strides=1, padding='same', activation='relu', **kwargs):
#         super().__init__(**kwargs)
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.strides = strides
#         self.padding = padding
#         self.activation = tf.keras.activations.get(activation)
#         self.conv = None  # Will be created in build()

#     def build(self, input_shape):
#         _, height, width, channels = input_shape
#         self.input_channels = channels + 2  # Add 2 for x and y coords

#         self.conv = Conv2D(
#             filters=self.out_channels,
#             kernel_size=self.kernel_size,
#             strides=self.strides,
#             padding=self.padding,
#             activation=self.activation
#         )

#     def call(self, inputs):
#         batch_size, height, width, _ = tf.unstack(tf.shape(inputs))

#         # Create coordinate tensors
#         x_coords = tf.linspace(-1.0, 1.0, width)
#         y_coords = tf.linspace(-1.0, 1.0, height)

#         x_coords = tf.tile(x_coords[tf.newaxis, tf.newaxis, :, tf.newaxis], [batch_size, height, 1, 1])
#         y_coords = tf.tile(y_coords[tf.newaxis, :, tf.newaxis, tf.newaxis], [batch_size, 1, width, 1])

#         coords = tf.concat([x_coords, y_coords], axis=-1)

#         # Concatenate to inputs
#         inputs_with_coords = tf.concat([inputs, coords], axis=-1)

#         return self.conv(inputs_with_coords)

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "out_channels": self.out_channels,
#             "kernel_size": self.kernel_size,
#             "strides": self.strides,
#             "padding": self.padding,
#             "activation": tf.keras.activations.serialize(self.activation),
#         })
#         return config



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        help="Pretrained model to test (.keras)",
                        default=None)
    parser.add_argument(
        "--dataloader",
        help='Which data loader to use for loading test data',
        default="tensorflow-tfds",
        choices=["keras", "tensorflow", "tensorflow-tfds", "torch", "torch-tfds"],
    )
    args = parser.parse_args()

        
    dataloader = args.dataloader

    logging.info(f"Using {keras.config.backend()} backend")
    logging.info(f"Using {dataloader} dataloader")

    if ("tfds" in dataloader) and ('TFDS_DATA_DIR' in os.environ):
        logging.info('Using TFDS dataset location at '+os.environ['TFDS_DATA_DIR'])
    
    # load model
    # Suppose you used this in a Lambda:
    def top_k_pooling(x, k=5):
        # x: (batch, H, W, 1) → reshape to (batch, H×W)
        x_flat = tf.reshape(x, [tf.shape(x)[0], -1])         # (batch, H×W)
        topk = tf.nn.top_k(x_flat, k=k, sorted=False).values # (batch, k)
        return tf.reduce_mean(topk, axis=1, keepdims=True)   # (batch, 1)
    def uncertainty_fn(x):
        return tf.abs(x - 0.5)

    model = tf.keras.models.load_model(args.model_path,safe_mode=False,compile=False,custom_objects={"<lambda>": uncertainty_fn})

    ## Set up data loader
    test_years = range(2013,2023)
    ds_test = get_dataloader(dataloader, 
                             DATA_ROOT, 
                             test_years, 
                             "test", 
                             128,
                             weights={'wN':1.0,'w0':1.0,'w1':1.0,'w2':1.0,'wW':1.0},
                             select_keys=list(model.input.keys()))


    # Compute various metrics
    from_logits=False
    metrics = [keras.metrics.AUC(from_logits=from_logits,curve='PR',name='AUCPR',num_thresholds=1000), 
                tfm.BinaryAccuracy(from_logits,name='BinaryAccuracy'), 
                tfm.TruePositives(from_logits,name='TruePositives'),
                tfm.FalsePositives(from_logits,name='FalsePositives'), 
                tfm.TrueNegatives(from_logits,name='TrueNegatives'),
                tfm.FalseNegatives(from_logits,name='FalseNegatives'), 
                tfm.Precision(from_logits,name='Precision'), 
                tfm.Recall(from_logits,name='Recall'),
                FalseAlarmRate(name='FalseAlarmRate'),
                tfm.F1Score(from_logits=from_logits,name='F1'),
                ThreatScore(name='ThreatScore')]
    model.compile(metrics=metrics,loss=keras.losses.BinaryCrossentropy())

    scores = model.evaluate(ds_test) 
    scores = {m.name:scores[k+1] for k,m in enumerate(metrics)}

    logging.info(scores)
 
if __name__=='__main__':
    main()
