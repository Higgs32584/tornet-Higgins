import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Layer, ReLU


class CoordConv2D(Layer):
    """
    CoordConv2D layer for working with polar data in Keras.

    This layer takes a tuple of inputs (image tensor, image coordinates)
    where:
        - image tensor is [batch, height, width, in_image_channels]
        - image coordinates is [batch, height, width, in_coord_channels]

    It returns a tuple containing the CoordConv convolution output and
    a (possibly downsampled) copy of the coordinate tensor.
    """

    def __init__(
        self,
        in_image_channels,
        in_coord_channels,
        out_channels,
        kernel_size,
        padding="same",
        strides=1,
        activation="relu",
        **kwargs,
    ):
        super(CoordConv2D, self).__init__(**kwargs)
        self.n_coord_channels = in_coord_channels
        self.conv = Conv2D(out_channels, kernel_size, strides=strides, padding=padding)
        self.strides = strides
        self.padding = padding
        self.kernel_size = kernel_size

        if activation is None:
            self.conv_activation = None
        elif activation == "relu":
            self.conv_activation = ReLU()
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")

    def call(self, inputs):
        """
        inputs is a tuple containing:
          (image tensor, image coordinates)

        image tensor: [batch, height, width, in_image_channels]
        image coordinates: [batch, height, width, in_coord_channels]
        """
        x, coords = inputs
        x = tf.concat([x, coords], axis=-1)
        x = self.conv(x)

        # Apply activation if defined
        if self.conv_activation:
            x = self.conv_activation(x)

        # Handle coordinate downsampling
        if self.padding == "same" and self.strides > 1:
            coords = coords[:, :: self.strides, :: self.strides, :]
        elif self.padding == "valid":
            i0 = self.kernel_size[0] // 2
            if i0 > 0:
                coords = coords[:, i0 : -i0 : self.strides, i0 : -i0 : self.strides, :]
            else:
                coords = coords[:, :: self.strides, :: self.strides, :]

        return x, coords
