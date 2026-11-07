"""
TornadoNet: one configurable Keras model for architecture ablations.

With every flag at its default, ``build_tornado_net`` builds the same layer
structure as ``build_model`` in ``scripts/tornado_detection/train_textbook.py``
(the best single-architecture model): normalize -> FillNaNs(-3) -> concat
range-folded mask -> wide residual CoordConv block -> squeeze-excite ->
Conv3x3(128)/BN/ReLU -> 1x1 sigmoid attention -> GAP+GMP -> Dense(64) ->
PReLU -> Dense(1, sigmoid).

Each option below changes one idea, so it can be switched on alone and
compared against that default:

``attention="multi_dilated"``
    Three 3x3 sigmoid attention maps with dilation rates 1, 2 and 4, mixed
    per sample by softmax gates computed from global average-pooled features.
    Hypothesis: tornadic signatures and their parent mesocyclones appear at
    different spatial scales, and a fixed single-scale 1x1 map cannot look
    at neighbourhood context. (The older ``multi_dilated_attention_head`` in
    the scripts used ``kernel_size=1``, for which ``dilation_rate`` has no
    effect; here the kernels are 3x3 so each branch really sees a different
    receptive field.)

``head="mil_lse"``
    Multiple-instance head. A 1x1 conv produces a per-location logit map
    (layer ``"instance_logits"``), pooled to one image logit with
    log-sum-exp pooling and a learnable positive temperature r::

        s = (1/r) * log(mean(exp(r * z)))

    which always lies between mean(z) (r -> 0) and max(z) (r -> inf).
    Hypothesis: a tornado occupies a few pixels of the chip, so "is there a
    tornado anywhere" is a MIL problem; LSE learns how peaky that decision
    should be and gives a localisation heatmap for free (``heatmap_model``).

``shear_features=True``
    Fixed (non-trainable) azimuthal and radial velocity derivatives computed
    from raw VEL, added as extra input channels. Hypothesis: rotation
    (azimuthal shear) is the main physical signature of a tornado, and the
    range normalisation (dividing by the arc length 2*r*dtheta) is hard for a
    small CNN to learn from separate R and 1/R coordinate channels.

``range_film=True``
    FiLM conditioning on range after the residual block: a per-pixel MLP
    (two 1x1 convs) on the R and 1/R coordinate channels outputs a scale and
    shift per feature channel, ``x * (1 + gamma) + beta``. The last conv is
    zero-initialised so the block starts as the identity. Hypothesis: beam
    broadening and height make the same storm look different at different
    ranges; letting range modulate features is a stronger form of
    conditioning than concatenating coordinates.

``output_logits=True``
    The final layer has no activation (use losses with ``from_logits=True``).
    The default keeps the sigmoid so existing training code still works.
"""

import math
from typing import List, Sequence, Tuple

import keras
from keras import layers, ops

from tornet.data.constants import ALL_VARIABLES, CHANNEL_MIN_MAX
from tornet.models.keras.layers import CoordConv2D

# Scale applied to range in tornet.data.preprocess.compute_coordinates.
RANGE_SCALE = 1e-5
# NEXRAD range gate spacing in meters.
GATE_SPACING_M = 250.0
# Minimum radar range used by compute_coordinates (meters).
MIN_RANGE_M = 2125.0

ATTENTION_OPTIONS = ("single", "multi_dilated")
HEAD_OPTIONS = ("pooled", "mil_lse")


# ---------------------------------------------------------------------------
# Custom layers
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="tornet")
class ChannelNormalize(layers.Layer):
    """(x - mean) / (std + 1e-6), the same as FastNormalize in train_textbook."""

    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        self.mean = float(mean)
        self.std = float(std)

    def call(self, x):
        return (x - self.mean) / (self.std + 1e-6)

    def get_config(self):
        config = super().get_config()
        config.update({"mean": self.mean, "std": self.std})
        return config


@keras.saving.register_keras_serializable(package="tornet")
class FillNaNs(layers.Layer):
    """Replace NaNs with ``fill_val``.

    Same behaviour as ``tornet.models.keras.layers.FillNaNs``, but implemented
    through ``call`` so it is a real node in the functional graph.
    """

    def __init__(self, fill_val, **kwargs):
        super().__init__(**kwargs)
        self.fill_val = float(fill_val)

    def call(self, x):
        return ops.where(ops.isnan(x), ops.cast(self.fill_val, x.dtype), x)

    def get_config(self):
        config = super().get_config()
        config.update({"fill_val": self.fill_val})
        return config


@keras.saving.register_keras_serializable(package="tornet")
class VelocityShear(layers.Layer):
    """Fixed velocity-derivative features from raw radial velocity.

    Inputs ``[vel, coords]``:
      vel     (batch, azimuth, range, tilt), raw VEL in m/s, may contain NaN.
      coords  (batch, azimuth, range, C), channel 0 is R = range_m * 1e-5.

    For each tilt it computes, with central differences,

      azimuthal shear  (V[az+1] - V[az-1]) / (2 * range_m * dtheta)   [1/s]
      radial gradient  (V[r+1]  - V[r-1])  / (2 * gate_spacing_m)     [1/s]

    Positive azimuthal shear means velocity increases clockwise (with
    azimuth), which is cyclonic rotation for the usual NEXRAD sign
    convention (positive = outbound).

    Scaling: both derivatives are divided by ``shear_scale`` (default
    0.1 1/s) and passed through tanh, so outputs lie in (-1, 1). A
    gate-to-gate tornado vortex signature (~0.05-0.3 1/s) maps to roughly
    0.5-1, a broad mesocyclone (~0.01 1/s) to ~0.1, and the huge values from
    velocity aliasing or dealiasing errors saturate at +/-1 instead of
    dominating the input.

    NaN handling: wherever the centre pixel or either neighbour used by the
    difference is NaN, or the neighbour is outside the image, the output is
    0 (no shear). Range is clipped below at ``min_range_m`` so the division
    is always safe.

    Output: (batch, azimuth, range, 2 * tilt), azimuthal channels first.
    The layer has no weights.
    """

    def __init__(
        self,
        azimuth_spacing_deg=0.5,
        gate_spacing_m=GATE_SPACING_M,
        range_scale=RANGE_SCALE,
        min_range_m=MIN_RANGE_M,
        shear_scale=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.azimuth_spacing_deg = float(azimuth_spacing_deg)
        self.gate_spacing_m = float(gate_spacing_m)
        self.range_scale = float(range_scale)
        self.min_range_m = float(min_range_m)
        self.shear_scale = float(shear_scale)

    @staticmethod
    def _central_diff(v, valid, axis):
        """Central difference along ``axis`` (1 or 2), 0 where not computable."""
        pad = [[0, 0], [0, 0], [0, 0], [0, 0]]
        pad[axis] = [1, 1]
        vp = ops.pad(v, pad)
        mp = ops.pad(ops.cast(valid, v.dtype), pad)
        if axis == 1:
            dv = vp[:, 2:] - vp[:, :-2]
            ok = mp[:, 2:] * mp[:, :-2]
        else:
            dv = vp[:, :, 2:] - vp[:, :, :-2]
            ok = mp[:, :, 2:] * mp[:, :, :-2]
        ok = ok * ops.cast(valid, v.dtype)
        return dv * ok

    def call(self, inputs):
        vel, coords = inputs
        vel = ops.cast(vel, "float32")
        coords = ops.cast(coords, "float32")

        valid = ops.logical_not(ops.isnan(vel))
        v = ops.where(valid, vel, ops.zeros_like(vel))

        range_m = coords[..., :1] / self.range_scale
        range_m = ops.maximum(range_m, self.min_range_m)  # (B, A, R, 1)
        dtheta = math.radians(self.azimuth_spacing_deg)

        az_shear = self._central_diff(v, valid, axis=1) / (2.0 * range_m * dtheta)
        radial = self._central_diff(v, valid, axis=2) / (2.0 * self.gate_spacing_m)

        out = ops.concatenate([az_shear, radial], axis=-1)
        return ops.cast(ops.tanh(out / self.shear_scale), self.compute_dtype)

    def compute_output_shape(self, input_shape):
        vel_shape, _ = input_shape
        n_tilts = vel_shape[-1]
        return tuple(vel_shape[:-1]) + (None if n_tilts is None else 2 * n_tilts,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "azimuth_spacing_deg": self.azimuth_spacing_deg,
                "gate_spacing_m": self.gate_spacing_m,
                "range_scale": self.range_scale,
                "min_range_m": self.min_range_m,
                "shear_scale": self.shear_scale,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="tornet")
class RangeFiLM(layers.Layer):
    """Per-pixel FiLM conditioning on range.

    Inputs ``[x, coords]``. A two-layer per-pixel MLP (1x1 convs) on the R and
    1/R coordinate channels (first and last channel of ``coords``) outputs
    ``gamma`` and ``beta`` for each of the ``filters`` channels of ``x``, and
    the layer returns ``x * (1 + gamma) + beta``. The output conv is
    zero-initialised, so the layer is the identity at initialisation.
    """

    def __init__(self, filters, hidden_units=16, **kwargs):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.hidden_units = int(hidden_units)
        self.hidden = layers.Conv2D(
            self.hidden_units, 1, activation="relu", name="film_hidden"
        )
        self.film = layers.Conv2D(
            2 * self.filters,
            1,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="film_out",
        )

    def build(self, input_shape):
        _, c_shape = input_shape
        spatial = tuple(c_shape[:-1])
        self.hidden.build(spatial + (2,))
        self.film.build(spatial + (self.hidden_units,))

    def call(self, inputs):
        x, coords = inputs
        rc = ops.concatenate([coords[..., :1], coords[..., -1:]], axis=-1)
        h = self.film(self.hidden(ops.cast(rc, self.compute_dtype)))
        gamma, beta = ops.split(h, 2, axis=-1)
        return x * (1.0 + gamma) + beta

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[0])

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "hidden_units": self.hidden_units})
        return config


@keras.saving.register_keras_serializable(package="tornet")
class GatedSum(layers.Layer):
    """Weighted sum of K maps: inputs ``[maps (B,H,W,K), gates (B,K)]``."""

    def call(self, inputs):
        maps, gates = inputs
        gates = ops.cast(gates, maps.dtype)
        return ops.sum(maps * gates[:, None, None, :], axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[0][:-1]) + (1,)


@keras.saving.register_keras_serializable(package="tornet")
class LSEPooling(layers.Layer):
    """Log-sum-exp spatial pooling with a learnable positive temperature.

    For a map z of shape (B, H, W, C) returns (B, C)::

        s = (1/r) * log(mean_{h,w} exp(r * z))

    with r = softplus(rho) + min_temperature. By Jensen's inequality
    mean(z) <= s <= max(z); r -> 0 gives mean pooling, r -> inf gives max.
    Computed stably by subtracting the max first.
    """

    def __init__(self, init_temperature=3.0, min_temperature=1e-2, **kwargs):
        super().__init__(**kwargs)
        if init_temperature <= min_temperature:
            raise ValueError("init_temperature must be > min_temperature")
        self.init_temperature = float(init_temperature)
        self.min_temperature = float(min_temperature)

    def build(self, input_shape):
        # Inverse softplus so the initial temperature is exactly init_temperature.
        rho0 = math.log(math.expm1(self.init_temperature - self.min_temperature))
        self.rho = self.add_weight(
            name="temperature_raw",
            shape=(),
            initializer=keras.initializers.Constant(rho0),
            trainable=True,
        )

    @property
    def temperature(self):
        return ops.softplus(self.rho) + self.min_temperature

    def call(self, z):
        z = ops.cast(z, "float32")
        r = ops.cast(self.temperature, "float32")
        m = ops.stop_gradient(ops.max(z, axis=(1, 2), keepdims=True))
        lse = (
            m + ops.log(ops.mean(ops.exp(r * (z - m)), axis=(1, 2), keepdims=True)) / r
        )
        return ops.cast(lse[:, 0, 0, :], self.compute_dtype)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_temperature": self.init_temperature,
                "min_temperature": self.min_temperature,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Building blocks (same structure as train_textbook.py)
# ---------------------------------------------------------------------------


def normalize(x, name: str):
    """Scale a radar variable to roughly [-1, 1] using CHANNEL_MIN_MAX."""
    min_val, max_val = CHANNEL_MIN_MAX[name]
    mean = (max_val + min_val) / 2
    std = (max_val - min_val) / 2
    return ChannelNormalize(mean, std, name=f"normalize_{name}")(x)


def se_block(x, ratio=16):
    """Squeeze-and-excitation channel gating."""
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    # max(1, ...) only matters for tiny test models (textbook uses filters // 16).
    se = layers.Dense(max(1, filters // ratio), activation="relu")(se)
    se = layers.Dense(filters, activation="sigmoid")(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([x, se])


def wide_resnet_block(
    x, c, filters, stride, l2_reg=1e-4, drop_rate=0.0, project_shortcut=False
):
    """Pre-activation residual block of two 3x3 CoordConvs."""
    shortcut_x, shortcut_c = x, c
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x, c = CoordConv2D(
        filters,
        kernel_size=3,
        strides=(stride, stride),
        padding="same",
        activation=None,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
    )([x, c])

    if drop_rate > 0:
        x = layers.Dropout(drop_rate)(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x, c = CoordConv2D(
        filters,
        kernel_size=3,
        strides=(1, 1),
        padding="same",
        activation=None,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
    )([x, c])

    if project_shortcut or shortcut_x.shape[-1] != filters or stride != 1:
        shortcut_x, c = CoordConv2D(
            filters,
            kernel_size=1,
            strides=(stride, stride),
            padding="same",
            activation=None,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
        )([shortcut_x, shortcut_c])

    x = layers.Add()([x, shortcut_x])
    return x, c


def single_attention(x):
    """1x1 sigmoid attention map (the textbook head)."""
    return layers.Conv2D(
        1, 1, activation="sigmoid", use_bias=False, name="attention_map"
    )(x)


def multi_dilated_attention(x, dilation_rates: Sequence[int] = (1, 2, 4)):
    """Multi-scale attention: 3x3 dilated sigmoid maps fused by softmax gates.

    Branch d is a 3x3 conv with dilation d (receptive field 2d+1 pixels), so
    the branches really look at different neighbourhood sizes. Gates come
    from global average-pooled features, one softmax weight per branch.
    The fused map is the layer named ``"attention_map"``.
    """
    maps = [
        layers.Conv2D(
            1,
            3,
            dilation_rate=d,
            padding="same",
            activation="sigmoid",
            use_bias=False,
            name=f"attention_dil{d}",
        )(x)
        for d in dilation_rates
    ]
    maps = layers.Concatenate(axis=-1, name="attention_branches")(maps)
    context = layers.GlobalAveragePooling2D(name="attention_context")(x)
    gates = layers.Dense(
        len(dilation_rates), activation="softmax", name="attention_gates"
    )(context)
    return GatedSum(name="attention_map")([maps, gates])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_tornado_net(
    shape: Tuple[int, int, int] = (120, 240, 2),
    c_shape: Tuple[int, int, int] = (120, 240, 2),
    input_variables: List[str] = ALL_VARIABLES,
    start_filters: int = 48,
    l2_reg: float = 3.83e-7,
    dropout_rate: float = 0.0997,
    attention_dropout: float = 0.2,
    *,
    attention: str = "single",
    head: str = "pooled",
    shear_features: bool = False,
    range_film: bool = False,
    output_logits: bool = False,
    azimuth_spacing_deg: float = 0.5,
    shear_scale: float = 0.1,
    lse_init_temperature: float = 3.0,
    head_filters: int = 128,
) -> keras.Model:
    """Build TornadoNet. See the module docstring for what each option does.

    The defaults for ``start_filters``, ``l2_reg`` and ``dropout_rate`` are the
    values in train_textbook.py's DEFAULT_CONFIG. With all keyword-only flags
    at their defaults the model has the same layers and parameter count as
    ``train_textbook.build_model`` called with the same positional arguments.

    Inputs (dict): each of ``input_variables`` with ``shape``,
    ``"range_folded_mask"`` with ``shape``, and ``"coordinates"`` with
    ``c_shape`` (channel 0 must be R = range_m * 1e-5, as produced by
    ``tornet.data.preprocess.compute_coordinates``).

    Output: (batch, 1) probability, or logit if ``output_logits=True``.
    """
    if attention not in ATTENTION_OPTIONS:
        raise ValueError(f"attention must be one of {ATTENTION_OPTIONS}")
    if head not in HEAD_OPTIONS:
        raise ValueError(f"head must be one of {HEAD_OPTIONS}")
    if shear_features and "VEL" not in input_variables:
        raise ValueError("shear_features=True requires VEL in input_variables")

    background_flag = -3.0
    shape = tuple(shape)
    c_shape = tuple(c_shape)
    n_sweeps = shape[2]

    inputs = {v: keras.Input(shape, name=v) for v in input_variables}
    x = layers.Concatenate(axis=-1)([normalize(inputs[v], v) for v in input_variables])
    x = FillNaNs(background_flag)(x)

    range_folded = keras.Input(shape[:2] + (n_sweeps,), name="range_folded_mask")
    inputs["range_folded_mask"] = range_folded
    x = layers.Concatenate(axis=-1)([x, range_folded])

    coords = keras.Input(c_shape, name="coordinates")
    inputs["coordinates"] = coords

    if shear_features:
        shear = VelocityShear(
            azimuth_spacing_deg=azimuth_spacing_deg,
            shear_scale=shear_scale,
            name="velocity_shear",
        )([inputs["VEL"], coords])
        x = layers.Concatenate(axis=-1, name="concat_shear")([x, shear])

    x, _ = wide_resnet_block(
        x=x,
        c=coords,
        stride=1,
        filters=start_filters,
        l2_reg=l2_reg,
        drop_rate=dropout_rate,
    )

    if range_film:
        x = RangeFiLM(start_filters, name="range_film")([x, coords])

    x = se_block(x)

    x = layers.Conv2D(head_filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    if attention == "single":
        attn = single_attention(x)
    else:
        attn = multi_dilated_attention(x)
    attn = layers.Dropout(rate=attention_dropout, name="attention_dropout")(attn)
    x = layers.Multiply()([x, attn])

    final_activation = None if output_logits else "sigmoid"
    if head == "pooled":
        x_avg = layers.GlobalAveragePooling2D()(x)
        x_max = layers.GlobalMaxPooling2D()(x)
        x = layers.Concatenate()([x_avg, x_max])
        x = layers.Dense(64)(x)
        x = layers.PReLU()(x)
        output = layers.Dense(
            1, activation=final_activation, dtype="float32", name="output"
        )(x)
    else:
        logit_map = layers.Conv2D(1, 1, dtype="float32", name="instance_logits")(x)
        logit = LSEPooling(
            init_temperature=lse_init_temperature, dtype="float32", name="lse_pool"
        )(logit_map)
        output = layers.Activation(
            final_activation or "linear", dtype="float32", name="output"
        )(logit)

    return keras.Model(inputs=inputs, outputs=output, name="tornado_net")


def heatmap_model(model: keras.Model) -> keras.Model:
    """Model mapping the same inputs to the per-location logit map.

    Only available for models built with ``head="mil_lse"``.
    """
    return keras.Model(model.inputs, model.get_layer("instance_logits").output)


__all__ = [
    "build_tornado_net",
    "heatmap_model",
    "ChannelNormalize",
    "FillNaNs",
    "VelocityShear",
    "RangeFiLM",
    "GatedSum",
    "LSEPooling",
]
