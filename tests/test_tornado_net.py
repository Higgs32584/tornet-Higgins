"""
Fast CPU tests for tornet.models.keras.tornado_net. No dataset needed.
"""

import math

import keras
import numpy as np
import pytest

from tornet.data.constants import ALL_VARIABLES
from tornet.models.keras.tornado_net import (
    LSEPooling,
    VelocityShear,
    build_tornado_net,
    heatmap_model,
    multi_dilated_attention,
)

SHAPE = (32, 64, 2)
SMALL = dict(shape=SHAPE, c_shape=SHAPE, start_filters=4, head_filters=8)
ALL_OPTIONS = dict(
    attention="multi_dilated",
    head="mil_lse",
    shear_features=True,
    range_film=True,
    output_logits=True,
)

# Parameter count of train_textbook.build_model(start_filters=48, ...), which
# does not depend on the image size. Checked against the script by hand.
TEXTBOOK_PARAMS_48 = 102524


def _inputs(batch=2, shape=SHAPE, seed=0):
    rng = np.random.default_rng(seed)
    x = {v: rng.normal(size=(batch, *shape)).astype("float32") for v in ALL_VARIABLES}
    x["VEL"] *= 20.0  # m/s
    for v in ("DBZ", "VEL"):
        x[v][0, :3, :5, 0] = np.nan  # NaNs must not reach the output
    x["range_folded_mask"] = np.zeros((batch, *shape), dtype="float32")
    # R = range_m * 1e-5 and 1/R, as in compute_coordinates(include_az=False)
    r = np.linspace(0.05, 0.6, shape[1], dtype="float32")
    R = np.broadcast_to(r, shape[:2])
    c = np.stack([R, 1.0 / R], axis=-1)
    x["coordinates"] = np.broadcast_to(c, (batch, *c.shape)).copy()
    return x


def test_default_matches_textbook_param_count():
    model = build_tornado_net(start_filters=48)
    assert model.count_params() == TEXTBOOK_PARAMS_48


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"attention": "multi_dilated"},
        {"head": "mil_lse"},
        {"shear_features": True},
        {"range_film": True},
        {"output_logits": True},
        ALL_OPTIONS,
    ],
    ids=["default", "multi_dilated", "mil_lse", "shear", "film", "logits", "all"],
)
def test_forward_pass(options):
    model = build_tornado_net(**SMALL, **options)
    y = keras.ops.convert_to_numpy(model(_inputs(), training=False))
    assert y.shape == (2, 1)
    assert np.all(np.isfinite(y))
    if not options.get("output_logits"):
        assert np.all((y >= 0) & (y <= 1))


def test_invalid_options_raise():
    with pytest.raises(ValueError):
        build_tornado_net(**SMALL, attention="bogus")
    with pytest.raises(ValueError):
        build_tornado_net(**SMALL, head="bogus")
    with pytest.raises(ValueError):
        build_tornado_net(**SMALL, input_variables=["DBZ"], shear_features=True)


def _shear_reference(vel, range_m, dtheta, gate=250.0, scale=0.1):
    """Straightforward numpy version of VelocityShear for one image."""
    A, R, T = vel.shape
    az = np.zeros_like(vel)
    rad = np.zeros_like(vel)
    for a in range(A):
        for r in range(R):
            for t in range(T):
                c = vel[a, r, t]
                if np.isnan(c):
                    continue
                if 0 < a < A - 1:
                    d = vel[a + 1, r, t] - vel[a - 1, r, t]
                    if np.isfinite(d):
                        az[a, r, t] = d / (2 * range_m[a, r] * dtheta)
                if 0 < r < R - 1:
                    d = vel[a, r + 1, t] - vel[a, r - 1, t]
                    if np.isfinite(d):
                        rad[a, r, t] = d / (2 * gate)
    return np.tanh(np.concatenate([az, rad], axis=-1) / scale)


def _couplet(A=16, R=12, T=2, a0=8, r0=6, dv=30.0):
    """Cyclonic couplet: inbound at lower azimuth, outbound at higher."""
    vel = np.zeros((A, R, T), dtype="float32")
    vel[a0 - 2 : a0, r0 - 1 : r0 + 2] = -dv
    vel[a0 + 1 : a0 + 3, r0 - 1 : r0 + 2] = dv
    vel[0, 0, 0] = np.nan
    vel[a0 + 4, r0, 1] = np.nan
    range_m = np.broadcast_to(np.linspace(5e3, 30e3, R, dtype="float32"), (A, R))
    coords = np.stack([range_m * 1e-5, 1e5 / range_m], axis=-1).astype("float32")
    return vel, coords, range_m


def test_shear_matches_numpy_and_flips_sign():
    vel, coords, range_m = _couplet()
    layer = VelocityShear(azimuth_spacing_deg=0.5)
    out = keras.ops.convert_to_numpy(layer([vel[None], coords[None]]))[0]
    ref = _shear_reference(vel, range_m, math.radians(0.5))
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
    assert out.shape == (16, 12, 4)
    assert np.all(np.isfinite(out))

    # Cyclonic rotation -> positive azimuthal shear at the couplet centre.
    a0, r0 = 8, 6
    assert out[a0, r0, 0] > 0.5
    flipped = keras.ops.convert_to_numpy(layer([-vel[None], coords[None]]))[0]
    np.testing.assert_allclose(flipped[..., :2], -out[..., :2], atol=1e-6)
    assert flipped[a0, r0, 0] < -0.5

    # NaN centre or neighbour -> zero shear.
    assert np.all(out[a0 + 4, r0, [1, 3]] == 0)
    assert out[a0 + 3, r0, 1] == 0 and out[a0 + 5, r0, 1] == 0


def test_lse_pooling_between_mean_and_max():
    rng = np.random.default_rng(1)
    z = rng.normal(scale=3.0, size=(4, 8, 10, 1)).astype("float32")
    mean = z.mean(axis=(1, 2))
    mx = z.max(axis=(1, 2))
    for t in (0.1, 3.0, 50.0):
        s = keras.ops.convert_to_numpy(LSEPooling(init_temperature=t)(z))
        assert s.shape == (4, 1)
        assert np.all(s >= mean - 1e-5) and np.all(s <= mx + 1e-5)
    low = keras.ops.convert_to_numpy(LSEPooling(init_temperature=0.02)(z))
    high = keras.ops.convert_to_numpy(LSEPooling(init_temperature=500.0)(z))
    np.testing.assert_allclose(low, mean, atol=0.2)
    np.testing.assert_allclose(high, mx, atol=0.05)


def test_multi_dilated_branches_have_different_receptive_fields():
    inp = keras.Input((21, 21, 2))
    out = multi_dilated_attention(inp, dilation_rates=(1, 2, 4))
    model = keras.Model(inp, out)
    x = np.zeros((1, 21, 21, 2), dtype="float32")
    x[0, 10, 10, :] = 1.0  # single nonzero pixel

    extents = {}
    for d in (1, 2, 4):
        layer = model.get_layer(f"attention_dil{d}")
        layer.set_weights([np.ones_like(w) for w in layer.get_weights()])
        branch = keras.ops.convert_to_numpy(layer(x))[0, ..., 0]
        rows, cols = np.nonzero(np.abs(branch - 0.5) > 1e-6)  # sigmoid(0) = 0.5
        extents[d] = (rows.max() - rows.min() + 1, cols.max() - cols.min() + 1)
    assert extents == {1: (3, 3), 2: (5, 5), 4: (9, 9)}

    fused = keras.ops.convert_to_numpy(model(x))
    assert fused.shape == (1, 21, 21, 1)


def test_heatmap_model():
    model = build_tornado_net(**SMALL, head="mil_lse")
    heat = keras.ops.convert_to_numpy(heatmap_model(model)(_inputs()))
    assert heat.shape == (2, *SHAPE[:2], 1)


def test_save_and_reload(tmp_path):
    keras.utils.set_random_seed(0)
    model = build_tornado_net(**SMALL, **ALL_OPTIONS)
    x = _inputs()
    before = keras.ops.convert_to_numpy(model(x, training=False))
    path = tmp_path / "tornado_net.keras"
    model.save(path)
    loaded = keras.saving.load_model(path)
    after = keras.ops.convert_to_numpy(loaded(x, training=False))
    np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("options", [{}, ALL_OPTIONS], ids=["default", "all"])
def test_one_training_step_reduces_loss(options):
    keras.utils.set_random_seed(0)
    model = build_tornado_net(
        **SMALL, dropout_rate=0.0, attention_dropout=0.0, **options
    )
    from_logits = options.get("output_logits", False)
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=from_logits)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3), loss=loss_fn, jit_compile=False
    )
    x = _inputs(batch=4)
    y = np.array([[0.0], [1.0], [0.0], [1.0]], dtype="float32")

    def loss():  # training=True: BatchNorm uses batch stats, as in the step
        return float(loss_fn(y, model(x, training=True)))

    before = loss()
    model.train_on_batch(x, y)
    assert loss() < before
