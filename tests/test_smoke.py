"""
Fast checks that need no dataset. These run in CI on every push and PR.

The tests in test_tornet.py need the full TorNet dataset and are skipped
unless TORNET_ROOT is set.
"""

import compileall
import pathlib

import keras
import numpy as np
import pytest

from tornet.data.constants import ALL_VARIABLES
from tornet.data.preprocess import compute_coordinates
from tornet.metrics.keras import metrics as tfm
from tornet.models.keras.cnn_baseline import build_model
from tornet.models.keras.layers import FillNaNs
from tornet.models.keras.losses import dice_loss, jaccard_loss, mae_loss

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Small image so the baseline CNN builds and runs quickly on CPU.
SHAPE = (32, 64, 2)


def _random_inputs(batch=2, shape=SHAPE, seed=0):
    rng = np.random.default_rng(seed)
    x = {v: rng.normal(size=(batch, *shape)).astype("float32") for v in ALL_VARIABLES}
    x["range_folded_mask"] = np.zeros((batch, *shape), dtype="float32")
    x["coordinates"] = rng.normal(size=(batch, *shape)).astype("float32")
    return x


def test_scripts_compile():
    """Every script and package module is valid Python."""
    for folder in ("tornet", "scripts", "tests"):
        assert compileall.compile_dir(REPO_ROOT / folder, quiet=1, force=True), folder


def test_baseline_model_forward_pass():
    model = build_model(shape=SHAPE, c_shape=SHAPE, start_filters=4)
    x = _random_inputs()
    x["DBZ"][0, 0, 0, 0] = np.nan  # NaNs must be filled, not propagated
    y = model.predict(x, verbose=0)
    assert y.shape == (2, 1)
    assert np.all(np.isfinite(y))


def test_baseline_model_trains_one_step():
    model = build_model(shape=SHAPE, c_shape=SHAPE, start_filters=4)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tfm.AUC(from_logits=True, name="AUC")],
    )
    history = model.fit(_random_inputs(), np.array([[0.0], [1.0]]), epochs=1, verbose=0)
    assert np.isfinite(history.history["loss"][0])


def test_fill_nans():
    x = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype="float32")
    out = keras.ops.convert_to_numpy(FillNaNs(-3.0)(x))
    np.testing.assert_array_equal(out, [[1.0, -3.0], [-3.0, 4.0]])


def test_metrics_apply_sigmoid_to_logits():
    y_true = np.array([0.0, 1.0, 1.0, 0.0])
    logits = np.array([-5.0, 5.0, -5.0, 5.0])  # two right, two wrong
    acc = tfm.BinaryAccuracy(from_logits=True)
    acc.update_state(y_true, logits)
    assert float(acc.result()) == pytest.approx(0.5)


@pytest.mark.parametrize("loss_fn", [mae_loss, jaccard_loss, dice_loss])
def test_losses_reward_correct_predictions(loss_fn):
    y_true = np.array([[0.0], [1.0]], dtype="float32")
    good = np.array([[-10.0], [10.0]], dtype="float32")
    bad = -good
    assert float(loss_fn(y_true, good)) < float(loss_fn(y_true, bad))


def test_compute_coordinates_shape():
    d = {
        "DBZ": np.zeros((120, 240, 2)),
        "az_lower": 0.0,
        "az_upper": 60.0,
        "rng_lower": 2125.0,
        "rng_upper": 62125.0,
    }
    assert compute_coordinates(d, include_az=False).shape == (120, 240, 2)
    assert compute_coordinates(d, include_az=True).shape == (120, 240, 3)
