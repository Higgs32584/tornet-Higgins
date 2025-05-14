import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import dask.array as da
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
import xarray as xr

from tornet.data.constants import ALL_VARIABLES

# ✅ Explicitly define dataset name
DATASET_NAME = "tornet"


def read_file_optimized(
    f: str, variables: List[str] = ALL_VARIABLES, n_frames: int = 4
) -> Dict[str, np.ndarray]:
    """Optimized file reading using Dask."""
    data = {}
    with xr.open_dataset(f, chunks="auto") as ds:  # Lazy loading with Dask

        # Read radar variables lazily and convert to NumPy efficiently
        for v in variables:
            data[v] = (
                ds[v][-n_frames:, :, :, :].load().astype(np.float32)
            )  # Only load required slices

        # Metadata processing
        data["range_folded_mask"] = (
            ds["range_folded_mask"][-n_frames:, :, :, :].astype(np.float32).load()
        )
        data["label"] = ds["frame_labels"][-n_frames:].astype(np.uint8).load()
        data["category"] = np.array(
            [{"TOR": 0, "NUL": 1, "WRN": 2}[ds.attrs["category"]]], dtype=np.int64
        )
        data["event_id"] = np.array([int(ds.attrs["event_id"])], dtype=np.int64)
        data["ef_number"] = np.array([int(ds.attrs["ef_number"])], dtype=np.int64)
        data["az_lower"] = np.array(ds["azimuth_limits"].values[0:1])
        data["az_upper"] = np.array(ds["azimuth_limits"].values[1:])
        data["rng_lower"] = np.array(ds["range_limits"].values[0:1])
        data["rng_upper"] = np.array(ds["range_limits"].values[1:])
        data["time"] = (
            (ds.time[-n_frames:].astype(np.int64) // 1e9).astype(np.int64).load()
        )

        # Tornado event timestamps
        if ds.attrs["ef_number"] >= 0 and "tornado_start_time" in ds.attrs:
            data["tornado_start_time"] = np.array(
                [pd.to_datetime(ds.attrs["tornado_start_time"]).timestamp()],
                dtype=np.int64,
            )
            data["tornado_end_time"] = np.array(
                [pd.to_datetime(ds.attrs["tornado_end_time"]).timestamp()],
                dtype=np.int64,
            )
        else:
            data["tornado_start_time"] = np.array([0], dtype=np.int64)
            data["tornado_end_time"] = np.array([0], dtype=np.int64)

    return data


class TorNetBuilder(tfds.core.GeneratorBasedBuilder):
    """Optimized DatasetBuilder for TorNet."""

    # ✅ Explicitly define dataset name
    BUILDER_CONFIGS = [
        tfds.core.BuilderConfig(name=DATASET_NAME, version=tfds.core.Version("1.1.0"))
    ]

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Download TorNet dataset and extract into the specified directory.
    """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "DBZ": tfds.features.Tensor(
                        shape=(4, 120, 240, 2), dtype=tf.float32, encoding="zlib"
                    ),
                    "VEL": tfds.features.Tensor(
                        shape=(4, 120, 240, 2), dtype=tf.float32, encoding="zlib"
                    ),
                    "KDP": tfds.features.Tensor(
                        shape=(4, 120, 240, 2), dtype=tf.float32, encoding="zlib"
                    ),
                    "RHOHV": tfds.features.Tensor(
                        shape=(4, 120, 240, 2), dtype=tf.float32, encoding="zlib"
                    ),
                    "ZDR": tfds.features.Tensor(
                        shape=(4, 120, 240, 2), dtype=tf.float32, encoding="zlib"
                    ),
                    "WIDTH": tfds.features.Tensor(
                        shape=(4, 120, 240, 2), dtype=tf.float32, encoding="zlib"
                    ),
                    "range_folded_mask": tfds.features.Tensor(
                        shape=(4, 120, 240, 2), dtype=tf.float32, encoding="zlib"
                    ),
                    "label": tfds.features.Tensor(shape=(4,), dtype=np.uint8),
                    "category": tfds.features.Tensor(shape=(1,), dtype=np.int64),
                    "event_id": tfds.features.Tensor(shape=(1,), dtype=np.int64),
                    "ef_number": tfds.features.Tensor(shape=(1,), dtype=np.int64),
                    "az_lower": tfds.features.Tensor(shape=(1,), dtype=np.float32),
                    "az_upper": tfds.features.Tensor(shape=(1,), dtype=np.float32),
                    "rng_lower": tfds.features.Tensor(shape=(1,), dtype=np.float32),
                    "rng_upper": tfds.features.Tensor(shape=(1,), dtype=np.float32),
                    "time": tfds.features.Tensor(shape=(4,), dtype=np.int64),
                    "tornado_start_time": tfds.features.Tensor(
                        shape=(1,), dtype=np.int64
                    ),
                    "tornado_end_time": tfds.features.Tensor(
                        shape=(1,), dtype=np.int64
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/mit-ll/tornet",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns all dataset splits."""
        tornet_root = dl_manager.manual_dir
        catalog_path = os.path.join(tornet_root, "catalog.csv")
        catalog = pd.read_csv(catalog_path)
        catalog["file"] = catalog["file"].apply(lambda x: os.path.join(tornet_root, x))

        splits = {
            f"train-{year}": catalog[catalog["split"] == f"train-{year}"]
            for year in range(2013, 2023)
        }
        splits.update(
            {
                f"test-{year}": catalog[catalog["split"] == f"test-{year}"]
                for year in range(2013, 2023)
            }
        )

        return [
            tfds.core.SplitGenerator(
                name=split_name, gen_kwargs={"file_paths": split_df["file"].tolist()}
            )
            for split_name, split_df in splits.items()
        ]

    def _generate_examples(self, file_paths):
        """Yields optimized examples from dataset."""
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(read_file_optimized, file_paths))

        for idx, data in enumerate(results):
            yield idx, data


# **✅ Optimized Build Script**
TORNET_ROOT = "/fsx/ns1"
TFDS_DATA_DIR = "/tfds2/"  # Where tfds data is to be rewritten

dl_config = tfds.download.DownloadConfig(manual_dir=TORNET_ROOT)
builder = tfds.builder(DATASET_NAME, data_dir=TFDS_DATA_DIR, file_format="tfrecord")
builder.download_and_prepare(download_config=dl_config)
