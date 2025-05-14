import os

import tensorflow_datasets as tfds

import tornet.tornet_dataset_builder_exp

TORNET_ROOT = "/fsx/ns1"  # where tornet files live
TFDS_DATA_DIR = "~/tfds2"
dl_config = tfds.download.DownloadConfig(manual_dir=TORNET_ROOT)
builder = tfds.builder(
    "tornet", data_dir=TFDS_DATA_DIR, **{"file_format": "array_record"}
)
builder.download_and_prepare(**{"download_config": dl_config})
