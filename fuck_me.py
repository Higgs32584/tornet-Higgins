# builds tornet in tfrecord format
import os
import tensorflow_datasets as tfds
import tornet.data.tfds.tornet.tornet_dataset_builder

TORNET_ROOT=os.environ['TORNET_ROOT'] # where tornet files live
TFDS_DATA_DIR=os.environ['TFDS_DATA_DIR'] # where tfds data is to be rewritten

dl_config=tfds.download.DownloadConfig(manual_dir=TORNET_ROOT)
builder = tfds.builder('tornet', 
                       data_dir=TFDS_DATA_DIR, 
                       **{'file_format':'tfrecord'})
builder.download_and_prepare(**{'download_config':dl_config})
