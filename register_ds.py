import tensorflow_datasets as tfds

builder = tfds.builder_from_directory(
    "~/my_tfds/hard_examples"
)
builder.download_and_prepare()