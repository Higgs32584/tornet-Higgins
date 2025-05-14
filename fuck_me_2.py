import os

import tensorflow_datasets as tfds

import tornet.data.tfds.tornet.tornet_dataset_builder as tdb


# Define the optimized function
def optimized_generate_examples(self, path):
    """Optimized `_generate_examples()` function."""
    import concurrent.futures

    import pandas as pd

    from tornet.data.loader import read_file

    data_type = path.parent.name
    year = int(path.name)
    catalog_path = path.parent.parent / "catalog.csv"

    # Optimized Pandas filtering
    catalog = pd.read_csv(catalog_path, parse_dates=["start_time", "end_time"])
    catalog["year"] = catalog["end_time"].dt.year  # Avoid repeated dt.year calls
    catalog = catalog.query("type == @data_type & year == @year").sample(
        frac=1, random_state=1234
    )

    def load_example(f):
        file_path = path / ("../../" + f)
        example = read_file(file_path, n_frames=4)
        example["file_path"] = str(file_path)  # Include metadata
        example["year"] = year
        example["data_type"] = data_type
        return f, example

    # Parallel loading with ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(load_example, catalog.filename)

    yield from results


# Override the function in the module
tdb._generate_examples = optimized_generate_examples

# Run the dataset builder with the modified function
TORNET_ROOT = os.environ["TORNET_ROOT"]
TFDS_DATA_DIR = "/home/ubuntu/tfds2"

dl_config = tfds.download.DownloadConfig(manual_dir=TORNET_ROOT)
builder = tfds.builder("tornet", data_dir=TFDS_DATA_DIR, **{"file_format": "tfrecord"})
builder.download_and_prepare(**{"download_config": dl_config})
