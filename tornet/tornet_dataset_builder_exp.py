"""tornet dataset."""
import os
import pathlib
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
from tornet.data.constants import ALL_VARIABLES
from typing import Dict, List, Callable
import sys
import xarray as xr
import concurrent.futures
#from tornet.data.loader import read_file

from typing import List, Dict
import numpy as np
import xarray as xr
import dask
import pandas as pd

#ALL_VARIABLES = [...]  # Define this globally

def read_file(f: str, variables: List[str] = ALL_VARIABLES, n_frames: int = 4) -> Dict[str, np.ndarray]:
    """Optimized file reading using Dask and efficient attribute handling."""
    data = {}
    
    # Use h5netcdf if files are in HDF5 format (significantly faster)
    with xr.open_dataset(f, chunks="auto") as ds:
        
        # Lazy-load radar variables and metadata
        loads = []
        for v in variables:
            data[v] = ds[v][-n_frames:, :, :, :]
            loads.append(data[v])
        
        data['range_folded_mask'] = ds['range_folded_mask'][-n_frames:, :, :, :]
        data['label'] = ds['frame_labels'][-n_frames:]
        loads.extend([data['range_folded_mask'], data['label']])

        # Load everything in parallel
        dask.compute(*loads)

        # Convert data to float32/uint8
        for v in variables:
            data[v] = data[v].astype(np.float16)

        data['range_folded_mask'] = data['range_folded_mask'].astype(np.float16)
        data['label'] = data['label'].astype(np.uint8)
        
        # Metadata extraction (use local variables to reduce attribute access time)
        attrs = ds.attrs
        category = attrs.get('category', 'NUL')
        event_id = attrs.get('event_id', 0)
        ef_number = attrs.get('ef_number', -1)

        data['category'] = np.array([{'TOR': 0, 'NUL': 1, 'WRN': 2}.get(category, 1)], dtype=np.int64)
        data['event_id'] = np.array([int(event_id)], dtype=np.int64)
        data['ef_number'] = np.array([int(ef_number)], dtype=np.int64)

        # Convert simple arrays without looping
        data['az_lower'] = np.array([ds['azimuth_limits'].values[0]])
        data['az_upper'] = np.array([ds['azimuth_limits'].values[1]])
        data['rng_lower'] = np.array([ds['range_limits'].values[0]])
        data['rng_upper'] = np.array([ds['range_limits'].values[1]])
        data['time'] = ds.time[-n_frames:].astype(np.int64) // 1e9

        # Load time metadata
        data['time'] = data['time'].load()

        # Tornado event timestamps (avoid unnecessary attribute lookups)
        if ef_number >= 0 and 'tornado_start_time' in attrs:
            tornado_start_time = pd.to_datetime(attrs['tornado_start_time']).timestamp()
            tornado_end_time = pd.to_datetime(attrs['tornado_end_time']).timestamp()
        else:
            tornado_start_time, tornado_end_time = 0, 0

        data['tornado_start_time'] = np.array([tornado_start_time], dtype=np.int64)
        data['tornado_end_time'] = np.array([tornado_end_time], dtype=np.int64)
    return data
class Builder(tfds.core.GeneratorBasedBuilder):
  """
  DatasetBuilder for tornet.  See README.md in this directory for how to build
  """

  VERSION = tfds.core.Version('1.1.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
      '1.1.0': 'Label Fix, added start/end times',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Find instructions to download TorNet on https://github.com/mit-ll/tornet
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'DBZ': tfds.features.Tensor(shape=(4, 120, 240, 2),dtype=np.float16,encoding='zlib'),
            'VEL': tfds.features.Tensor(shape=(4, 120, 240, 2),dtype=np.float16,encoding='zlib'),
            'KDP': tfds.features.Tensor(shape=(4, 120, 240, 2),dtype=np.float16,encoding='zlib'),
            'RHOHV': tfds.features.Tensor(shape=(4, 120, 240, 2),dtype=np.float16,encoding='zlib'),
            'ZDR': tfds.features.Tensor(shape=(4, 120, 240, 2),dtype=np.float16,encoding='zlib'),
            'WIDTH': tfds.features.Tensor(shape=(4, 120, 240, 2),dtype=np.float16,encoding='zlib'),
            'range_folded_mask': tfds.features.Tensor(shape=(4, 120, 240, 2),dtype=np.float16,encoding='zlib'),
            'label': tfds.features.Tensor(shape=(4,),dtype=np.uint8),
            'category': tfds.features.Tensor(shape=(1,),dtype=np.int64),
            'event_id': tfds.features.Tensor(shape=(1,),dtype=np.int64),
            'ef_number': tfds.features.Tensor(shape=(1,),dtype=np.int64),
            'az_lower': tfds.features.Tensor(shape=(1,),dtype=np.float16),
            'az_upper': tfds.features.Tensor(shape=(1,),dtype=np.float16),
            'rng_lower': tfds.features.Tensor(shape=(1,),dtype=np.float16),
            'rng_upper': tfds.features.Tensor(shape=(1,),dtype=np.float16),
            'time': tfds.features.Tensor(shape=(4,),dtype=np.int64),
            'tornado_start_time': tfds.features.Tensor(shape=(1,),dtype=np.int64),
            'tornado_end_time': tfds.features.Tensor(shape=(1,),dtype=np.int64),
        }),
        supervised_keys=None,  # Set to `None` to disable
        homepage='https://github.com/mit-ll/tornet',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    
    # Assumes data is already downloaded and extracted from tar files
    # manual_dir should point to where tar files were extracted
    archive_path = dl_manager.manual_dir
    
    # Defines the splits
    # Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train-2013': self._generate_examples(archive_path / 'train/2013'),
        'train-2014': self._generate_examples(archive_path / 'train/2014'),
        'train-2015': self._generate_examples(archive_path / 'train/2015'),
        'train-2016': self._generate_examples(archive_path / 'train/2016'),
        'train-2017': self._generate_examples(archive_path / 'train/2017'),
        'train-2018': self._generate_examples(archive_path / 'train/2018'),
        'train-2019': self._generate_examples(archive_path / 'train/2019'),
        'train-2020': self._generate_examples(archive_path / 'train/2020'),
        'train-2021': self._generate_examples(archive_path / 'train/2021'),
        'train-2022': self._generate_examples(archive_path / 'train/2022'),
        'test-2013': self._generate_examples(archive_path / 'test/2013'),
        'test-2014': self._generate_examples(archive_path / 'test/2014'),
        'test-2015': self._generate_examples(archive_path / 'test/2015'),
        'test-2016': self._generate_examples(archive_path / 'test/2016'),
        'test-2017': self._generate_examples(archive_path / 'test/2017'),
        'test-2018': self._generate_examples(archive_path / 'test/2018'),
        'test-2019': self._generate_examples(archive_path / 'test/2019'),
        'test-2020': self._generate_examples(archive_path / 'test/2020'),
        'test-2021': self._generate_examples(archive_path / 'test/2021'),
        'test-2022': self._generate_examples(archive_path / 'test/2022')
    }

  def _generate_examples(self, path):
    data_type = path.parent.name  # 'train' or 'test'
    try:
        year = int(path.name)  # Convert year from directory name
    except ValueError:
        raise ValueError(f"Expected a year (integer) as directory name, but got: {path.name}")

    # Resolve catalog path properly
    catalog_path = (path / '../../catalog.csv').resolve()
    # Read CSV efficiently (specify dtypes and load only required columns)
    catalog = pd.read_csv(
        catalog_path,
        parse_dates=['start_time', 'end_time'],
        dtype={'type': 'category', 'filename': 'string'},
        usecols=['type','start_time', 'end_time', 'filename']
    )

    # Optimize filtering using `.query()` instead of multiple filters
    catalog = catalog.query("type == @data_type and end_time.dt.year == @year")

    # Shuffle only if necessary
    catalog = catalog.sample(frac=1, random_state=1234)

    # Use pathlib for path resolution
    base_path = path.parent.parent

    def process_file(f):
        file_path = base_path / f
        if not file_path.exists():
            print(f"Warning: File not found - {file_path}")
            return None
        return (f, read_file(file_path, n_frames=4))

    # Use ThreadPoolExecutor for parallel file reads

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = executor.map(process_file, catalog.filename)

    # Yield results
    for result in results:
        if result is not None:
            yield result
 
