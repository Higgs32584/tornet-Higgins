import os

import numpy as np
import tensorflow_datasets as tfds


class HardExamples(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        features = {
            "DBZ": tfds.features.Tensor(shape=(120, 240, 2), dtype=tf.float32),
            "VEL": tfds.features.Tensor(shape=(120, 240, 2), dtype=tf.float32),
            "KDP": tfds.features.Tensor(shape=(120, 240, 2), dtype=tf.float32),
            "ZDR": tfds.features.Tensor(shape=(120, 240, 2), dtype=tf.float32),
            "RHOHV": tfds.features.Tensor(shape=(120, 240, 2), dtype=tf.float32),
            "WIDTH": tfds.features.Tensor(shape=(120, 240, 2), dtype=tf.float32),
            "range_folded_mask": tfds.features.Tensor(
                shape=(120, 240, 2), dtype=tf.float32
            ),
            "coordinates": tfds.features.Tensor(shape=(120, 240, 2), dtype=tf.float32),
            "label": tfds.features.Scalar(dtype=tf.int32),
        }

        return tfds.core.DatasetInfo(
            builder=self,
            description="Hard examples mined from Tornet predictions",
            features=tfds.features.FeaturesDict(features),
            supervised_keys=("DBZ", "label"),  # Not critical
        )

    def _split_generators(self, dl_manager):
        data_path = os.path.join(self.data_dir, "hard_data", "data.npz")
        return {
            "train": self._generate_examples(data_path),
        }

    def _generate_examples(self, npz_path):
        data = np.load(npz_path)
        num_examples = data["label"].shape[0]
        for i in range(num_examples):
            yield i, {key: data[key][i] for key in data.files}
