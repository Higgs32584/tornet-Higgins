import os
import glob
import numpy as np
import tensorflow as tf
from tqdm import tqdm

NPZ_DIR = "evaluation_results"
TFRECORD_DIR = "tfrecords"
os.makedirs(TFRECORD_DIR, exist_ok=True)

# --- Helper functions for serializing data ---
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(x: dict, label: int):
    feature = {
        "label": _int64_feature(label)
    }
    for k, v in x.items():
        tensor_bytes = tf.io.serialize_tensor(v).numpy()
        feature[k] = _bytes_feature(tensor_bytes)

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# --- Conversion ---
def convert_npz_to_tfrecord(npz_path, output_path):
    data = np.load(npz_path)
    x = {k: data[k] for k in data.files if k != "label"}
    y = data["label"]

    with tf.io.TFRecordWriter(output_path) as writer:
        for i in range(len(y)):
            sample_x = {k: x[k][i] for k in x}
            serialized = serialize_example(sample_x, int(y[i]))
            writer.write(serialized)

# --- Run on all .npz files ---
npz_files = sorted(glob.glob(os.path.join(NPZ_DIR, "hard_samples_batch_*.npz")))

print(f"Converting {len(npz_files)} NPZ files to TFRecords...")

for npz_file in tqdm(npz_files):
    base_name = os.path.basename(npz_file).replace(".npz", ".tfrecord")
    tfrecord_path = os.path.join(TFRECORD_DIR, base_name)
    convert_npz_to_tfrecord(npz_file, tfrecord_path)

print("✅ All files converted.")

