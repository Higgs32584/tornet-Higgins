import glob
import tensorflow as tf
def count_examples_in_tfrecords(path_pattern="tfrecords/*.tfrecord"):
    files = sorted(glob.glob(path_pattern))
    return sum(1 for _ in tf.data.TFRecordDataset(files))