import os

import tensorflow as tf

# Define FSx Lustre path
fsx_path = "/fsx/ns1/tornado_dataset_test.tfdataset"

# Check if the dataset directory exists
if not os.path.exists(fsx_path):
    print(f"[ERROR] Dataset directory '{fsx_path}' does not exist!")
    exit(1)

print(f"[INFO] Found dataset at '{fsx_path}'. Attempting to load...")

try:
    # Load dataset
    loaded_ds = tf.data.Dataset.load(fsx_path)

    # Check if dataset is empty
    num_elements = sum(1 for _ in loaded_ds)
    if num_elements == 0:
        print("[ERROR] Loaded dataset is empty!")
        exit(1)

    print(f"[SUCCESS] Dataset loaded! Total elements: {num_elements}")

    # Print first 5 elements as a sample
    print("[INFO] Sample data from dataset:")
    for element in loaded_ds.take(5):
        print(element.numpy())  # Convert Tensor to NumPy for readability

except Exception as e:
    print(f"[ERROR] Failed to load dataset: {e}")
    exit(1)
