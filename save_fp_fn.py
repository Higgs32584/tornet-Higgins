import argparse
import logging
import os

import keras
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import tornet.data.tfds.tornet.tornet_dataset_builder
from tornet.data.loader import get_dataloader

logging.basicConfig(level=logging.INFO)

# Set dataset paths
DATA_ROOT = '/home/ubuntu/tfds'
os.environ['TORNET_ROOT'] = DATA_ROOT
os.environ['TFDS_DATA_DIR'] = DATA_ROOT

# Save a chunk of hard examples
def save_multiinput_hard_examples(hard_examples, name):
    if not hard_examples:
        return
    grouped = {}
    for key in hard_examples[0].keys():
        try:
            grouped[key] = np.stack([ex[key] for ex in hard_examples])
        except:
            grouped[key] = np.array([ex[key] for ex in hard_examples])
    np.savez(f"evaluation_results/{name}.npz", **grouped)
    logging.info(f"Saved {len(hard_examples)} examples to evaluation_results/{name}.npz")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="Pretrained model to test (.keras)", required=True)
    args = parser.parse_args()

    os.makedirs("evaluation_results_val", exist_ok=True)
    model = keras.saving.load_model(args.model_path, compile=False)

    dataloader = "tensorflow-tfds"
    train_years = range(2013, 2021)  # You can expand this once memory issues are solved
    ds_train = get_dataloader(
        dataloader, DATA_ROOT, train_years, "train", 128, select_keys=list(model.input.keys())
    )

    hard_examples = []
    batch_index = 0

    for x_batch, y_batch in ds_train:
        probs = model.predict_on_batch(x_batch)
        preds = (probs > 0.0101).astype(np.int32).flatten()
        y_true = y_batch.numpy().astype(np.int32).flatten()

        x_np = {k: v.numpy() for k, v in x_batch.items()}

        for i in range(len(y_true)):
            if (preds[i] == 0 and y_true[i] == 1) or \
               (preds[i] == 1 and y_true[i] == 1):

                sample = {k: x_np[k][i] for k in x_np}
                if (preds[i] == 0 and y_true[i] == 1):
                    sample["label"] = 1
                elif()
                hard_examples.append(sample)

        # Save every 1000 hard examples
        if len(hard_examples) >= 1000:
            save_multiinput_hard_examples(hard_examples, name=f"hard_samples_batch_{batch_index}")
            batch_index += 1
            hard_examples = []

    # Save any remaining examples
    if hard_examples:
        save_multiinput_hard_examples(hard_examples, name=f"hard_samples_batch_{batch_index}")

if __name__ == "__main__":
    main()
