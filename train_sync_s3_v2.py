import sys
import os
import numpy as np
import json
import shutil
import keras
import logging
import tensorflow as tf
import subprocess
from tornet.data.loader import get_dataloader
from tornet.data.preprocess import get_shape
from tornet.data.constants import ALL_VARIABLES
from tornet.models.keras.losses import mae_loss
from tornet.models.keras.cnn_experiment import build_model_exp
from tornet.metrics.keras import metrics as tfm
from tornet.utils.general import make_exp_dir, make_callback_dirs
import tensorflow_datasets as tfds
import tornet.data.tfds.tornet.tornet_dataset_builder

logging.basicConfig(level=logging.ERROR)

# Environment Variables
LOCAL_EXP_DIR = "/home/ubuntu/tornet-Higgins/tmp/tornet-checkpoints"  # Local storage
S3_EXP_DIR = "s3://tornet-checkpoints/train_incep"  # S3 bucket for persistent storage
DATA_ROOT = "/home/ubuntu/tfds"
TFDS_DATA_DIR = "/home/ubuntu/tfds"

os.environ['TORNET_ROOT'] = DATA_ROOT
os.environ['TFDS_DATA_DIR'] = TFDS_DATA_DIR

# Ensure local experiment directory exists
os.makedirs(LOCAL_EXP_DIR, exist_ok=True)

logging.info(f'TORNET_ROOT={DATA_ROOT}')
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private" 
# Enable GPU Memory Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

strategy = tf.distribute.MirroredStrategy()
tf.config.optimizer.set_jit(True)  # Enable XLA (Accelerated Linear Algebra)
logging.info(f"Number of devices: {strategy.num_replicas_in_sync}")

# Default Configuration
DEFAULT_CONFIG = {
    'epochs': 200,
    'input_variables': ALL_VARIABLES,
    'train_years': list(range(2013, 2020)),
    'val_years': list(range(2020, 2022)),
    'batch_size': 128,
    'model': 'inception',
    'start_filters': 64,
    'learning_rate': 1e-3,
    'decay_steps': 2500,
    'decay_rate': 0.92,
    'l2_reg': 1e-5,
    'wN': 1.0,
    'w0': 1.0,
    'w1': 1.0,
    'w2': 2.0,
    'wW': 0.5,
    'label_smooth': 0.1,
    'loss': 'cce',
    'head': 'maxpool',
    'exp_name': 'tornet_baseline',
    'exp_dir': LOCAL_EXP_DIR,  # Use local storage
    'dataloader': "tensorflow-tfds",
    'dataloader_kwargs': {}
}

FINAL_MODEL_PATH = os.path.join(LOCAL_EXP_DIR, "final_model.keras")

def sync_from_s3():
    """Download the latest checkpoint and final model from S3 before training starts."""
    logging.info("Syncing latest checkpoint and final model from S3...")
    subprocess.run(f"aws s3 sync {S3_EXP_DIR} {LOCAL_EXP_DIR} --exact-timestamps", shell=True)

    # Check if a final model exists
    if os.path.exists(FINAL_MODEL_PATH):
        logging.info(f"Final model found: {FINAL_MODEL_PATH}")
        return FINAL_MODEL_PATH
    return None

def sync_to_s3():
    """Upload the latest checkpoint and final model to S3."""
    logging.info("Syncing latest checkpoint to S3...")
    subprocess.run(f"aws s3 sync {LOCAL_EXP_DIR} {S3_EXP_DIR} --delete", shell=True)

def save_training_state(epoch):
    """Save the last completed epoch."""
    training_state_path = os.path.join(LOCAL_EXP_DIR, "training_state.json")
    with open(training_state_path, "w") as f:
        json.dump({"last_epoch": epoch}, f)
    sync_to_s3()  # Save state to S3

def load_training_state():
    """Load the last completed epoch from file."""
    training_state_path = os.path.join(LOCAL_EXP_DIR, "training_state.json")
    if os.path.exists(training_state_path):
        with open(training_state_path, "r") as f:
            return json.load(f).get("last_epoch", 0)
    return 0  # Start from epoch 0 if no state file exists

def main(config):
    # Gather all hyperparams
    epochs=config.get('epochs')
    batch_size=config.get('batch_size')
    start_filters=config.get('start_filters')
    learning_rate=config.get('learning_rate')
    decay_steps=config.get('decay_steps')
    decay_rate=config.get('decay_rate')
    l2_reg=config.get('l2_reg')
    wN=config.get('wN')
    w0=config.get('w0')
    w1=config.get('w1')
    w2=config.get('w2')
    wW=config.get('wW')
    head=config.get('head')
    label_smooth=config.get('label_smooth')
    loss_fn = config.get('loss')
    input_variables=config.get('input_variables')
    exp_name=config.get('exp_name')
    model=config.get('model')
    exp_dir=config.get('exp_dir')
    train_years=config.get('train_years')
    val_years=config.get('val_years')
    dataloader=config.get('dataloader')
    dataloader_kwargs = config.get('dataloader_kwargs')

    weights={'wN':wN,'w0':w0,'w1':w1,'w2':w2,'wW':wW}
    dataloader_kwargs.update({'select_keys': input_variables + ['range_folded_mask', 'coordinates']})


    final_model = sync_from_s3()  # Check if the final model exists in S3

    if final_model:
        logging.info(f"Loading pre-trained final model from {final_model}")
        nn = keras.models.load_model(final_model)  # Load the fully trained model
        logging.info("Model is fully trained. No need to resume training.")
        return  # Skip training if the final model exists

    # Gather all hyperparams
    epochs = config.get('epochs')
    last_epoch = load_training_state()
    remaining_epochs = max(0, epochs - last_epoch)

    if remaining_epochs == 0:
        logging.info("All epochs already completed. Skipping training.")
        return

    logging.info(f"Resuming training from epoch {last_epoch + 1} with {remaining_epochs} epochs left.")

    # Load Data
    ds_train = get_dataloader(config['dataloader'], DATA_ROOT, config['train_years'], "train", config['batch_size'],weights, **config['dataloader_kwargs'])
    ds_val = get_dataloader(config['dataloader'], DATA_ROOT, config['val_years'], "train", config['batch_size'],weights, **config['dataloader_kwargs'])

    # Model Input Shapes
    x, _, _ = next(iter(ds_train))
    in_shapes = (None, None, get_shape(x)[-1])
    c_shapes = (None, None, x["coordinates"].shape[-1])

    # Build Model
    nn = build_model_exp(shape=in_shapes, c_shape=c_shapes, start_filters=config['start_filters'], 
                         l2_reg=config['l2_reg'], input_variables=config['input_variables'], model=config['model'])
    print(nn.summary())

    # Loss & Optimizer
    loss = keras.losses.BinaryCrossentropy(label_smoothing=config['label_smooth'])
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(config['learning_rate'], config['decay_steps'], config['decay_rate'])
    opt = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4, amsgrad=True)

    # Metrics
    from_logits=False
    metrics = [keras.metrics.AUC(from_logits=from_logits,curve='PR',name='AUCPR',num_thresholds=2000), 
                tfm.BinaryAccuracy(from_logits,name='BinaryAccuracy'), 
                tfm.TruePositives(from_logits,name='TruePositives'),
                tfm.FalsePositives(from_logits,name='FalsePositives'), 
                tfm.TrueNegatives(from_logits,name='TrueNegatives'),
                tfm.FalseNegatives(from_logits,name='FalseNegatives'), 
                tfm.Precision(from_logits,name='Precision'), 
                tfm.Recall(from_logits,name='Recall'),
                tfm.F1Score(from_logits=from_logits,name='F1')]
    nn.compile(loss=loss, metrics=metrics, optimizer=opt)

    # Experiment Directory
    expdir = LOCAL_EXP_DIR
    logging.info(f'expdir={expdir}')
    import os

    # List the checkpoint directory
    logging.info(f"Checking contents of {LOCAL_EXP_DIR}...")
    print(os.listdir(LOCAL_EXP_DIR))  # 🚨 See if the checkpoint is actually there
    latest_checkpoint = tf.train.latest_checkpoint(LOCAL_EXP_DIR)
    logging.info(f"✅ TensorFlow sees latest checkpoint as: {latest_checkpoint}")


    # Checkpoint Handling for Spot Instance
    checkpoint_path = os.path.join(expdir, "model_checkpoint_epoch_{epoch:03d}.keras")

    # Callbacks
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        checkpoint_path, monitor='val_AUCPR', save_best_only=False, save_weights_only=False
    )
    early_stop_cb = keras.callbacks.EarlyStopping(monitor='val_AUCPR', patience=5, restore_best_weights=True)
    csv_logger_cb = keras.callbacks.CSVLogger(os.path.join(LOCAL_EXP_DIR, "history.csv"), append=True)

    sync_cb = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: save_training_state(epoch + 1))

    callbacks = [checkpoint_cb, early_stop_cb, csv_logger_cb,sync_cb]

    checkpoints = sorted([f for f in os.listdir(LOCAL_EXP_DIR) if f.endswith(".keras")])

    if checkpoints:
        latest_checkpoint = os.path.join(LOCAL_EXP_DIR, checkpoints[-1])  # Load the most recent checkpoint
        print("Loading:", latest_checkpoint)

        # Load the model
        nn = keras.models.load_model(latest_checkpoint)
        print("Model successfully loaded!")
    else:
        print("No checkpoints found.")


    @tf.function
    def train_step():
        history = nn.fit(
            ds_train,
            validation_data=ds_val,
            initial_epoch=last_epoch,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        return history
    train_step()

    # Save the final trained model
    nn.save(FINAL_MODEL_PATH)
    subprocess.run(f"aws s3 cp {FINAL_MODEL_PATH} {S3_EXP_DIR}/final_model.keras", shell=True)
    logging.info(f"Final model saved to {FINAL_MODEL_PATH} and uploaded to S3.")

    save_training_state(epochs)  # Save final state

if __name__ == '__main__':
    config = DEFAULT_CONFIG
    if len(sys.argv) > 1:
        config.update(json.load(open(sys.argv[1], 'r')))
    
    main(config)