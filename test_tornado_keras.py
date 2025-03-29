"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

import os
import keras
import tqdm
import tensorflow as tf
from tornet.data.loader import get_dataloader
from tornet.metrics.keras import metrics as tfm

import argparse
import logging
logging.basicConfig(level=logging.INFO)


TFDS_DATA_DIR="/home/ubuntu/tfds"
EXP_DIR=os.environ.get('EXP_DIR','.')
TORNET_ROOT=TFDS_DATA_DIR
#TFDS_DATA_DIR=os.environ['TFDS_DATA_DIR']
import tensorflow_datasets as tfds
import tornet.data.tfds.tornet.tornet_dataset_builder # registers 'tornet'
EXP_DIR = "."
DATA_ROOT = '/home/ubuntu/tfds'
TORNET_ROOT=DATA_ROOT
TFDS_DATA_DIR = '/home/ubuntu/tfds'
DATA_ROOT = "/home/ubuntu/tfds"
TFDS_DATA_DIR = "/home/ubuntu/tfds"
os.environ['TORNET_ROOT']= DATA_ROOT
os.environ['TFDS_DATA_DIR']=TFDS_DATA_DIR
class FalseAlarmRate(tf.keras.metrics.Metric):
    def __init__(self, name="false_alarm_rate", **kwargs):
        super().__init__(name=name, **kwargs)
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.epsilon = 1e-7

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Binary predictions
        y_true = tf.cast(y_true, tf.float32)

        fp = tf.reduce_sum((1 - y_true) * y_pred)
        tp = tf.reduce_sum(y_true * y_pred)

        self.false_positives.assign_add(fp)
        self.true_positives.assign_add(tp)

    def result(self):
        return self.false_positives / (self.false_positives + self.true_positives + self.epsilon)

    def reset_states(self):
        self.false_positives.assign(0)
        self.true_positives.assign(0)

class ThreatScore(tf.keras.metrics.Metric):
    def __init__(self, name="threat_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")
        self.epsilon = 1e-7

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.fp.assign_add(tf.reduce_sum((1 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1 - y_pred)))

    def result(self):
        return self.tp / (self.tp + self.fp + self.fn + self.epsilon)

    def reset_states(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)


#logging.info('TORNET_ROOT='+TORNET_ROOT)
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        help="Pretrained model to test (.keras)",
                        default=None)
    args = parser.parse_args()

    trained_model = args.model_path
        
    dataloader = "tensorflow-tfds"

    logging.info(f"Using {keras.config.backend()} backend")
    logging.info(f"Using {dataloader} dataloader")

    if ("tfds" in dataloader) and ('TFDS_DATA_DIR' in os.environ):
        logging.info('Using TFDS dataset location at '+os.environ['TFDS_DATA_DIR'])
    
    # load model
    model = keras.saving.load_model(trained_model,compile=False)

    ## Set up data loader
    import tensorflow_datasets as tfds
    import tornet.data.tfds.tornet.tornet_dataset_builder # registers 'tornet'
    test_years = range(2013,2023)
    ds_test = get_dataloader(dataloader, TORNET_ROOT, test_years, 
                             "test", 
                             64,
                             select_keys=list(model.input.keys()))
    #ds_train = get_dataloader(dataloader, DATA_ROOT, train_years, "test", batch_size, weights, **dataloader_kwargs)


    # Compute various metrics
    from_logits=False
    metrics = [
                keras.metrics.AUC(from_logits=from_logits,curve='PR',name='AUCPR',num_thresholds=1000), 
                tfm.BinaryAccuracy(from_logits,name='BinaryAccuracy'), 
                tfm.TruePositives(from_logits,name='TruePositives'),
                tfm.FalsePositives(from_logits,name='FalsePositives'), 
                tfm.TrueNegatives(from_logits,name='TrueNegatives'),
                tfm.FalseNegatives(from_logits,name='FalseNegatives'), 
                tfm.Precision(from_logits,name='Precision'), 
                tfm.Recall(from_logits,name='Recall'),
                FalseAlarmRate(name='FalseAlarmRate'),
                tfm.F1Score(from_logits=from_logits,name='F1'),
                ThreatScore(name='ThreatScore')]
    
    model.compile(metrics=metrics)

    scores = model.evaluate(ds_test) 
    scores = {m.name:scores[k+1] for k,m in enumerate(metrics)}

    logging.info(scores)

 
if __name__=='__main__':
    main()
