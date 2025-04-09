import keras
from keras import ops

class FromLogitsMixin:
    def __init__(self, from_logits=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.from_logits:
            y_pred = ops.sigmoid(y_pred)
        return super().update_state(y_true, y_pred, sample_weight)

class AUC(FromLogitsMixin, keras.metrics.AUC):
    ...

class BinaryAccuracy(FromLogitsMixin, keras.metrics.BinaryAccuracy):
    ...

class TruePositives(FromLogitsMixin, keras.metrics.TruePositives):
    ...

class FalsePositives(FromLogitsMixin, keras.metrics.FalsePositives):
    ...

class TrueNegatives(FromLogitsMixin, keras.metrics.TrueNegatives):
    ...

class FalseNegatives(FromLogitsMixin, keras.metrics.FalseNegatives):
    ...

class Precision(FromLogitsMixin, keras.metrics.Precision):
    ...

class Recall(FromLogitsMixin, keras.metrics.Recall):
    ...

@keras.utils.register_keras_serializable()
class F1Score(keras.metrics.Metric):
    def __init__(self, name="f1", from_logits=False, **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision(from_logits)
        self.recall = Recall(from_logits)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return (2 * p * r) / (p + r + keras.config.epsilon())

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()
