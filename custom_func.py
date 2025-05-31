import tensorflow as tf
from tensorflow import keras


@tf.keras.utils.register_keras_serializable()
class FalseAlarmRate(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name="false_alarm_rate", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.epsilon = 1e-7

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred >= self.threshold, [-1]), tf.float32)

        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1.0 - y_true) * y_pred)

        if sample_weight is not None:
            sample_weight = tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)
            tp = tf.reduce_sum(sample_weight * y_true * y_pred)
            fp = tf.reduce_sum(sample_weight * (1.0 - y_true) * y_pred)

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)

    def result(self):
        precision = self.true_positives / (
            self.true_positives + self.false_positives + self.epsilon
        )
        return 1.0 - precision

    def reset_states(self):
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config


@tf.keras.utils.register_keras_serializable()
class ThreatScore(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name="threat_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")
        self.epsilon = 1e-7

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred >= self.threshold, [-1]), tf.float32)

        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1.0 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1.0 - y_pred))

        if sample_weight is not None:
            sample_weight = tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)
            tp = tf.reduce_sum(sample_weight * y_true * y_pred)
            fp = tf.reduce_sum(sample_weight * (1.0 - y_true) * y_pred)
            fn = tf.reduce_sum(sample_weight * y_true * (1.0 - y_pred))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        return self.tp / (self.tp + self.fp + self.fn + self.epsilon)

    def reset_states(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config


def focal_loss(gamma=2.0, alpha=0.85):
    def loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -tf.reduce_mean(alpha * tf.pow(1.0 - pt, gamma) * tf.math.log(pt))

    return loss_fn


def tversky_loss(alpha=0.3, beta=0.7, smooth=1e-6):
    """
    Tversky Loss: adjusts trade-off between FP and FN.
    alpha = weight for FP
    beta = weight for FN
    """

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        return 1 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)

    return loss_fn


def combo_loss(
    alpha=0.7, alpha_tversky=0.5, beta_tversky=0.5, focal_gamma=2.0, focal_alpha=0.85
):
    return lambda y_true, y_pred: alpha * tversky_loss(
        alpha=alpha_tversky, beta=beta_tversky
    )(y_true, y_pred) + (1 - alpha) * focal_loss(gamma=focal_gamma, alpha=focal_alpha)(
        y_true, y_pred
    )
