import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class FalseAlarmRate(tf.keras.metrics.Metric):
    def __init__(self, name="false_alarm_rate", **kwargs):
        super().__init__(name=name, **kwargs)
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.true_negatives = self.add_weight(name="tn", initializer="zeros")
        self.epsilon = 1e-7

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred > 0.5, [-1]), tf.float32)

        fp = tf.reduce_sum((1.0 - y_true) * y_pred)
        tn = tf.reduce_sum((1.0 - y_true) * (1.0 - y_pred))

        self.false_positives.assign_add(fp)
        self.true_negatives.assign_add(tn)

    def result(self):
        return self.false_positives / (
            self.false_positives + self.true_negatives + self.epsilon
        )

    def reset_states(self):
        self.false_positives.assign(0.0)
        self.true_negatives.assign(0.0)


@tf.keras.utils.register_keras_serializable()
class ThreatScore(tf.keras.metrics.Metric):
    def __init__(self, name="threat_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")
        self.epsilon = 1e-7

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred > 0.5, [-1]), tf.float32)

        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1.0 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1.0 - y_pred))

        if sample_weight is not None:
            sample_weight = tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)
            tp *= tf.reduce_mean(sample_weight)
            fp *= tf.reduce_mean(sample_weight)
            fn *= tf.reduce_mean(sample_weight)

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        return self.tp / (self.tp + self.fp + self.fn + self.epsilon)

    def reset_states(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)


def focal_loss(gamma=2.0, alpha=0.85):
    def loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -tf.reduce_mean(alpha * tf.pow(1.0 - pt, gamma) * tf.math.log(pt))

    return loss_fn
