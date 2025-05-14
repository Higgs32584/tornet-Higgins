import numpy as np
import tensorflow as tf

# Load the model and sample data
model = tf.keras.models.load_model(
    "/home/ubuntu/tornet-Higgins/best_models_so_far/tornadoDetector_v6.keras"
)
sample_input = np.random.rand(1, *model.input_shape[1:]).astype(
    np.float32
)  # replace with real sample if possible

# Choose a loss and dummy label for gradient tracking
loss_fn = tf.keras.losses.BinaryCrossentropy()
dummy_label = tf.convert_to_tensor([[1.0]])  # adjust depending on your task

# For collecting info
zero_grad_params = []
low_activation_filters = []

# Set up GradientTape to track gradients
with tf.GradientTape() as tape:
    tape.watch(model.trainable_weights)
    prediction = model(sample_input, training=True)
    loss = loss_fn(dummy_label, prediction)

# Get gradients w.r.t. weights
grads = tape.gradient(loss, model.trainable_weights)

# Thresholds
grad_threshold = 1e-6
activation_threshold = 1e-3

# Analyze gradients and activations
for w, g in zip(model.trainable_weights, grads):
    if g is None:
        continue
    grad_norm = tf.reduce_sum(tf.abs(g)).numpy()
    if grad_norm < grad_threshold:
        zero_grad_params.append((w.name, grad_norm))

# Analyze activations for conv layers
activation_model = tf.keras.Model(
    inputs=model.input,
    outputs=[
        layer.output
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.Conv2D)
    ],
)

activations = activation_model(sample_input, training=False)

for layer, act in zip(activation_model.outputs, activations):
    if np.mean(np.abs(act.numpy())) < activation_threshold:
        low_activation_filters.append((layer.name, np.mean(np.abs(act.numpy()))))

# --- Results ---
print("\n🔎 Weights with near-zero gradients:")
for name, grad in zero_grad_params:
    print(f" - {name}: grad norm = {grad:.2e}")

print("\n🧊 Conv layers with low activation:")
for name, act in low_activation_filters:
    print(f" - {name}: mean activation = {act:.2e}")
