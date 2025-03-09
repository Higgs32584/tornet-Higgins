import tensorflow as tf
import numpy as np

def check_layer_utilization(model_path):
    """Loads a .keras model and checks layer utilization based on activation statistics."""
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Create a dictionary to store layer utilization
    layer_utilization = {}
    
    # Generate random inputs matching the expected input structure
    input_shapes = model.input_shape
    
    if isinstance(input_shapes, dict):  # Handle named inputs
        random_inputs = {key: np.random.rand(*[dim if dim is not None else 1 for dim in shape]).astype(np.float32) for key, shape in input_shapes.items()}
    elif isinstance(input_shapes, tuple):  # Handle multiple inputs
        random_inputs = [np.random.rand(*[dim if dim is not None else 1 for dim in shape]).astype(np.float32) for shape in input_shapes]
    else:
        raise ValueError(f"Unsupported input structure: {input_shapes}. Expected dictionary-based named inputs or a tuple of input shapes.")
    
    # Forward pass and capture activations
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=[layer.output for layer in model.layers])
    activations = activation_model(random_inputs, training=False)  # Use function call instead of .predict()
    
    # Check utilization per layer
    for layer, activation in zip(model.layers, activations):
        if isinstance(activation, np.ndarray):
            active_neurons = np.sum(np.abs(activation) > 1e-5)  # Count nonzero activations
            total_neurons = np.prod(activation.shape) if activation.shape else 1
            utilization = active_neurons / total_neurons if total_neurons > 0 else 0
            layer_utilization[layer.name] = utilization
    
    # Print utilization report
    print("Layer Utilization Report:")
    for layer, utilization in layer_utilization.items():
        print(f"Layer: {layer}, Utilization: {utilization:.4f}")
    
    return layer_utilization

# Example usage
if __name__ == "__main__":
    model_path = "your_model.keras"  # Replace with actual model path
    layer_utilization = check_layer_utilization(model_path)