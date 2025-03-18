import tensorflow as tf

def load_and_print_model(model_path):
    """Loads a TensorFlow model from a given path and prints its structure."""
    try:
        # Load the model
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        # Print the model summary
        model.summary()
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Example usage
if __name__ == "__main__":
    model_path = ""  # Change this to your actual model path
    load_and_print_model(model_path)
``
