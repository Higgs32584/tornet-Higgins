for layer in model.layers:
    if hasattr(layer, "weights"):
        weights = layer.get_weights()[0]  # Get layer's weight matrix
        avg_weight = np.mean(np.abs(weights))
        print(f"Layer: {layer.name}, Avg Weight Magnitude: {avg_weight:.6f}")
