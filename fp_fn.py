import numpy as np

fp_samples = np.load("evaluation_results/false_positives_samples.npy")
fn_samples = np.load("evaluation_results/false_negatives_samples.npy")

print("False Positives Shape:", fp_samples.shape)
print("False Negatives Shape:", fn_samples.shape)
