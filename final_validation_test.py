import json
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load expected output
with open("embedding_data/jesus1_layer9_lda_embeddings.json", 'r') as f:
    expected_data = json.load(f)

expected_x_values = np.array([x[0] for x in expected_data["embeddings_2d"]])
print(f"Expected X values length: {len(expected_x_values)}")
print(f"Expected value range: {min(expected_x_values):.2f} to {max(expected_x_values):.2f}")

# Load our output
with open("test_output.json", 'r') as f:
    our_data = json.load(f)

our_x_values = np.array(our_data["x_values"])
print(f"Our X values length: {len(our_x_values)}")
print(f"Our value range: {min(our_x_values):.2f} to {max(our_x_values):.2f}")

# Compare
difference = np.abs(expected_x_values - our_x_values)
print(f"\nComparison:")
print(f"Max absolute difference: {np.max(difference):.6f}")
print(f"Mean absolute difference: {np.mean(difference):.6f}")
print(f"Standard deviation of differences: {np.std(difference):.6f}")
print(f"Values match within 1e-3: {np.all(difference < 1e-3)}")
print(f"Values match within 1e-6: {np.all(difference < 1e-6)}")

# Show first 10 values
print(f"\nFirst 10 values comparison:")
print("Index  Expected          Our               Difference")
for i in range(10):
    diff = abs(expected_x_values[i] - our_x_values[i])
    print(f"{i:2d}     {expected_x_values[i]:12.6f}   {our_x_values[i]:12.6f}   {diff:.6f}")

# Show last 10 values
print(f"\nLast 10 values comparison:")
print("Index  Expected          Our               Difference")
for i in range(-10, 0):
    diff = abs(expected_x_values[i] - our_x_values[i])
    print(f"{len(expected_x_values)+i:2d}     {expected_x_values[i]:12.6f}   {our_x_values[i]:12.6f}   {diff:.6f}")

# Calculate correlation
correlation = np.corrcoef(expected_x_values, our_x_values)[0, 1]
print(f"\nCorrelation: {correlation:.6f}")

# Calculate R-squared
ss_res = np.sum((expected_x_values - our_x_values) ** 2)
ss_tot = np.sum((expected_x_values - np.mean(expected_x_values)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print(f"R-squared: {r_squared:.6f}")

print(f"\nValidation result: {'PASS' if np.all(difference < 1e-3) else 'FAIL'}")
