import json
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load expected output
with open("embedding_data/jesus1_layer9_lda_embeddings.json", 'r') as f:
    expected_data = json.load(f)

expected_x_values = [x[0] for x in expected_data["embeddings_2d"]]
print(f"Expected X values length: {len(expected_x_values)}")
print(f"Expected value range: {min(expected_x_values):.2f} to {max(expected_x_values):.2f}")

# Load audio file
audio_file = "example_clips/jesus1.wav"
waveform, sample_rate = torchaudio.load(audio_file)

# Convert to mono if stereo
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

# Resample if needed (to 16kHz)
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = resampler(waveform)

# Convert to 1D tensor
waveform = waveform.squeeze()
print(f"Waveform shape: {waveform.shape}")

# Extract embeddings using the same approach as generate_embeddings_data.py
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval()

# Process audio for model input
input_values = processor(
    waveform.numpy(), 
    sampling_rate=16000, 
    return_tensors="pt"
).input_values

# Extract embeddings (layer 9)
with torch.no_grad():
    model.config.output_hidden_states = True
    outputs = model(input_values, output_hidden_states=True)
    
    # Access layer 9 embeddings
    embeddings = outputs.hidden_states[9].squeeze().cpu().numpy()
    print(f"Embeddings shape: {embeddings.shape}")

# Load LDA model with ALL attributes (the correct approach)
with open("embedding_data/lda_model_layer9.json", 'r') as f:
    model_data = json.load(f)

# Create LDA model with the same parameters
n_classes = len(model_data["classes"])
n_components = min(2, n_classes - 1)

# Create dummy data to fit the model initially to get the right structure
X_dummy = np.random.rand(100, 768)  # 100 samples with 768 features
y_dummy = np.random.choice(model_data["classes"], 100)  # Random labels from the actual classes

# Create and fit a dummy LDA model to get the right structure
lda = LinearDiscriminantAnalysis(n_components=n_components)
lda.fit(X_dummy, y_dummy)

# BUT WAIT - we need to check if the original model was saved with all attributes
# Let's check what attributes are actually in the saved model_data
print(f"\nSaved model data keys: {model_data.keys()}")

# The original code only saved classes, coef, and intercept
# But we now know we need more attributes for transform to work correctly
# Let's check if we can reconstruct the missing attributes or if we need to save them differently

# For now, let's try to manually set the attributes that are critical for transform
lda.classes_ = np.array(model_data["classes"])
lda.coef_ = np.array(model_data["coef"])
lda.intercept_ = np.array(model_data["intercept"])

# The issue is that we don't have the other attributes that transform needs
# Let's check what happens if we try to use decision_function instead of transform
# since decision_function only needs coef_ and intercept_

decision_values = []
for embedding in embeddings:
    # Manual decision function computation (this should match what we saw earlier)
    decision = np.dot(embedding, lda.coef_[0]) + lda.intercept_[0]
    decision_values.append(decision)

decision_values = np.array(decision_values)
print(f"Decision function values shape: {decision_values.shape}")
print(f"Decision function value range: {min(decision_values):.2f} to {max(decision_values):.2f}")

# Compare with expected
print(f"\nDecision function vs Expected:")
print(f"Max difference: {np.max(np.abs(decision_values - expected_x_values))}")
print(f"Mean difference: {np.mean(np.abs(decision_values - expected_x_values))}")

# Show comparison of first 10 values
print("\nComparison of first 10 values:")
print("Expected       Decision       Difference")
for i in range(10):
    diff = abs(expected_x_values[i] - decision_values[i])
    print(f"{expected_x_values[i]:10.6f}   {decision_values[i]:10.6f}   {diff:.6f}")

# Calculate correlation
correlation = np.corrcoef(decision_values, expected_x_values)[0, 1]
print(f"\nCorrelation between decision and expected values: {correlation:.6f}")

# This suggests that the expected values might actually be the decision function values
# not the transform values. Let's check if they match exactly or need a sign flip
print(f"\nChecking if they're exact matches or need sign flip:")
print(f"Direct match: {np.allclose(decision_values, expected_x_values)}")
print(f"Sign flipped: {np.allclose(-decision_values, expected_x_values)}")

# If sign flipped works, let's use that
if np.allclose(-decision_values, expected_x_values):
    final_values = -decision_values
    print("Using sign-flipped decision function values")
else:
    final_values = decision_values
    print("Using direct decision function values")

print(f"Final values match expected: {np.allclose(final_values, expected_x_values)}")
