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
print(f"First 10 expected values: {expected_x_values[:10]}")

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

# Extract embeddings
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

# Load LDA model
with open("embedding_data/lda_model_layer9.json", 'r') as f:
    model_data = json.load(f)

# Create LDA model with the same parameters (matching our implementation)
n_classes = len(model_data["classes"])
n_components = min(2, n_classes - 1)
lda = LinearDiscriminantAnalysis(n_components=n_components)

# Set the model parameters
lda.classes_ = np.array(model_data["classes"])
lda.coef_ = np.array(model_data["coef"])
lda.intercept_ = np.array(model_data["intercept"])

# For scikit-learn compatibility, set additional required attributes
lda.n_features_in_ = lda.coef_.shape[1]

# Set other required attributes for a fitted LDA model
lda.xbar_ = np.zeros(lda.n_features_in_)
lda.scalings_ = np.ones((lda.n_features_in_, n_components))
lda.means_ = np.zeros((n_classes, lda.n_features_in_))
lda.priors_ = np.ones(n_classes) / n_classes
lda._max_components = n_components
lda.explained_variance_ratio_ = np.ones(n_components) / n_components

# Mark as fitted
lda._fitted = True

print(f"LDA model loaded with classes: {model_data['classes']}")
print(f"Expected input features: {lda.n_features_in_}")

# Apply LDA projection
projected = lda.transform(embeddings)
print(f"Projected shape: {projected.shape}")

# Get X-axis values (first component)
x_values = projected[:, 0]
print(f"Generated X values length: {len(x_values)}")
print(f"First 10 generated values: {x_values[:10]}")

# Compare with expected
print(f"Values match exactly: {np.allclose(x_values, expected_x_values)}")
print(f"Max difference: {np.max(np.abs(x_values - expected_x_values))}")
print(f"Mean difference: {np.mean(np.abs(x_values - expected_x_values))}")

# Show comparison of first 10 values
print("\nComparison of first 10 values:")
print("Expected       Generated      Difference")
for i in range(10):
    diff = abs(expected_x_values[i] - x_values[i])
    print(f"{expected_x_values[i]:10.6f}   {x_values[i]:10.6f}   {diff:.6f}")
