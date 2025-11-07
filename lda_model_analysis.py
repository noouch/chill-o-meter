import json
import numpy as np

# Load the LDA model from file (layer 9)
with open("embedding_data/lda_model_layer9.json", 'r') as f:
    model_data = json.load(f)

print(f"Model data keys: {model_data.keys()}")
print(f"Classes: {model_data['classes']}")
print(f"Coefficient shape: {np.array(model_data['coef']).shape}")
print(f"Intercept shape: {np.array(model_data['intercept']).shape}")

# Show first 10 values of coefficient
coef_array = np.array(model_data['coef'])
print(f"\nFirst 10 coefficient values: {coef_array[0, :10]}")

# Show intercept value
intercept_array = np.array(model_data['intercept'])
print(f"Intercept value: {intercept_array}")

# Let's also check what the expected output looks like
import json
with open("embedding_data/jesus1_layer9_lda_embeddings.json", 'r') as f:
    expected_data = json.load(f)

expected_x_values = [x[0] for x in expected_data["embeddings_2d"]]
print(f"\nExpected X values length: {len(expected_x_values)}")
print(f"Expected value range: {min(expected_x_values):.2f} to {max(expected_x_values):.2f}")
print(f"First 10 expected values: {expected_x_values[:10]}")
print(f"Last 10 expected values: {expected_x_values[-10:]}")

# Let's try to manually compute what the LDA transformation should produce
# For a single sample x, LDA transformation is: sign * (x @ coef.T + intercept)
# Since we have coef of shape (1, 768) and intercept of shape (1,)
# The result should be of shape (n_samples, 1)

# Let's check if we can reproduce one of the expected values
# We'll need the embeddings for this, so let's load them
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model

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

print(f"\nEmbeddings shape: {embeddings.shape}")

# Now let's manually compute the LDA transformation for the first embedding
first_embedding = embeddings[0]  # Shape: (768,)
print(f"First embedding shape: {first_embedding.shape}")

# Manual LDA computation: result = first_embedding @ coef.T + intercept
# coef is (1, 768), so coef.T is (768, 1)
# first_embedding @ coef.T should give us a scalar
manual_result = np.dot(first_embedding, coef_array[0]) + intercept_array[0]
print(f"Manual computation for first embedding: {manual_result}")
print(f"Expected first value: {expected_x_values[0]}")
print(f"Difference: {abs(manual_result - expected_x_values[0])}")

# Let's try for a few more embeddings
print("\nManual computation for first 10 embeddings:")
for i in range(10):
    manual_result = np.dot(embeddings[i], coef_array[0]) + intercept_array[0]
    print(f"Embedding {i}: Manual={manual_result:.6f}, Expected={expected_x_values[i]:.6f}, Diff={abs(manual_result - expected_x_values[i]):.6f}")
