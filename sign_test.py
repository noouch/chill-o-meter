import json
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Load the LDA model from file (layer 9)
with open("embedding_data/lda_model_layer9.json", 'r') as f:
    model_data = json.load(f)

# Get coefficient and intercept
coef_array = np.array(model_data['coef'])
intercept_array = np.array(model_data['intercept'])

# Load expected output
with open("embedding_data/jesus1_layer9_lda_embeddings.json", 'r') as f:
    expected_data = json.load(f)

expected_x_values = [x[0] for x in expected_data["embeddings_2d"]]

# Load audio file and extract embeddings
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

# Manual LDA computation with different variations
first_embedding = embeddings[0]  # Shape: (768,)

# Standard computation
manual_result = np.dot(first_embedding, coef_array[0]) + intercept_array[0]
print(f"Standard computation: {manual_result}")
print(f"Expected first value: {expected_x_values[0]}")
print(f"Difference: {abs(manual_result - expected_x_values[0])}")

# Try negating
neg_result = -(np.dot(first_embedding, coef_array[0]) + intercept_array[0])
print(f"Negated computation: {neg_result}")
print(f"Difference: {abs(neg_result - expected_x_values[0])}")

# Try just the dot product without intercept
dot_only = np.dot(first_embedding, coef_array[0])
print(f"Dot product only: {dot_only}")
print(f"Difference: {abs(dot_only - expected_x_values[0])}")

# Try negating dot product only
neg_dot = -np.dot(first_embedding, coef_array[0])
print(f"Negated dot product: {neg_dot}")
print(f"Difference: {abs(neg_dot - expected_x_values[0])}")

# Try with intercept subtracted
sub_intercept = np.dot(first_embedding, coef_array[0]) - intercept_array[0]
print(f"Subtract intercept: {sub_intercept}")
print(f"Difference: {abs(sub_intercept - expected_x_values[0])}")

# Try negating with intercept subtracted
neg_sub = -(np.dot(first_embedding, coef_array[0]) - intercept_array[0])
print(f"Negated subtract intercept: {neg_sub}")
print(f"Difference: {abs(neg_sub - expected_x_values[0])}")
