import torch
import torchaudio
import numpy as np
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
print(f"Waveform shape: {waveform.shape}")

# Method 1: Extract embeddings using our approach (realtime_lda_waterfall_tk.py)
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
model.config.output_hidden_states = True
outputs = model(input_values, output_hidden_states=True)

# Access layer 9 embeddings
embeddings1 = outputs.hidden_states[9].squeeze().detach().cpu().numpy()
print(f"Method 1 embeddings shape: {embeddings1.shape}")

# Method 2: Extract embeddings using generate_embeddings_data.py approach
# Process audio for model input
input_values2 = processor(
    waveform.numpy(), 
    sampling_rate=16000, 
    return_tensors="pt"
).input_values

# Extract embeddings (layer 9)
with torch.no_grad():
    model.config.output_hidden_states = True
    outputs2 = model(input_values2)
    
    # Access layer 9 embeddings
    embeddings2 = outputs2.hidden_states[9].squeeze().cpu().numpy()
    print(f"Method 2 embeddings shape: {embeddings2.shape}")

# Compare embeddings
print(f"Embeddings are equal: {np.allclose(embeddings1, embeddings2)}")
print(f"Max difference: {np.max(np.abs(embeddings1 - embeddings2))}")
print(f"Mean difference: {np.mean(np.abs(embeddings1 - embeddings2))}")

# Show first few values
print(f"First 5 values from method 1: {embeddings1[0, :5]}")
print(f"First 5 values from method 2: {embeddings2[0, :5]}")
