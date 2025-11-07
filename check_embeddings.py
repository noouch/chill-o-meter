import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Initialize processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval()

# Generate a test waveform
waveform = torch.randn(16000)  # 1 second of random audio at 16kHz

# Process audio for model input
input_values = processor(
    waveform.numpy(), 
    sampling_rate=16000, 
    return_tensors="pt"
).input_values

print("Input values shape:", input_values.shape)

# Extract embeddings with different layers
print("\n--- Default (last_hidden_state) ---")
with torch.no_grad():
    outputs = model(input_values)
    embeddings = outputs.last_hidden_state.squeeze().detach().numpy()
    print("Embeddings shape:", embeddings.shape)

print("\n--- Hidden states ---")
model.config.output_hidden_states = True
with torch.no_grad():
    outputs = model(input_values)
    print("Number of hidden states:", len(outputs.hidden_states))
    for i, hidden_state in enumerate(outputs.hidden_states):
        print(f"  Layer {i} shape:", hidden_state.shape)

print("\n--- Projected states ---")
if hasattr(outputs, 'projected_states') and outputs.projected_states is not None:
    print("Projected states shape:", outputs.projected_states.shape)
else:
    print("No projected states available")
