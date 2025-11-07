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

# Replicate the exact process from generate_embeddings_data.py for main embeddings
# Load audio file using the same function
def load_audio_file(file_path, target_sample_rate=16000):
    """
    Load an audio file (wav or mp3) and resample to target sample rate.
    
    Args:
        file_path (str): Path to the audio file
        target_sample_rate (int): Target sample rate for the model
    
    Returns:
        tuple: (waveform tensor, original sample rate)
    """
    # Load audio file
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
    
    return waveform.squeeze(), sample_rate

# Load audio using the exact same method
audio_file = "example_clips/jesus1.wav"
waveform, sample_rate = load_audio_file(audio_file)
print(f"Waveform shape after loading: {waveform.shape}")

# Initialize processor and model using the exact same method
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval()

# Move model to GPU if available (matching generate_embeddings_data.py)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Process audio for model input using the exact same method
input_values = processor(
    waveform.numpy(), 
    sampling_rate=16000, 
    return_tensors="pt"
).input_values
input_values = input_values.to(device)

# Extract embeddings using the exact same method for main embeddings
with torch.no_grad():
    downstream_layer = "9"  # Layer 9
    if downstream_layer:
        # For accessing intermediate layers, we need to output hidden states
        model.config.output_hidden_states = True
        outputs = model(input_values)
        
        # Try to access a specific hidden layer by index
        try:
            layer_index = int(downstream_layer)
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                if layer_index < len(outputs.hidden_states):
                    embeddings = outputs.hidden_states[layer_index].squeeze().cpu().numpy()
                else:
                    print(f"Warning: Layer {layer_index} not available, using last_hidden_state")
                    embeddings = outputs.last_hidden_state.squeeze().cpu().numpy()
            else:
                print(f"Warning: hidden_states not available, using last_hidden_state")
                embeddings = outputs.last_hidden_state.squeeze().cpu().numpy()
        except ValueError:
            print(f"Warning: Unknown downstream layer '{downstream_layer}', using last_hidden_state")
            embeddings = outputs.last_hidden_state.squeeze().cpu().numpy()
    else:
        # Default behavior - use last hidden state
        outputs = model(input_values)
        embeddings = outputs.last_hidden_state.squeeze().cpu().numpy()

print(f"Embeddings shape: {embeddings.shape}")

# Load LDA model using the same approach as in generate_embeddings_data.py
with open("embedding_data/lda_model_layer9.json", 'r') as f:
    model_data = json.load(f)

# Create LDA model with the same parameters (matching generate_embeddings_data.py)
n_classes = len(model_data["classes"])
n_components = min(2, n_classes - 1)
lda = LinearDiscriminantAnalysis(n_components=n_components)
lda.fit(np.random.rand(100, 768), np.random.choice(model_data["classes"], 100))
lda.classes_ = np.array(model_data["classes"])
lda.coef_ = np.array(model_data["coef"])
lda.intercept_ = np.array(model_data["intercept"])

print(f"LDA model loaded with classes: {model_data['classes']}")
print(f"Expected input features: {lda.n_features_in_}")

# Apply LDA projection (matching generate_embeddings_data.py)
projected = lda.transform(embeddings)
print(f"Projected shape: {projected.shape}")

# Get X-axis values (first component) - for main embeddings, we don't duplicate if we have 1 component
# In generate_embeddings_data.py, they only duplicate if n_components == 1, but we have n_components = 1
x_values = projected[:, 0]
print(f"Generated X values length: {len(x_values)}")
print(f"Generated value range: {min(x_values):.2f} to {max(x_values):.2f}")

# Compare with expected
print(f"Values match exactly: {np.allclose(x_values, expected_x_values)}")
print(f"Max difference: {np.max(np.abs(x_values - expected_x_values))}")
print(f"Mean difference: {np.mean(np.abs(x_values - expected_x_values))}")
print(f"Standard deviation of differences: {np.std(x_values - expected_x_values)}")

# Show comparison of first 10 values
print("\nComparison of first 10 values:")
print("Expected       Generated      Difference")
for i in range(10):
    diff = abs(expected_x_values[i] - x_values[i])
    print(f"{expected_x_values[i]:10.6f}   {x_values[i]:10.6f}   {diff:.6f}")

# Show comparison of last 10 values
print("\nComparison of last 10 values:")
print("Expected       Generated      Difference")
for i in range(-10, 0):
    diff = abs(expected_x_values[i] - x_values[i])
    print(f"{expected_x_values[i]:10.6f}   {x_values[i]:10.6f}   {diff:.6f}")

# Calculate correlation
correlation = np.corrcoef(x_values, expected_x_values)[0, 1]
print(f"\nCorrelation between generated and expected values: {correlation:.6f}")

# Save our results for comparison
result = {
    "audio_file": audio_file,
    "model_path": "embedding_data/lda_model_layer9.json",
    "downstream_layer": "9",
    "x_values": x_values.tolist(),
    "num_frames": len(x_values)
}

with open("our_results.json", 'w') as f:
    json.dump(result, f, indent=2)

print(f"\nOur results saved to our_results.json")
