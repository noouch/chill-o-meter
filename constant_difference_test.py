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

# Load audio file and extract embeddings (as before)
audio_file = "example_clips/jesus1.wav"
waveform, sample_rate = torchaudio.load(audio_file)

if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = resampler(waveform)

waveform = waveform.squeeze()

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval()

input_values = processor(
    waveform.numpy(), 
    sampling_rate=16000, 
    return_tensors="pt"
).input_values

with torch.no_grad():
    model.config.output_hidden_states = True
    outputs = model(input_values, output_hidden_states=True)
    embeddings = outputs.hidden_states[9].squeeze().cpu().numpy()

# Load LDA model and compute decision function values (as before)
with open("embedding_data/lda_model_layer9.json", 'r') as f:
    model_data = json.load(f)

n_classes = len(model_data["classes"])
n_components = min(2, n_classes - 1)

X_dummy = np.random.rand(100, 768)
y_dummy = np.random.choice(model_data["classes"], 100)

lda = LinearDiscriminantAnalysis(n_components=n_components)
lda.fit(X_dummy, y_dummy)

lda.classes_ = np.array(model_data["classes"])
lda.coef_ = np.array(model_data["coef"])
lda.intercept_ = np.array(model_data["intercept"])

# Compute decision function values
decision_values = []
for embedding in embeddings:
    decision = np.dot(embedding, lda.coef_[0]) + lda.intercept_[0]
    decision_values.append(decision)

decision_values = np.array(decision_values)

# Check the constant difference
differences = expected_x_values - decision_values
constant_difference = np.mean(differences)
print(f"Constant difference: {constant_difference}")

# Check if all differences are approximately the same
print(f"Standard deviation of differences: {np.std(differences)}")
print(f"Min difference: {np.min(differences)}")
print(f"Max difference: {np.max(differences)}")

# Check if adding this constant to our decision values gives the expected values
adjusted_values = decision_values + constant_difference
print(f"Adjusted values match expected: {np.allclose(adjusted_values, expected_x_values)}")

# Let's also check if there's a scaling factor involved
if not np.allclose(adjusted_values, expected_x_values):
    # Try to find a scale factor and offset
    # expected = scale * decision + offset
    scale = np.sum((decision_values - np.mean(decision_values)) * (expected_x_values - np.mean(expected_x_values))) / \
            np.sum((decision_values - np.mean(decision_values)) ** 2)
    offset = np.mean(expected_x_values) - scale * np.mean(decision_values)
    
    scaled_values = scale * decision_values + offset
    print(f"Scale factor: {scale}")
    print(f"Offset: {offset}")
    print(f"Scaled values match expected: {np.allclose(scaled_values, expected_x_values)}")
    
    # Show first few values with scaling
    print("\nFirst 10 values comparison (scaled):")
    print("Expected       Scaled        Difference")
    for i in range(10):
        diff = abs(expected_x_values[i] - scaled_values[i])
        print(f"{expected_x_values[i]:10.6f}   {scaled_values[i]:10.6f}   {diff:.6f}")
