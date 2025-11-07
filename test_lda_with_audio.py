import json
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import torchaudio

def load_lda_model(model_path="embedding_data/lda_model.json"):
    """
    Load the LDA model from JSON file.
    """
    with open(model_path, 'r') as f:
        model_data = json.load(f)
    
    # Create LDA model with the same parameters
    n_classes = len(model_data["classes"])
    n_components = min(2, n_classes - 1)
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    
    # Set the model parameters
    lda.classes_ = np.array(model_data["classes"])
    lda.coef_ = np.array(model_data["coef"])
    lda.intercept_ = np.array(model_data["intercept"])
    
    # For scikit-learn compatibility, set additional required attributes
    lda.n_features_in_ = lda.coef_.shape[1]
    lda.xbar_ = np.zeros(lda.n_features_in_)
    lda.scalings_ = np.ones((lda.n_features_in_, n_components))
    lda.means_ = np.zeros((n_classes, lda.n_features_in_))
    lda.priors_ = np.ones(n_classes) / n_classes
    lda._max_components = n_components
    lda.explained_variance_ratio_ = np.ones(n_components) / n_components
    
    # Mark as fitted
    lda._fitted = True
    
    print(f"Loaded LDA model with classes: {model_data['classes']}")
    print(f"LDA model expects {lda.n_features_in_} features")
    return lda

def extract_embeddings(waveform, downstream_layer=None):
    """
    Extract embeddings from audio waveform.
    """
    # Initialize processor and model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    model.eval()
    
    # Process audio for model input
    input_values = processor(
        waveform.numpy(), 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_values
    
    # Extract embeddings
    with torch.no_grad():
        if downstream_layer:
            # For accessing intermediate layers, we need to output hidden states
            model.config.output_hidden_states = True
            outputs = model(input_values)
            
            if downstream_layer == "hidden_states":
                # Use the last hidden state (default behavior)
                embeddings = outputs.last_hidden_state.squeeze().numpy()
            elif downstream_layer == "projected_states":
                # Use projected states if available
                if hasattr(outputs, 'projected_states') and outputs.projected_states is not None:
                    embeddings = outputs.projected_states.squeeze().numpy()
                else:
                    print("Warning: projected_states not available, using last_hidden_state")
                    embeddings = outputs.last_hidden_state.squeeze().numpy()
            else:
                # Try to access a specific hidden layer by index
                try:
                    layer_index = int(downstream_layer)
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                        if layer_index < len(outputs.hidden_states):
                            embeddings = outputs.hidden_states[layer_index].squeeze().numpy()
                        else:
                            print(f"Warning: Layer {layer_index} not available, using last_hidden_state")
                            embeddings = outputs.last_hidden_state.squeeze().numpy()
                    else:
                        print("Warning: hidden_states not available, using last_hidden_state")
                        embeddings = outputs.last_hidden_state.squeeze().numpy()
                except ValueError:
                    print(f"Warning: Unknown downstream layer '{downstream_layer}', using last_hidden_state")
                    embeddings = outputs.last_hidden_state.squeeze().numpy()
        else:
            # Default behavior - use last hidden state
            outputs = model(input_values)
            embeddings = outputs.last_hidden_state.squeeze().numpy()
    
    return embeddings

def apply_lda_projection(lda_model, embeddings):
    """
    Apply LDA projection to embeddings.
    """
    # Apply LDA transformation
    projected = lda_model.transform(embeddings)
    
    # If we only got 1 component, duplicate it to make it 2D for visualization
    if projected.shape[1] == 1:
        projected = np.hstack([projected, np.zeros((projected.shape[0], 1))])
    
    return projected

def main():
    print("Loading LDA model...")
    lda_model = load_lda_model()
    
    # Generate a test waveform (1 second of random audio at 16kHz)
    print("Generating test audio...")
    waveform = torch.randn(16000)
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = extract_embeddings(waveform)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Apply LDA projection
    print("Applying LDA projection...")
    projected = apply_lda_projection(lda_model, embeddings)
    print(f"Projected shape: {projected.shape}")
    
    # Calculate average X position (first component)
    avg_x = np.mean(projected[:, 0])
    print(f"Average X position: {avg_x}")
    
    # Normalize to -1 to 1 range
    normalized_value = np.tanh(avg_x)  # Maps to roughly -1 to 1
    print(f"Normalized value: {normalized_value}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    main()
