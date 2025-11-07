import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from pathlib import Path

def load_audio_file(file_path, target_sample_rate=16000):
    """
    Load an audio file (wav or mp3) and resample to target sample rate.
    
    Args:
        file_path (str): Path to the audio file
        target_sample_rate (int): Target sample rate for the model
    
    Returns:
        torch.Tensor: Audio waveform tensor
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
    
    return waveform.squeeze()

def extract_embeddings(audio_folder, model_name="facebook/wav2vec2-base", downstream_layer=None):
    """
    Extract wav2vec2 embeddings from all audio files in a folder.
    
    Args:
        audio_folder (str): Path to folder containing audio files
        model_name (str): Name of the wav2vec2 model to use
        downstream_layer (str): Layer to extract embeddings from (e.g., "hidden_states", "projected_states")
    
    Returns:
        list: List of tuples (embeddings, filename) for each audio file
    """
    # Initialize processor and model
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    
    # Set model to evaluation mode
    model.eval()
    
    # Supported audio formats
    audio_extensions = {'.wav', '.mp3'}
    
    all_embeddings = []
    
    # Process each audio file
    for file_path in Path(audio_folder).iterdir():
        if file_path.suffix.lower() in audio_extensions:
            print(f"Processing {file_path.name}...")
            
            try:
                # Load audio
                waveform = load_audio_file(str(file_path))
                
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
                                print(f"Warning: projected_states not available, using last_hidden_state for {file_path.name}")
                                embeddings = outputs.last_hidden_state.squeeze().numpy()
                        else:
                            # Try to access a specific hidden layer by index
                            try:
                                layer_index = int(downstream_layer)
                                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                                    if layer_index < len(outputs.hidden_states):
                                        embeddings = outputs.hidden_states[layer_index].squeeze().numpy()
                                    else:
                                        print(f"Warning: Layer {layer_index} not available, using last_hidden_state for {file_path.name}")
                                        embeddings = outputs.last_hidden_state.squeeze().numpy()
                                else:
                                    print(f"Warning: hidden_states not available, using last_hidden_state for {file_path.name}")
                                    embeddings = outputs.last_hidden_state.squeeze().numpy()
                            except ValueError:
                                print(f"Warning: Unknown downstream layer '{downstream_layer}', using last_hidden_state for {file_path.name}")
                                embeddings = outputs.last_hidden_state.squeeze().numpy()
                    else:
                        # Default behavior - use last hidden state
                        outputs = model(input_values)
                        embeddings = outputs.last_hidden_state.squeeze().numpy()
                
                all_embeddings.append((embeddings, file_path.name))
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
                continue
    
    return all_embeddings

def perform_pca(embeddings_list):
    """
    Perform PCA on all embeddings to reduce to 2D.
    
    Args:
        embeddings_list (list): List of (embeddings, filename) tuples
    
    Returns:
        list: List of (2d_embeddings, filename) tuples
    """
    # Combine all embeddings for PCA
    all_features = []
    file_indices = []  # To track which file each embedding belongs to
    filenames = []
    
    for i, (embeddings, filename) in enumerate(embeddings_list):
        all_features.append(embeddings)
        file_indices.extend([i] * len(embeddings))
        filenames.append(filename)
    
    # Flatten all embeddings
    all_features = np.vstack(all_features)
    
    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_features)
    
    # Split back into per-file embeddings
    result = []
    start_idx = 0
    
    for i, (embeddings, filename) in enumerate(embeddings_list):
        end_idx = start_idx + len(embeddings)
        file_embeddings_2d = embeddings_2d[start_idx:end_idx]
        result.append((file_embeddings_2d, filename))
        start_idx = end_idx
    
    return result

def plot_embeddings(embeddings_2d_list, output_path="embedding_plot.png"):
    """
    Plot embeddings as colored lines in 2D space.
    
    Args:
        embeddings_2d_list (list): List of (2d_embeddings, filename) tuples
        output_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Plot each file as a separate colored line
    for embeddings_2d, filename in embeddings_2d_list:
        plt.plot(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1], 
            marker='o', 
            markersize=3,
            linewidth=1,
            label=filename
        )
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Audio Embeddings Trajectories in PCA Space')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save and show plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to {output_path}")

def main(audio_folder, output_path="embedding_plot.png", downstream_layer=None):
    """
    Main function to process audio files and create visualization.
    
    Args:
        audio_folder (str): Path to folder containing audio files
        output_path (str): Path to save the plot
        downstream_layer (str): Layer to extract embeddings from
    """
    print("Extracting embeddings from audio files...")
    embeddings_list = extract_embeddings(audio_folder, downstream_layer=downstream_layer)
    
    if not embeddings_list:
        print("No valid audio files found!")
        return
    
    print("Performing PCA...")
    embeddings_2d_list = perform_pca(embeddings_list)
    
    print("Creating plot...")
    plot_embeddings(embeddings_2d_list, output_path)
    
    print("Done!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract wav2vec2 embeddings from audio files and visualize with PCA")
    parser.add_argument("audio_folder", help="Path to folder containing audio files (wav/mp3)")
    parser.add_argument("--output", default="embedding_plot.png", help="Output plot filename")
    parser.add_argument("--downstream", help="Extract embeddings from downstream layer (e.g., 'projected_states', layer index)")
    
    args = parser.parse_args()
    
    main(args.audio_folder, args.output, args.downstream)
