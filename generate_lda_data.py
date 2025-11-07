import os
import torch
import torchaudio
import numpy as np
import json
import string
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from pathlib import Path

def load_audio_file(file_path, target_sample_rate=16000):
    """Load and preprocess audio file."""
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
    return waveform.squeeze(), sample_rate

def extract_audio_peaks(waveform, sample_rate, num_embedding_frames):
    """
    Extract audio peaks at the same frame rate as the embeddings.
    
    Args:
        waveform (torch.Tensor): Audio waveform tensor
        sample_rate (int): Original sample rate of the audio
        num_embedding_frames (int): Number of embedding frames
    
    Returns:
        list: List of peak values for each frame
    """
    # Calculate the number of samples per embedding frame
    duration = waveform.shape[0] / sample_rate
    samples_per_frame = waveform.shape[0] / num_embedding_frames
    
    peaks = []
    
    # Extract peaks for each frame
    for i in range(num_embedding_frames):
        start_sample = int(i * samples_per_frame)
        end_sample = int((i + 1) * samples_per_frame)
        
        # Handle the last frame which might be shorter
        if i == num_embedding_frames - 1:
            end_sample = waveform.shape[0]
        
        # Extract the segment for this frame
        segment = waveform[start_sample:end_sample]
        
        # Calculate the peak (maximum absolute value) for this segment
        if segment.numel() > 0:
            peak = torch.max(torch.abs(segment)).item()
        else:
            peak = 0.0
        
        peaks.append(peak)
    
    return peaks

def extract_embeddings_for_file(file_path, processor, model, downstream_layer=None):
    """Extract wav2vec2 embeddings from a single audio file."""
    try:
        waveform, sample_rate = load_audio_file(str(file_path))
        
        # Get device from model
        device = next(model.parameters()).device
        
        # Process input values and move to GPU if available
        input_values = processor(
            waveform.numpy(), 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_values
        input_values = input_values.to(device)
        
        with torch.no_grad():
            if downstream_layer:
                # For accessing intermediate layers, we need to output hidden states
                model.config.output_hidden_states = True
                outputs = model(input_values)
                
                if downstream_layer == "hidden_states":
                    # Use the last hidden state (default behavior)
                    embeddings = outputs.last_hidden_state.squeeze().cpu().numpy()
                elif downstream_layer == "projected_states":
                    # Use projected states if available
                    if hasattr(outputs, 'projected_states') and outputs.projected_states is not None:
                        embeddings = outputs.projected_states.squeeze().cpu().numpy()
                    else:
                        print(f"Warning: projected_states not available, using last_hidden_state for {file_path.name}")
                        embeddings = outputs.last_hidden_state.squeeze().cpu().numpy()
                else:
                    # Try to access a specific hidden layer by index
                    try:
                        layer_index = int(downstream_layer)
                        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                            if layer_index < len(outputs.hidden_states):
                                embeddings = outputs.hidden_states[layer_index].squeeze().cpu().numpy()
                            else:
                                print(f"Warning: Layer {layer_index} not available, using last_hidden_state for {file_path.name}")
                                embeddings = outputs.last_hidden_state.squeeze().cpu().numpy()
                        else:
                            print(f"Warning: hidden_states not available, using last_hidden_state for {file_path.name}")
                            embeddings = outputs.last_hidden_state.squeeze().cpu().numpy()
                    except ValueError:
                        print(f"Warning: Unknown downstream layer '{downstream_layer}', using last_hidden_state for {file_path.name}")
                        embeddings = outputs.last_hidden_state.squeeze().cpu().numpy()
            else:
                # Default behavior - use last hidden state
                outputs = model(input_values)
                embeddings = outputs.last_hidden_state.squeeze().cpu().numpy()
        
        # Extract audio peaks at the same frame rate as embeddings
        peaks = extract_audio_peaks(waveform, sample_rate, len(embeddings))
        
        return embeddings, peaks
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return None, None

def extract_embeddings_with_classes(audio_folder, model_name="facebook/wav2vec2-base", downstream_layer=None):
    """Extract wav2vec2 embeddings from audio files in class subfolders and main folder."""
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    audio_extensions = {'.wav', '.mp3'}
    class_embeddings = []
    main_embeddings = []
    
    # Process main folder files (for visualization only, not for LDA training)
    print("Processing main folder...")
    for file_path in Path(audio_folder).iterdir():
        if file_path.suffix.lower() in audio_extensions:
            print(f"Processing {file_path.name} (for visualization only)...")
            embeddings, peaks = extract_embeddings_for_file(file_path, processor, model, downstream_layer)
            if embeddings is not None:
                main_embeddings.append((embeddings, peaks, file_path.name))
    
    # Process class subfolders (A, B, C, etc.) for LDA training
    for letter in string.ascii_uppercase:
        subfolder_path = Path(audio_folder) / letter
        if subfolder_path.exists() and subfolder_path.is_dir():
            print(f"Processing class {letter} folder...")
            for file_path in subfolder_path.iterdir():
                if file_path.suffix.lower() in audio_extensions:
                    full_filename = f"{letter}/{file_path.name}"
                    print(f"Processing {file_path.name} (class: {letter})...")
                    embeddings, peaks = extract_embeddings_for_file(file_path, processor, model, downstream_layer)
                    if embeddings is not None:
                        class_embeddings.append((embeddings, peaks, full_filename, letter))
    
    return class_embeddings, main_embeddings

def perform_lda_and_save_data(embeddings_list_with_classes, main_embeddings_list, output_dir="embedding_data", downstream_layer=None):
    """Perform LDA on class embeddings and save data for web visualization."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine class embeddings for LDA training
    class_features = []
    class_labels = []
    
    for embeddings, peaks, filename, class_label in embeddings_list_with_classes:
        class_features.append(embeddings)
        class_labels.extend([class_label] * len(embeddings))
    
    # Flatten class embeddings
    class_features = np.vstack(class_features)
    
    # Check if we have at least 2 classes for LDA
    unique_classes = list(set(class_labels))
    if len(unique_classes) < 2:
        print("Warning: LDA requires at least 2 classes. Found only:", unique_classes)
        return
    
    # Apply LDA to class embeddings only
    n_components = min(2, len(unique_classes) - 1)
    if n_components < 1:
        print("Error: Not enough classes for LDA. Need at least 2 classes.")
        return
        
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    try:
        # Train LDA model on class embeddings only
        lda.fit(class_features, class_labels)
        
        # Save LDA model
        lda_data = {
            "classes": unique_classes,
            "coef": lda.coef_.tolist() if hasattr(lda, 'coef_') else [],
            "intercept": lda.intercept_.tolist() if hasattr(lda, 'intercept_') else []
        }
        
        # Add layer info to filename if downstream is enabled
        lda_filename = "lda_model.json"
        if downstream_layer:
            lda_filename = f"lda_model_layer{downstream_layer}.json"
        
        with open(os.path.join(output_dir, lda_filename), "w") as f:
            json.dump(lda_data, f)
        
        # Project class embeddings using trained LDA model
        class_embeddings_2d = lda.transform(class_features)
        
        # If we only got 1 component, duplicate it to make it 2D for visualization
        if n_components == 1:
            class_embeddings_2d = np.hstack([class_embeddings_2d, np.zeros((class_embeddings_2d.shape[0], 1))])
        
        # Split back into per-file embeddings and save class embedding data
        start_idx = 0
        file_data_list = []
        
        for embeddings, peaks, filename, class_label in embeddings_list_with_classes:
            end_idx = start_idx + len(embeddings)
            file_embeddings_2d = class_embeddings_2d[start_idx:end_idx]
            
            # Extract base filename for storage
            base_filename = os.path.basename(filename)
            
            # Save individual file data
            file_data = {
                "filename": base_filename,
                "full_path": filename,
                "class_label": class_label,
                "embeddings_2d": file_embeddings_2d.tolist(),
                "peaks": peaks
            }
            
            # Replace forward slashes in filename
            safe_filename = filename.replace('/', '_').replace('\\', '_')
            
            # Add layer info to filename if downstream is enabled
            base_name = os.path.splitext(safe_filename)[0]
            if downstream_layer:
                data_filename = f"{base_name}_layer{downstream_layer}_lda_embeddings.json"
            else:
                data_filename = f"{base_name}_lda_embeddings.json"
            
            file_data_path = os.path.join(output_dir, data_filename)
            with open(file_data_path, "w") as f:
                json.dump(file_data, f)
            
            file_data_list.append({
                "filename": base_filename,
                "full_path": filename,
                "class_label": class_label,
                "data_file": data_filename,
                "num_frames": len(embeddings)
            })
            
            start_idx = end_idx
        
        # Project main embeddings using the same trained LDA model
        if main_embeddings_list:
            # Combine main embeddings
            main_features = []
            for embeddings, peaks, filename in main_embeddings_list:
                main_features.append(embeddings)
            
            # Flatten main embeddings
            main_features = np.vstack(main_features)
            
            # Project main embeddings using trained LDA model
            main_embeddings_2d = lda.transform(main_features)
            
            # If we only got 1 component, duplicate it to make it 2D for visualization
            if n_components == 1:
                main_embeddings_2d = np.hstack([main_embeddings_2d, np.zeros((main_embeddings_2d.shape[0], 1))])
            
            # Split back into per-file embeddings and save main embedding data
            start_idx = 0
            for embeddings, peaks, filename in main_embeddings_list:
                end_idx = start_idx + len(embeddings)
                file_embeddings_2d = main_embeddings_2d[start_idx:end_idx]
                
                # Save individual file data (no class label for main files)
                file_data = {
                    "filename": filename,
                    "embeddings_2d": file_embeddings_2d.tolist(),
                    "peaks": peaks
                }
                
                # Add layer info to filename if downstream is enabled
                base_name = os.path.splitext(filename)[0]
                if downstream_layer:
                    data_filename = f"{base_name}_layer{downstream_layer}_lda_embeddings.json"
                else:
                    data_filename = f"{base_name}_lda_embeddings.json"
                
                file_data_path = os.path.join(output_dir, data_filename)
                with open(file_data_path, "w") as f:
                    json.dump(file_data, f)
                
                file_data_list.append({
                    "filename": filename,
                    "data_file": data_filename,
                    "num_frames": len(embeddings)
                })
                
                start_idx = end_idx
        
        # Save file index
        index_data = {
            "files": file_data_list,
            "lda": True
        }
        
        with open(os.path.join(output_dir, "file_index.json"), "w") as f:
            json.dump(index_data, f)
        
        print(f"Saved LDA embedding data for {len(embeddings_list_with_classes) + len(main_embeddings_list)} files to {output_dir}")
        
    except ValueError as e:
        print(f"Error performing LDA: {e}")
        print("LDA requires more samples than classes. Try adding more audio files.")

def main(audio_folder, output_dir="embedding_data", downstream_layer=None):
    """Main function to process audio files and save LDA embedding data."""
    print("Extracting embeddings from audio files (with class subfolders)...")
    class_embeddings, main_embeddings = extract_embeddings_with_classes(audio_folder, downstream_layer=downstream_layer)
    
    if not class_embeddings and not main_embeddings:
        print("No valid audio files found!")
        return
    
    print("Performing LDA and saving data...")
    perform_lda_and_save_data(class_embeddings, main_embeddings, output_dir, downstream_layer)
    print("Done!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract wav2vec2 embeddings and perform LDA")
    parser.add_argument("audio_folder", help="Path to folder containing class subfolders (A, B, C, etc.) and main files")
    parser.add_argument("--output", default="embedding_data", help="Output directory for embedding data")
    parser.add_argument("--downstream", help="Extract embeddings from downstream layer (e.g., 'projected_states', layer index)")
    
    args = parser.parse_args()
    main(args.audio_folder, args.output, args.downstream)
