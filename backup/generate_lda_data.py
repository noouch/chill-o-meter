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

def extract_embeddings_with_classes(audio_folder, model_name="facebook/wav2vec2-base"):
    """Extract wav2vec2 embeddings from audio files in class subfolders."""
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()
    
    audio_extensions = {'.wav', '.mp3'}
    class_embeddings = []
    
    # Process class subfolders (A, B, C, etc.) for LDA training
    for letter in string.ascii_uppercase:
        subfolder_path = Path(audio_folder) / letter
        if subfolder_path.exists() and subfolder_path.is_dir():
            for file_path in subfolder_path.iterdir():
                if file_path.suffix.lower() in audio_extensions:
                    full_filename = f"{letter}/{file_path.name}"
                    try:
                        waveform, sample_rate = load_audio_file(str(file_path))
                        input_values = processor(
                            waveform.numpy(), 
                            sampling_rate=16000, 
                            return_tensors="pt"
                        ).input_values
                        
                        with torch.no_grad():
                            outputs = model(input_values)
                            embeddings = outputs.last_hidden_state.squeeze().numpy()
                        
                        class_embeddings.append((embeddings, full_filename, letter))
                    except Exception as e:
                        print(f"Error processing {file_path.name}: {e}")
                        continue
    
    return class_embeddings

def perform_lda_and_save_data(embeddings_list_with_classes, output_dir="embedding_data"):
    """Perform LDA on class embeddings and save data."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine class embeddings for LDA training
    class_features = []
    class_labels = []
    class_filenames = []
    
    for embeddings, filename, class_label in embeddings_list_with_classes:
        class_features.append(embeddings)
        class_labels.extend([class_label] * len(embeddings))
        class_filenames.append(filename)
    
    class_features = np.vstack(class_features)
    
    # Check if we have at least 2 classes for LDA
    unique_classes = list(set(class_labels))
    if len(unique_classes) < 2:
        print("Error: LDA requires at least 2 classes.")
        return
    
    # Apply LDA
    n_components = min(2, len(unique_classes) - 1)
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    
    try:
        lda.fit(class_features, class_labels)
        
        # Save LDA model
        lda_data = {
            "classes": unique_classes,
            "coef": lda.coef_.tolist() if hasattr(lda, 'coef_') else [],
            "intercept": lda.intercept_.tolist() if hasattr(lda, 'intercept_') else []
        }
        
        with open(os.path.join(output_dir, "lda_model.json"), "w") as f:
            json.dump(lda_data, f)
        
        # Project embeddings using trained LDA model
        class_embeddings_2d = lda.transform(class_features)
        
        # If we only got 1 component, duplicate it for visualization
        if n_components == 1:
            class_embeddings_2d = np.hstack([class_embeddings_2d, np.zeros((class_embeddings_2d.shape[0], 1))])
        
        # Split back into per-file embeddings and save
        start_idx = 0
        file_data_list = []
        
        for embeddings, filename, class_label in embeddings_list_with_classes:
            end_idx = start_idx + len(embeddings)
            file_embeddings_2d = class_embeddings_2d[start_idx:end_idx]
            
            # Extract base filename for storage
            base_filename = os.path.basename(filename)
            
            # Save individual file data
            file_data = {
                "filename": base_filename,
                "full_path": filename,
                "class_label": class_label,
                "embeddings_2d": file_embeddings_2d.tolist()
            }
            
            # Replace forward slashes in filename
            safe_filename = filename.replace('/', '_').replace('\\', '_')
            file_data_path = os.path.join(output_dir, f"{os.path.splitext(safe_filename)[0]}_lda_embeddings.json")
            with open(file_data_path, "w") as f:
                json.dump(file_data, f)
            
            file_data_list.append({
                "filename": base_filename,
                "full_path": filename,
                "class_label": class_label,
                "data_file": f"{os.path.splitext(safe_filename)[0]}_lda_embeddings.json",
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
        
        print(f"Saved LDA embedding data for {len(embeddings_list_with_classes)} files to {output_dir}")
        
    except ValueError as e:
        print(f"Error performing LDA: {e}")

def main(audio_folder, output_dir="embedding_data"):
    """Main function to process audio files and save LDA embedding data."""
    print("Extracting embeddings from audio files (with class subfolders)...")
    class_embeddings = extract_embeddings_with_classes(audio_folder)
    
    if not class_embeddings:
        print("No valid audio files found!")
        return
    
    print("Performing LDA and saving data...")
    perform_lda_and_save_data(class_embeddings, output_dir)
    print("Done!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract wav2vec2 embeddings and perform LDA")
    parser.add_argument("audio_folder", help="Path to folder containing class subfolders (A, B, C, etc.)")
    parser.add_argument("--output", default="embedding_data", help="Output directory for embedding data")
    
    args = parser.parse_args()
    main(args.audio_folder, args.output)
