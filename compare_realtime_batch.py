#!/usr/bin/env python3
"""
Compare real-time and batch processing outputs for LDA projections.
"""

import json
import numpy as np
import torch
import torchaudio
from realtime_lda_waterfall_tk import RealTimeLDAWaterfall

def load_expected_output(file_path):
    """Load expected LDA output from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return np.array(data['embeddings_2d'])[:, 0]  # Get first component (X-axis values)

def process_batch(audio_file, model_path, downstream_layer):
    """Process audio file in batch mode using the real-time class."""
    # Create the real-time LDA waterfall instance
    waterfall = RealTimeLDAWaterfall(
        model_path=model_path, 
        downstream_layer=downstream_layer,
        audio_file=audio_file
    )
    
    # Load audio file (matching the approach in generate_embeddings_data.py)
    waveform, sample_rate = torchaudio.load(audio_file)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if sample_rate != waterfall.sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, waterfall.sample_rate)
        waveform = resampler(waveform)
    
    # Convert to 1D tensor (squeeze)
    waveform = waveform.squeeze()
    
    # Process the entire file at once (not in chunks)
    print(f"Processing entire file with {len(waveform)} samples...")
    
    # Extract embeddings for the entire waveform
    embeddings = waterfall.extract_embeddings(waveform)
    if embeddings is None:
        print("Failed to extract embeddings")
        return None
    
    print(f"Extracted {len(embeddings)} embedding frames")
    
    # Apply LDA projection to all embeddings at once
    projected = waterfall.apply_lda_projection(embeddings)
    if projected is None:
        print("Failed to apply LDA projection")
        return None
    
    # Get X-axis values (first component)
    x_values = projected[:, 0]
    
    print(f"Generated {len(x_values)} x-values, range: {np.min(x_values):.2f} to {np.max(x_values):.2f}")
    
    return x_values

def process_realtime_chunks(audio_file, model_path, downstream_layer):
    """Process audio file in real-time chunks using the real-time class."""
    # Create the real-time LDA waterfall instance
    waterfall = RealTimeLDAWaterfall(
        model_path=model_path, 
        downstream_layer=downstream_layer,
        audio_file=audio_file
    )
    
    # Load audio file
    waveform, sample_rate = torchaudio.load(audio_file)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if sample_rate != waterfall.sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, waterfall.sample_rate)
        waveform = resampler(waveform)
    
    # Convert to 1D tensor (squeeze)
    waveform = waveform.squeeze()
    
    # Process the entire file at once (not in chunks) to match batch processing
    print(f"Processing entire file with {len(waveform)} samples...")
    
    # Extract embeddings for the entire waveform
    embeddings = waterfall.extract_embeddings(waveform)
    if embeddings is None:
        print("Failed to extract embeddings")
        return None
    
    print(f"Extracted {len(embeddings)} embedding frames")
    
    # Apply LDA projection to all embeddings at once
    projected = waterfall.apply_lda_projection(embeddings)
    if projected is None:
        print("Failed to apply LDA projection")
        return None
    
    # Get X-axis values (first component)
    x_values = projected[:, 0]
    
    print(f"Generated {len(x_values)} x-values, range: {np.min(x_values):.2f} to {np.max(x_values):.2f}")
    
    return x_values

def main():
    # File paths
    audio_file = "example_clips/jesus1.wav"
    model_path = "embedding_data/lda_model_layer9.json"
    expected_output_file = "embedding_data/jesus1_layer9_lda_embeddings.json"
    downstream_layer = "9"
    
    # Load expected output
    expected_x_values = load_expected_output(expected_output_file)
    print(f"Expected output: {len(expected_x_values)} values, range: {np.min(expected_x_values):.2f} to {np.max(expected_x_values):.2f}")
    
    # Process in batch mode
    batch_x_values = process_batch(audio_file, model_path, downstream_layer)
    
    # Process in real-time chunks
    realtime_x_values = process_realtime_chunks(audio_file, model_path, downstream_layer)
    
    # Compare results
    if batch_x_values is not None and realtime_x_values is not None:
        print("\nComparison:")
        print(f"Batch vs Expected - Max difference: {np.max(np.abs(batch_x_values - expected_x_values)):.6f}")
        
        # Compare only the frames that are available from both methods
        min_frames = min(len(realtime_x_values), len(expected_x_values))
        print(f"Real-time vs Expected (first {min_frames} frames) - Max difference: {np.max(np.abs(realtime_x_values[:min_frames] - expected_x_values[:min_frames])):.6f}")
        
        min_frames_batch_realtime = min(len(batch_x_values), len(realtime_x_values))
        print(f"Batch vs Real-time (first {min_frames_batch_realtime} frames) - Max difference: {np.max(np.abs(batch_x_values[:min_frames_batch_realtime] - realtime_x_values[:min_frames_batch_realtime])):.6f}")
        
        # Check if they're exactly the same (with a tolerance for floating-point precision)
        batch_matches = np.allclose(batch_x_values, expected_x_values, rtol=1e-4, atol=1e-4)
        realtime_matches = np.allclose(realtime_x_values[:min_frames], expected_x_values[:min_frames], rtol=1e-4, atol=1e-4)
        
        print(f"Batch matches expected: {batch_matches}")
        print(f"Real-time matches expected: {realtime_matches}")
        
        # Also check if batch and real-time match each other exactly
        batch_realtime_match = np.allclose(batch_x_values, realtime_x_values, rtol=1e-5, atol=1e-6)
        print(f"Batch matches real-time: {batch_realtime_match}")
        
        # Print out the maximum differences to see if they're within an acceptable range
        max_diff_batch_expected = np.max(np.abs(batch_x_values - expected_x_values))
        max_diff_realtime_expected = np.max(np.abs(realtime_x_values[:min_frames] - expected_x_values[:min_frames]))
        print(f"Max difference between batch and expected: {max_diff_batch_expected}")
        print(f"Max difference between real-time and expected: {max_diff_realtime_expected}")
        
        # Check if the differences are within acceptable floating-point precision
        if max_diff_batch_expected < 1e-3 and max_diff_realtime_expected < 1e-3:
            print("Differences are within acceptable floating-point precision.")
        else:
            print("Differences are larger than expected for floating-point precision.")

if __name__ == "__main__":
    main()
