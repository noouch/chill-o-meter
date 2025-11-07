#!/usr/bin/env python3
"""
Real-time audio analysis using LDA model with waterfall visualization.

This script uses a trained LDA model to analyze live audio input or audio files
and displays a waterfall plot of the X-axis of the LDA projection using tkinter.
"""

import argparse
import json
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pyaudio
import threading
import queue
import time
import os
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk


class RealTimeLDAWaterfall:
    def __init__(self, model_path="embedding_data/lda_model.json", downstream_layer=None, 
                 input_device=None, audio_file=None):
        """
        Initialize the real-time LDA waterfall visualization.
        
        Args:
            model_path (str): Path to the LDA model JSON file
            downstream_layer (str): Layer to extract embeddings from
            input_device (int): Audio input device index
            audio_file (str): Path to audio file for playback
        """
        self.model_path = model_path
        self.downstream_layer = downstream_layer
        self.input_device = input_device
        self.audio_file = audio_file
        self.audio_queue = queue.Queue()
        self.running = False
        self.sample_rate = 16000  # Wav2Vec2 expects 16kHz
        self.chunk_size = 160  # Process 0.01 seconds at a time for ~100Hz update rate
        self.buffer_size = 800  # Process 0.05 seconds of audio at a time
        
        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load the LDA model
        self.lda_model = self.load_lda_model()
        
        # Initialize Wav2Vec2 model and processor
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.model.eval()
        
        # Move model to GPU if available
        if self.device.type == "cuda":
            self.model = self.model.to(self.device)
        
        # Create the tkinter window
        self.root = tk.Tk()
        self.root.title("Real-time LDA Waterfall")
        self.root.geometry("800x200")
        self.root.resizable(False, False)
        
        # Create canvas
        self.canvas = tk.Canvas(self.root, width=800, height=200, bg="black")
        self.canvas.pack()
        
        # Initialize waterfall data
        self.waterfall_width = 800
        self.waterfall_height = 200
        self.waterfall_data = []
        
        # Initialize audio processing
        if self.audio_file:
            self.audio_thread = threading.Thread(target=self.file_processing_loop, daemon=True)
        else:
            self.audio_thread = threading.Thread(target=self.audio_processing_loop, daemon=True)
        
        # Start update loop
        self.root.after(33, self.update_waterfall)  # ~30 FPS
        
    def load_lda_model(self):
        """
        Load the LDA model from JSON file.
        
        Returns:
            LinearDiscriminantAnalysis: Loaded LDA model
        """
        try:
            with open(self.model_path, 'r') as f:
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
            
            # Set other required attributes for a fitted LDA model
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
        except Exception as e:
            print(f"Error loading LDA model: {e}")
            return None
    
    def extract_embeddings(self, waveform):
        """
        Extract embeddings from audio waveform.
        
        Args:
            waveform (torch.Tensor): Audio waveform tensor
            
        Returns:
            np.array: Extracted embeddings
        """
        try:
            # Process audio for model input
            input_values = self.processor(
                waveform.numpy(), 
                sampling_rate=self.sample_rate, 
                return_tensors="pt"
            ).input_values
            
            # Move input values to GPU if available
            if self.device.type == "cuda":
                input_values = input_values.to(self.device)
            
            # Extract embeddings
            with torch.no_grad():
                if self.downstream_layer:
                    # For accessing intermediate layers, we need to output hidden states
                    self.model.config.output_hidden_states = True
                    outputs = self.model(input_values, output_hidden_states=True)
                    
                    if self.downstream_layer == "hidden_states":
                        # Use the last hidden state (default behavior)
                        embeddings = outputs.last_hidden_state.squeeze().cpu().numpy()
                    elif self.downstream_layer == "projected_states":
                        # Use projected states if available
                        if hasattr(outputs, 'projected_states') and outputs.projected_states is not None:
                            embeddings = outputs.projected_states.squeeze().cpu().numpy()
                        else:
                            print("Warning: projected_states not available, using last_hidden_state")
                            embeddings = outputs.last_hidden_state.squeeze().cpu().numpy()
                    else:
                        # Try to access a specific hidden layer by index
                        try:
                            layer_index = int(self.downstream_layer)
                            if outputs.hidden_states is not None:
                                if 0 <= layer_index < len(outputs.hidden_states):
                                    embeddings = outputs.hidden_states[layer_index].squeeze().cpu().numpy()
                                else:
                                    print(f"Warning: Layer {layer_index} not available, using last_hidden_state")
                                    embeddings = outputs.last_hidden_state.squeeze().cpu().numpy()
                            else:
                                print("Warning: hidden_states not available, using last_hidden_state")
                                embeddings = outputs.last_hidden_state.squeeze().cpu().numpy()
                        except ValueError:
                            print(f"Warning: Unknown downstream layer '{self.downstream_layer}', using last_hidden_state")
                            embeddings = outputs.last_hidden_state.squeeze().cpu().numpy()
                else:
                    # Default behavior - use last hidden state
                    outputs = self.model(input_values)
                    embeddings = outputs.last_hidden_state.squeeze().cpu().numpy()
            
            return embeddings
        except Exception as e:
            print(f"Error extracting embeddings: {e}")
            return None
    
    def apply_lda_projection(self, embeddings):
        """
        Apply LDA projection to embeddings.
        
        Args:
            embeddings (np.array): Input embeddings
            
        Returns:
            np.array: 2D projected embeddings
        """
        try:
            if self.lda_model is None:
                return None
                
            # Apply LDA transformation
            projected = self.lda_model.transform(embeddings)
            
            # If we only got 1 component, duplicate it to make it 2D for visualization
            if projected.shape[1] == 1:
                projected = np.hstack([projected, np.zeros((projected.shape[0], 1))])
                
            return projected
        except Exception as e:
            print(f"Error applying LDA projection: {e}")
            return None
    
    def audio_processing_loop(self):
        """
        Process real-time audio input.
        """
        print("Starting real-time audio processing loop...")
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Audio stream parameters
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = self.sample_rate
        CHUNK = self.chunk_size
        
        # Open audio stream
        stream_kwargs = {
            'format': FORMAT,
            'channels': CHANNELS,
            'rate': RATE,
            'input': True,
            'frames_per_buffer': CHUNK
        }
        
        if self.input_device is not None:
            stream_kwargs['input_device_index'] = self.input_device
            
        stream = p.open(**stream_kwargs)
        
        # Buffer to accumulate audio data for processing
        audio_buffer = []
        
        try:
            while self.running:
                # Read audio data
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.float32)
                audio_buffer.extend(audio_data)
                
                # When we have enough data, process it
                if len(audio_buffer) >= self.buffer_size:
                    # Convert to tensor
                    waveform = torch.tensor(audio_buffer[:self.buffer_size], dtype=torch.float32)
                    audio_buffer = audio_buffer[self.buffer_size:]  # Remove processed data
                    
                    # Calculate amplitude peaks for transparency
                    amplitude_peaks = np.abs(waveform).max().item()
                    
                    # Extract embeddings
                    embeddings = self.extract_embeddings(waveform)
                    if embeddings is None:
                        continue
                    
                    # Apply LDA projection
                    projected = self.apply_lda_projection(embeddings)
                    if projected is None:
                        continue
                    
                    # Get X-axis values (first component)
                    x_values = projected[:, 0]
                    
                    # Put the result in the queue for visualization update
                    self.audio_queue.put((x_values, amplitude_peaks))
                    
        except Exception as e:
            print(f"Error in audio processing: {e}")
        finally:
            # Clean up
            stream.stop_stream()
            stream.close()
            p.terminate()
    
    def file_processing_loop(self):
        """
        Process audio file for playback.
        """
        print(f"Starting audio file processing loop for {self.audio_file}...")
        
        try:
            # Load audio file
            waveform, sample_rate = torchaudio.load(self.audio_file)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to 1D tensor
            waveform = waveform.squeeze()
            
            # Initialize PyAudio for playback
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paFloat32,
                            channels=1,
                            rate=self.sample_rate,
                            output=True)
            
            # Process in chunks
            idx = 0
            while self.running and idx < len(waveform):
                # Get chunk of audio data
                end_idx = min(idx + self.buffer_size, len(waveform))
                chunk = waveform[idx:end_idx]
                idx = end_idx
                
                # Play the audio chunk
                stream.write(chunk.numpy().astype(np.float32).tobytes())
                
                # If chunk is smaller than buffer size, pad with zeros for processing
                if len(chunk) < self.buffer_size:
                    padding = torch.zeros(self.buffer_size - len(chunk))
                    chunk = torch.cat([chunk, padding])
                
                # Calculate amplitude peaks for transparency
                amplitude_peaks = np.abs(chunk).max().item()
                
                # Extract embeddings
                embeddings = self.extract_embeddings(chunk)
                if embeddings is None:
                    continue
                
                # Apply LDA projection
                projected = self.apply_lda_projection(embeddings)
                if projected is None:
                    continue
                
                # Get X-axis values (first component)
                x_values = projected[:, 0]
                
                # Put the result in the queue for visualization update
                self.audio_queue.put((x_values, amplitude_peaks))
                
            # Clean up
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as e:
            print(f"Error in file processing: {e}")
    
    def update_waterfall(self):
        """
        Update the waterfall plot with new data.
        """
        # Process any new audio results
        updated = False
        while not self.audio_queue.empty():
            try:
                x_values, amplitude_peaks = self.audio_queue.get_nowait()
                
                # Print debug information
                print(f"Processing {len(x_values)} x-values, range: {np.min(x_values):.2f} to {np.max(x_values):.2f}, amplitude: {amplitude_peaks:.2f}")
                
                # Add new data to the waterfall
                self.waterfall_data.insert(0, (x_values, amplitude_peaks))
                
                # Limit the number of rows to the height of the display
                if len(self.waterfall_data) > self.waterfall_height:
                    self.waterfall_data.pop()
                
                updated = True
                
            except queue.Empty:
                break
        
        # Redraw the waterfall if there's new data
        if updated:
            self.draw_waterfall()
        
        # Schedule next update
        if self.running:
            self.root.after(33, self.update_waterfall)  # ~30 FPS
    
    def draw_waterfall(self):
        """
        Draw the waterfall plot on the canvas.
        """
        # Move all existing lines down by 1 pixel
        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            if len(coords) >= 4:  # Ensure we have a line with 4 coordinates
                # Move the line down by 1 pixel
                self.canvas.coords(item, coords[0], coords[1] + 1, coords[2], coords[3] + 1)
        
        # Remove lines that have moved beyond the bottom of the canvas
        items_to_delete = []
        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            if len(coords) >= 2 and coords[1] >= self.waterfall_height:
                items_to_delete.append(item)
        
        for item in items_to_delete:
            self.canvas.delete(item)
        
        # Draw the new line at the top (row 0)
        if self.waterfall_data:
            x_values, amplitude_peaks = self.waterfall_data[0]
            
            # Calculate transparency based on amplitude peaks
            # Normalize amplitude peaks to make the effect more visible
            # We'll amplify the amplitude values to make the transparency effect more pronounced
            normalized_amplitude = min(1.0, amplitude_peaks * 5)  # Amplify and clamp to [0, 1]
            alpha = int(50 + (normalized_amplitude * 205))
            
            # Draw 2px dots for each x-value
            for x_val in x_values:
                # Map x_val from [-10, 10] to [0, 799]
                if -10 <= x_val <= 10:
                    pixel_x = int((x_val + 10) * (self.waterfall_width - 1) / 20)
                    # Ensure pixel_x is within bounds
                    pixel_x = max(0, min(self.waterfall_width - 1, pixel_x))
                    
                    # Draw the dot with transparency
                    # Since tkinter doesn't support alpha directly, we'll use a color intensity
                    # to simulate transparency
                    # Scale alpha to a color intensity (0-255) but ensure minimum visibility
                    intensity = max(100, alpha)  # Ensure minimum brightness of 100
                    # Create a color that varies from blue (low values) to red (high values)
                    if x_val < -5:
                        # More blue for low values
                        color = f"#{intensity//4:02x}{intensity//2:02x}{min(255, intensity):02x}"
                    elif x_val > 5:
                        # More red for high values
                        color = f"#{min(255, intensity):02x}{intensity//2:02x}{intensity//4:02x}"
                    else:
                        # Greenish for middle values
                        color = f"#{intensity//2:02x}{min(255, intensity):02x}{intensity//2:02x}"
                    
                    # Draw a 2px wide dot at the top of the canvas (y=0)
                    self.canvas.create_line(pixel_x-1, 0, pixel_x+1, 0, fill=color, width=2)
    
    def start(self):
        """
        Start the real-time audio analysis.
        """
        self.running = True
        
        # Start audio processing thread
        self.audio_thread.start()
        
        # Start tkinter main loop
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        self.root.mainloop()
    
    def stop(self):
        """
        Stop the real-time audio analysis.
        """
        self.running = False
        self.root.quit()
        self.root.destroy()


def detect_layer_from_filename(model_path):
    """
    Detect downstream layer from filename.
    
    Args:
        model_path (str): Path to the LDA model file
        
    Returns:
        str or None: Layer identifier or None if not found
    """
    filename = os.path.basename(model_path)
    if "layer" in filename:
        # Extract layer number from filename like "lda_model_layer9.json"
        parts = filename.split("layer")
        if len(parts) > 1:
            layer_part = parts[1].split(".")[0]  # Get part before extension
            return layer_part
    return None


def main():
    parser = argparse.ArgumentParser(description="Real-time audio analysis with LDA waterfall visualization")
    parser.add_argument("--model", default="embedding_data/lda_model_layer9.json", 
                        help="Path to LDA model JSON file")
    parser.add_argument("--downstream", help="Extract embeddings from downstream layer (overrides auto-detection)")
    parser.add_argument("--device", type=int, help="Audio input device index")
    parser.add_argument("--file", help="Path to audio file for playback")
    
    args = parser.parse_args()
    
    # Detect layer from filename if not explicitly specified
    downstream_layer = args.downstream
    if downstream_layer is None:
        downstream_layer = detect_layer_from_filename(args.model)
        if downstream_layer:
            print(f"Auto-detected downstream layer: {downstream_layer}")
    
    # Create and start the real-time LDA waterfall
    waterfall = RealTimeLDAWaterfall(
        model_path=args.model, 
        downstream_layer=downstream_layer,
        input_device=args.device,
        audio_file=args.file
    )
    
    try:
        waterfall.start()
    except KeyboardInterrupt:
        print("\nStopping...")
        waterfall.stop()


if __name__ == "__main__":
    main()
