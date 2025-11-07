#!/usr/bin/env python3
"""
Real-time audio analysis using LDA model with waterfall visualization.

This script uses a trained LDA model to analyze live audio input or audio files
and displays a waterfall plot of the X-axis of the LDA projection.
"""

import argparse
import json
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyaudio
import threading
import queue
import time
import os


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
        self.chunk_size = 320  # Process 0.02 seconds at a time for ~50Hz update rate
        self.buffer_size = 3200  # Process 0.2 seconds of audio at a time
        
        # Load the LDA model
        self.lda_model = self.load_lda_model()
        
        # Initialize Wav2Vec2 model and processor
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.model.eval()
        
        # Create the plot
        self.fig, self.ax = plt.subplots(figsize=(8, 2))  # 8 inches * 100 dpi = 800px, 2 inches * 100 dpi = 200px
        self.fig.canvas.manager.set_window_title("Real-time LDA Waterfall")
        
        # Initialize waterfall data
        self.waterfall_width = 800  # pixels
        self.waterfall_height = 200  # pixels
        self.waterfall_data = np.zeros((self.waterfall_height, self.waterfall_width))
        
        # Create image plot
        self.im = self.ax.imshow(self.waterfall_data, cmap='viridis', aspect='auto', 
                                vmin=-10, vmax=2, extent=[0, self.waterfall_width, 0, self.waterfall_height])
        
        # Set plot properties
        self.ax.set_xlim(0, self.waterfall_width)
        self.ax.set_ylim(0, self.waterfall_height)
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Samples')
        self.ax.set_title('LDA X-Axis Projection Waterfall')
        
        # Add colorbar
        cbar = self.fig.colorbar(self.im, ax=self.ax)
        cbar.set_label('LDA X-Value')
        
        # Initialize audio processing
        if self.audio_file:
            self.audio_thread = threading.Thread(target=self.file_processing_loop, daemon=True)
        else:
            self.audio_thread = threading.Thread(target=self.audio_processing_loop, daemon=True)
        
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
            
            # Extract embeddings
            with torch.no_grad():
                if self.downstream_layer:
                    # For accessing intermediate layers, we need to output hidden states
                    self.model.config.output_hidden_states = True
                    outputs = self.model(input_values)
                    
                    if self.downstream_layer == "hidden_states":
                        # Use the last hidden state (default behavior)
                        embeddings = outputs.last_hidden_state.squeeze().numpy()
                    elif self.downstream_layer == "projected_states":
                        # Use projected states if available
                        if hasattr(outputs, 'projected_states') and outputs.projected_states is not None:
                            embeddings = outputs.projected_states.squeeze().numpy()
                        else:
                            print("Warning: projected_states not available, using last_hidden_state")
                            embeddings = outputs.last_hidden_state.squeeze().numpy()
                    else:
                        # Try to access a specific hidden layer by index
                        try:
                            layer_index = int(self.downstream_layer)
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
                            print(f"Warning: Unknown downstream layer '{self.downstream_layer}', using last_hidden_state")
                            embeddings = outputs.last_hidden_state.squeeze().numpy()
                else:
                    # Default behavior - use last hidden state
                    outputs = self.model(input_values)
                    embeddings = outputs.last_hidden_state.squeeze().numpy()
            
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
                    self.audio_queue.put(x_values)
                    
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
            
            # Process in chunks
            idx = 0
            while self.running and idx < len(waveform):
                # Get chunk of audio data
                end_idx = min(idx + self.buffer_size, len(waveform))
                chunk = waveform[idx:end_idx]
                idx = end_idx
                
                # If chunk is smaller than buffer size, pad with zeros
                if len(chunk) < self.buffer_size:
                    padding = torch.zeros(self.buffer_size - len(chunk))
                    chunk = torch.cat([chunk, padding])
                
                # Extract embeddings
                embeddings = self.extract_embeddings(chunk)
                if embeddings is None:
                    continue
                
                print(f"Embeddings shape: {embeddings.shape}")
                
                # Apply LDA projection
                projected = self.apply_lda_projection(embeddings)
                if projected is None:
                    continue
                
                print(f"Projected shape: {projected.shape}")
                
                # Get X-axis values (first component)
                x_values = projected[:, 0]
                
                # Put the result in the queue for visualization update
                self.audio_queue.put(x_values)
                
                # Sleep to simulate real-time playback
                time.sleep(self.buffer_size / self.sample_rate)
                
        except Exception as e:
            print(f"Error in file processing: {e}")
    
    def update_waterfall(self, frame):
        """
        Update the waterfall plot with new data.
        """
        # Process any new audio results
        updated = False
        while not self.audio_queue.empty():
            try:
                x_values = self.audio_queue.get_nowait()
                
                # Print debug information
                print(f"Processing {len(x_values)} x-values, range: {np.min(x_values):.2f} to {np.max(x_values):.2f}")
                
                # Scroll the waterfall data down
                self.waterfall_data = np.roll(self.waterfall_data, 1, axis=0)
                
                # Clear the top row
                self.waterfall_data[0, :] = 0
                
                # Draw 2px dots for each x-value
                for x_val in x_values:
                    # Map x_val from [-10, 10] to [0, 799]
                    if -10 <= x_val <= 10:
                        pixel_x = int((x_val + 10) * (self.waterfall_width - 1) / 20)
                        # Ensure pixel_x is within bounds
                        pixel_x = max(0, min(self.waterfall_width - 1, pixel_x))
                        # Set the pixel and its neighbors to create a 2px dot
                        self.waterfall_data[0, pixel_x] = x_val
                        if pixel_x > 0:
                            self.waterfall_data[0, pixel_x - 1] = x_val
                        if pixel_x < self.waterfall_width - 1:
                            self.waterfall_data[0, pixel_x + 1] = x_val
                
                # Update the image data
                self.im.set_array(self.waterfall_data)
                updated = True
                
            except queue.Empty:
                break
        
        # Return the artists that were updated
        return [self.im] if updated else []
    
    def start(self):
        """
        Start the real-time audio analysis.
        """
        self.running = True
        
        # Start audio processing thread
        self.audio_thread.start()
        
        # Create animation
        ani = animation.FuncAnimation(self.fig, self.update_waterfall, interval=33, blit=True, save_count=100)  # ~30 FPS
        
        # Show plot
        plt.show()
        
        # Wait for the audio thread to finish
        self.audio_thread.join()
    
    def stop(self):
        """
        Stop the real-time audio analysis.
        """
        self.running = False


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
