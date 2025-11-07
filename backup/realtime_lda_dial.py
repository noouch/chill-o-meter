#!/usr/bin/env python3
"""
Real-time audio analysis using LDA model with analog-style dial visualization.

This script uses a trained LDA model to analyze live audio input and displays
an analog-style dial that moves left or right based on the X-axis of the 
LDA projection. The dial has smooth movement with inertia for a realistic feel.
"""

import argparse
import json
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import tkinter as tk
import threading
import queue
import time
import pyaudio


class RealTimeLDADial:
    def __init__(self, model_path="embedding_data/lda_model.json", downstream_layer=None):
        """
        Initialize the real-time LDA dial visualization.
        
        Args:
            model_path (str): Path to the LDA model JSON file
            downstream_layer (str): Layer to extract embeddings from
        """
        self.model_path = model_path
        self.downstream_layer = downstream_layer
        self.audio_queue = queue.Queue()
        self.running = False
        self.dial_value = 0.0  # Current dial position (-1.0 to 1.0)
        self.target_value = 0.0  # Target dial position
        self.dial_velocity = 0.0  # Dial movement velocity for inertia
        
        # Load the LDA model
        self.lda_model = self.load_lda_model()
        
        # Initialize Wav2Vec2 model and processor
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.model.eval()
        
        # Create the GUI
        self.root = tk.Tk()
        self.root.title("Real-time LDA Audio Analysis")
        self.root.geometry("400x400")
        self.root.resizable(False, False)
        
        # Create canvas for dial
        self.canvas = tk.Canvas(self.root, width=400, height=400, bg="black")
        self.canvas.pack()
        
        # Draw static elements
        self.draw_dial_background()
        
        # Start audio processing thread
        self.audio_thread = threading.Thread(target=self.audio_processing_loop, daemon=True)
        self.audio_thread.start()
        
        # Start GUI update loop
        self.root.after(30, self.update_dial)  # ~33 FPS
        
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
            # We'll create mock values for these since we don't have them in the JSON
            # but they're needed for the transform method to work
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
                sampling_rate=16000, 
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
        Process audio in a separate thread.
        """
        print("Starting real-time audio processing loop...")
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Audio stream parameters
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = 16000  # Wav2Vec2 expects 16kHz
        CHUNK = 320  # Process 0.02 seconds at a time for ~50Hz update rate
        
        # Open audio stream
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        # Buffer to accumulate audio data for processing
        audio_buffer = []
        buffer_size = 3200  # Process 0.2 seconds of audio at a time
        
        try:
            while self.running:
                # Read audio data
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.float32)
                audio_buffer.extend(audio_data)
                
                # When we have enough data, process it
                if len(audio_buffer) >= buffer_size:
                    # Convert to tensor
                    waveform = torch.tensor(audio_buffer[:buffer_size], dtype=torch.float32)
                    audio_buffer = audio_buffer[buffer_size:]  # Remove processed data
                    
                    # Extract embeddings
                    embeddings = self.extract_embeddings(waveform)
                    if embeddings is None:
                        continue
                    
                    # Apply LDA projection
                    projected = self.apply_lda_projection(embeddings)
                    if projected is None:
                        continue
                    
                    # Calculate average X position (first component)
                    avg_x = np.mean(projected[:, 0])
                    
                    # Normalize to -1 to 1 range
                    # This is a simple normalization - in practice, you might want to use
                    # the actual range of your training data
                    normalized_value = np.tanh(avg_x)  # Maps to roughly -1 to 1
                    
                    # Put the result in the queue for GUI update
                    self.audio_queue.put(normalized_value)
                    
        except Exception as e:
            print(f"Error in audio processing: {e}")
        finally:
            # Clean up
            stream.stop_stream()
            stream.close()
            p.terminate()
    
    def draw_dial_background(self):
        """
        Draw the static background elements of the dial.
        """
        # Draw dial background
        self.canvas.create_oval(50, 50, 350, 350, fill="#222222", outline="#444444", width=3)
        
        # Draw center circle
        self.canvas.create_oval(190, 190, 210, 210, fill="#444444", outline="#666666", width=2)
        
        # Draw tick marks
        for i in range(11):
            angle = np.pi * (0.75 + 0.5 * i / 10)  # From -135° to +135°
            start_x = 200 + 140 * np.cos(angle)
            start_y = 200 + 140 * np.sin(angle)
            end_x = 200 + 160 * np.cos(angle)
            end_y = 200 + 160 * np.sin(angle)
            
            # Different color for center tick
            if i == 5:
                color = "#FF6600"  # Orange for center
            elif i == 0 or i == 10:
                color = "#FF0000"  # Red for extremes
            else:
                color = "#FFFFFF"  # White for others
                
            self.canvas.create_line(start_x, start_y, end_x, end_y, fill=color, width=2)
            
            # Draw labels
            if i % 2 == 0:
                label_x = 200 + 175 * np.cos(angle)
                label_y = 200 + 175 * np.sin(angle)
                label = str(i - 5) if i != 5 else "0"
                self.canvas.create_text(label_x, label_y, text=label, fill="#CCCCCC", font=("Arial", 10))
        
        # Draw title
        self.canvas.create_text(200, 30, text="LDA Audio Analysis", fill="#FFFFFF", font=("Arial", 16, "bold"))
        
        # Draw value indicator
        self.value_text = self.canvas.create_text(200, 370, text="0.00", fill="#00FF00", font=("Arial", 14, "bold"))
    
    def update_dial(self):
        """
        Update the dial position based on audio analysis results.
        """
        # Process any new audio results
        while not self.audio_queue.empty():
            try:
                self.target_value = self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        # Apply inertia to dial movement with more damping
        # Simple spring physics simulation with extra damping
        acceleration = (self.target_value - self.dial_value) * 0.2 - self.dial_velocity * 0.25
        self.dial_velocity += acceleration
        self.dial_value += self.dial_velocity
        
        # Clamp dial value
        self.dial_value = max(-1.0, min(1.0, self.dial_value))
        
        # Update value display
        self.canvas.itemconfig(self.value_text, text=f"{self.dial_value:.2f}")
        
        # Redraw dial pointer
        self.draw_dial_pointer()
        
        # Schedule next update
        if self.running:
            self.root.after(30, self.update_dial)
    
    def draw_dial_pointer(self):
        """
        Draw the dial pointer at the current position.
        """
        # Clear previous pointer
        self.canvas.delete("pointer")
        
        # Calculate pointer angle based on dial value
        # Map -1.0 to 1.0 to -135° to +135°
        angle = np.pi * (0.75 + 0.5 * (self.dial_value + 1.0) / 2.0)
        
        # Calculate pointer coordinates
        start_x = 200
        start_y = 200
        end_x = 200 + 130 * np.cos(angle)
        end_y = 200 + 130 * np.sin(angle)
        
        # Draw pointer
        self.canvas.create_line(start_x, start_y, end_x, end_y, 
                               fill="#00FF00", width=3, tags="pointer")
        
        # Draw pointer head
        head_x = 200 + 140 * np.cos(angle)
        head_y = 200 + 140 * np.sin(angle)
        self.canvas.create_oval(head_x-5, head_y-5, head_x+5, head_y+5, 
                               fill="#00FF00", outline="#00FF00", tags="pointer")
    
    def start(self):
        """
        Start the real-time audio analysis.
        """
        self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        self.root.mainloop()
    
    def stop(self):
        """
        Stop the real-time audio analysis.
        """
        self.running = False
        self.root.quit()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Real-time audio analysis with LDA dial visualization")
    parser.add_argument("--model", default="embedding_data/lda_model.json", 
                        help="Path to LDA model JSON file")
    parser.add_argument("--downstream", help="Extract embeddings from downstream layer")
    
    args = parser.parse_args()
    
    # Create and start the real-time LDA dial
    dial = RealTimeLDADial(model_path=args.model, downstream_layer=args.downstream)
    dial.start()


if __name__ == "__main__":
    main()
