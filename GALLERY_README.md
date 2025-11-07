# Wav2Vec2 Embedding Gallery

This is an interactive web application that allows you to visualize how audio embeddings evolve over time in the PCA space while playing back the audio files.

## Features

- Interactive visualization of audio embeddings as scatter plots
- Play audio files and watch the embedding evolution in real-time
- All other audio files shown as grayed-out scatter points for context
- Current position highlighted with a red marker
- Progress bar showing playback position
- High-resolution rendering (4x) for better visualization
- Support for both PCA and LDA dimensionality reduction

## How It Works

1. **Data Preparation**: The `generate_embeddings_data.py` script extracts wav2vec2 embeddings from audio files and performs either PCA or LDA to reduce them to 2D. The results are saved as JSON files.

2. **Web Interface**: The `embedding_gallery.html` file provides a web-based interface with:
   - File selection dropdown
   - Audio player controls
   - Interactive 2D visualization canvas
   - Progress indicator

3. **Real-time Visualization**: When you play an audio file:
   - The embeddings of the current file are shown as green scatter points
   - All other files are shown as grayed-out scatter points
   - A red marker shows the current position in the embedding space
   - The marker moves through the scatter points as the audio plays
   - When using LDA, files from subfolders are grouped by class (A, B, C, etc.) and files from the parent directory are included in the visualization but were not used for training the LDA model

## Setup and Usage

1. **Generate Embedding Data** (if not already done):
```bash
# For PCA (default)
python generate_embeddings_data.py example_clips --output embedding_data [--downstream layer]
   
# For LDA (includes subfolders A, B, C, etc. as classes)
python generate_embeddings_data.py example_clips --output embedding_data --lda [--downstream layer]
```

2. **Start the Gallery Server**:
   ```bash
   python start_gallery.py
   ```

3. **Using the Gallery**:
   - The browser will automatically open to the gallery page
   - Select an audio file from the dropdown
   - Use the play button to start playback
   - Watch the red marker move through the green scatter points as the audio plays
   - All other files are shown as grayed-out scatter points for context
   - When using LDA, files are labeled with their class (subfolder name)

## Technical Details

- **Frontend**: Pure HTML, CSS, and JavaScript (no frameworks)
- **Visualization**: Canvas-based rendering for smooth animations
- **Data Loading**: Asynchronous JSON loading for embedding data
- **Audio Synchronization**: Frame-by-frame synchronization with audio playback

## Customization

You can customize the gallery by modifying:
- `embedding_gallery.html`: Layout and styling
- Colors and visualization parameters in the JavaScript code
- Server settings in `start_gallery.py` (port number, etc.)

## Troubleshooting

- If the gallery doesn't load, make sure the HTTP server is running
- If audio files don't play, check that they're in a supported format (WAV/MP3)
- If embeddings don't display, verify that the embedding data was generated correctly
