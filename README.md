# Wav2Vec2 Audio Embedding Visualization

This script extracts embeddings from audio files using the Wav2Vec2 model and visualizes how audio embeddings evolve over time in a PCA or LDA space.

## Features

- Supports WAV and MP3 audio files
- Extracts frame-level embeddings using Facebook's Wav2Vec2 model
- Performs PCA or LDA to reduce embeddings to 2D space
- Visualizes each audio file as a colored trajectory line
- Shows how embeddings evolve through latent space over time
- LDA support for class-based analysis (subfolders A, B, C, etc.)
- Includes files from parent directory in LDA visualization (projected but not used for training)

## Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. The script will automatically download the Wav2Vec2 model on first run (requires internet connection)

## Usage

```bash
# For PCA visualization
python audio_embedding_pca.py path/to/audio/folder [--output output_plot.png] [--downstream layer]

# For LDA visualization (includes subfolders A, B, C, etc. as classes)
python generate_embeddings_data.py path/to/audio/folder --output embedding_data --lda [--downstream layer]

# For real-time LDA analysis with analog dial visualization
python realtime_lda_dial.py [--model embedding_data/lda_model.json] [--downstream layer]
```

### Example

```bash
# For PCA visualization
python audio_embedding_pca.py ./audio_samples --output my_plot.png

# For LDA visualization
python generate_embeddings_data.py ./audio_samples --output embedding_data --lda

# For downstream layer embeddings
python audio_embedding_pca.py ./audio_samples --output my_plot.png --downstream projected_states
python generate_embeddings_data.py ./audio_samples --output embedding_data --lda --downstream 6
```

## How It Works

1. **Audio Loading**: The script loads all WAV and MP3 files from the specified folder, automatically converting them to mono and resampling to 16kHz as required by Wav2Vec2.

2. **Embedding Extraction**: For each audio file, the script extracts frame-level embeddings using the Wav2Vec2 model. Each audio file produces a sequence of embeddings representing different time frames.

3. **Dimensionality Reduction**: All embeddings from all files are combined and reduced to 2D using either:
   - Principal Component Analysis (PCA) for general visualization
   - Linear Discriminant Analysis (LDA) for class-based analysis (when using subfolders A, B, C, etc.)

4. **Visualization**: Each audio file is plotted as a colored line showing how its embeddings traverse the 2D space over time. The trajectory shows the evolution of the audio through the latent space.

## Output

The script generates:
- A 2D plot showing embedding trajectories for all audio files
- A saved PNG image of the plot (default: `embedding_plot.png`)

## Requirements

- Python 3.7+
- PyTorch
- torchaudio
- transformers
- scikit-learn
- matplotlib
- numpy

## Model Information

The script uses the `facebook/wav2vec2-base` model by default. You can modify the `model_name` parameter in the `extract_embeddings` function to use other Wav2Vec2 variants.
