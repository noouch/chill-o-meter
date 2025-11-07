@echo off
echo Wav2Vec2 Audio Embedding PCA Visualization
echo ========================================
echo This script will analyze audio files in the example_clips folder
echo and generate a 2D visualization of their embeddings.
echo.

REM Check if example_clips folder exists
if not exist "example_clips" (
    echo Error: example_clips folder not found!
    echo Please place your audio files (wav/mp3) in the example_clips folder
    echo and run this script again.
    echo.
    pause
    exit /b
)

REM Check if output file already exists
if exist "embedding_analysis.png" (
    echo Warning: embedding_analysis.png already exists and will be overwritten.
    echo.
)

echo Processing audio files...
echo.

python audio_embedding_pca.py example_clips --output embedding_analysis.png

echo.
echo Analysis complete!
echo Results saved to embedding_analysis.png
echo.
pause
