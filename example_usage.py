"""
Example usage of the audio_embedding_pca.py script
"""

import os
import subprocess
import sys

def main():
    # Define the audio folder and output file
    audio_folder = "example_clips"
    output_file = "example_analysis.png"
    
    # Check if the audio folder exists
    if not os.path.exists(audio_folder):
        print(f"Error: {audio_folder} folder not found!")
        print("Please create the folder and add some audio files (wav/mp3) to it.")
        return
    
    # Check if there are any audio files in the folder
    audio_files = [f for f in os.listdir(audio_folder) 
                   if f.lower().endswith(('.wav', '.mp3'))]
    
    if not audio_files:
        print(f"No audio files found in {audio_folder}!")
        print("Please add some audio files (wav/mp3) to the folder.")
        return
    
    print(f"Found {len(audio_files)} audio files in {audio_folder}")
    print("Files:", audio_files[:5], "..." if len(audio_files) > 5 else "")
    
    # Run the main script
    print(f"\nRunning audio embedding PCA analysis...")
    print(f"Output will be saved to {output_file}")
    
    try:
        # Run the script as a subprocess
        result = subprocess.run([
            sys.executable, 
            "audio_embedding_pca.py", 
            audio_folder, 
            "--output", 
            output_file
        ], check=True, capture_output=True, text=True)
        
        print("Analysis completed successfully!")
        print(f"Results saved to {output_file}")
        
        # Print any output from the script
        if result.stdout:
            print("\nScript output:")
            print(result.stdout)
            
    except subprocess.CalledProcessError as e:
        print(f"Error running the analysis script: {e}")
        if e.stderr:
            print("Error details:")
            print(e.stderr)
    except FileNotFoundError:
        print("Error: audio_embedding_pca.py not found!")
        print("Please make sure you're running this script from the correct directory.")

if __name__ == "__main__":
    main()
