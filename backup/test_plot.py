import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def generate_test_data():
    """
    Generate test data to verify plotting functionality.
    """
    # Generate synthetic embedding data for 3 "audio files"
    test_data = []
    
    # File 1: Spiral pattern
    t1 = np.linspace(0, 4*np.pi, 50)
    x1 = t1 * np.cos(t1)
    y1 = t1 * np.sin(t1)
    embeddings1 = np.column_stack([x1, y1, np.random.normal(0, 0.1, 50)])  # Add 3rd dimension with noise
    test_data.append((embeddings1, "spiral_audio.wav"))
    
    # File 2: Straight line
    t2 = np.linspace(0, 10, 30)
    x2 = t2
    y2 = 2 * t2 + np.random.normal(0, 0.2, 30)  # Add some noise
    embeddings2 = np.column_stack([x2, y2, np.random.normal(0, 0.1, 30)])  # Add 3rd dimension with noise
    test_data.append((embeddings2, "line_audio.mp3"))
    
    # File 3: Circle
    t3 = np.linspace(0, 2*np.pi, 40)
    x3 = 5 * np.cos(t3) + np.random.normal(0, 0.1, 40)  # Add noise
    y3 = 5 * np.sin(t3) + np.random.normal(0, 0.1, 40)  # Add noise
    embeddings3 = np.column_stack([x3, y3, np.random.normal(0, 0.1, 40)])  # Add 3rd dimension with noise
    test_data.append((embeddings3, "circle_audio.wav"))
    
    return test_data

def perform_pca(embeddings_list):
    """
    Perform PCA on test data to reduce to 2D.
    """
    # Combine all embeddings for PCA
    all_features = []
    filenames = []
    
    for embeddings, filename in embeddings_list:
        all_features.append(embeddings)
        filenames.append(filename)
    
    # Flatten all embeddings
    all_features = np.vstack(all_features)
    
    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_features)
    
    # Split back into per-file embeddings
    result = []
    start_idx = 0
    
    for embeddings, filename in embeddings_list:
        end_idx = start_idx + len(embeddings)
        file_embeddings_2d = embeddings_2d[start_idx:end_idx]
        result.append((file_embeddings_2d, filename))
        start_idx = end_idx
    
    return result

def plot_embeddings(embeddings_2d_list, output_path="test_plot.png"):
    """
    Plot embeddings as colored lines in 2D space.
    """
    plt.figure(figsize=(10, 8))
    
    # Plot each file as a separate colored line
    for embeddings_2d, filename in embeddings_2d_list:
        plt.plot(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1], 
            marker='o', 
            markersize=4,
            linewidth=2,
            label=filename
        )
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Test Embeddings Trajectories in PCA Space')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save and show plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Test plot saved to {output_path}")

def main():
    """
    Test the plotting functionality with synthetic data.
    """
    print("Generating test data...")
    test_data = generate_test_data()
    
    print("Performing PCA on test data...")
    test_data_2d = perform_pca(test_data)
    
    print("Creating test plot...")
    plot_embeddings(test_data_2d, "test_plot.png")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    main()
