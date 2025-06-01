import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time


# Function to generate random word embeddings for demonstration
def generate_sample_embeddings(n_samples=50, embedding_dim=50):  # Reduced embedding dimension
    """Generate random word embeddings for demonstration purposes"""
    print("Generating embeddings...")
    np.random.seed(42)  # For reproducibility
    embeddings = np.random.normal(0, 1, (n_samples, embedding_dim))
    # Normalize embeddings to unit length
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # Generate some sample words
    sample_words = [f"word_{i}" for i in range(n_samples)]

    return embeddings, sample_words


# Function to reduce dimensions for visualization with timeout
def reduce_dimensions(embeddings, method='pca', n_components=3):
    """Reduce embedding dimensions to 3D for visualization"""
    print(f"Reducing dimensions using {method}...")

    if method == 'pca':
        # PCA is much faster and more reliable
        model = PCA(n_components=n_components)
        reduced_embeddings = model.fit_transform(embeddings)
    else:
        # Default to PCA for speed and reliability
        print("Using PCA instead for better performance.")
        model = PCA(n_components=n_components)
        reduced_embeddings = model.fit_transform(embeddings)

    return reduced_embeddings


# Function to generate RGB colors directly based on 3D positions
def generate_direct_rgb_colors(vectors_3d):
    """Generate RGB colors directly from the 3D positions"""
    print("Generating colors...")

    # Normalize to [0,1] range for RGB values
    min_vals = np.min(vectors_3d, axis=0)
    max_vals = np.max(vectors_3d, axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Avoid division by zero

    normalized = (vectors_3d - min_vals) / range_vals

    # Convert to RGB strings
    colors = []
    for vec in normalized:
        r, g, b = vec[0], vec[1], vec[2]
        colors.append(f'rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})')

    return colors


# Create an interactive 3D plot
def plot_vectors_3d(vectors, labels, colors, method='pca'):
    """Create an interactive 3D plot of word vectors"""
    print("Creating visualization...")

    # Create figure
    fig = go.Figure()

    # Create the scatter plot
    fig.add_trace(go.Scatter3d(
        x=vectors[:, 0],
        y=vectors[:, 1],
        z=vectors[:, 2],
        mode='markers',  # Removed text mode for better performance
        marker=dict(
            size=10,
            color=colors,
            opacity=0.8,
            line=dict(width=0.5, color='white')
        ),
        hoverinfo='text',
        hovertext=[
            f"Word: {label}<br>R: {int(int(colors[i].split('(')[1].split(',')[0]) / 2.55)}%<br>G: {int(int(colors[i].split(',')[1].strip()) / 2.55)}%<br>B: {int(int(colors[i].split(',')[2].split(')')[0].strip()) / 2.55)}%"
            for i, label in enumerate(labels)]
    ))

    # Update layout
    fig.update_layout(
        title=f"3D Color Vector Visualization ({method.upper()})",
        scene=dict(
            xaxis=dict(title="R (Red)", color="red", gridcolor="lightpink"),
            yaxis=dict(title="G (Green)", color="green", gridcolor="lightgreen"),
            zaxis=dict(title="B (Blue)", color="blue", gridcolor="lightblue"),
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig


# Main function with optimized parameters
def main(n_samples=50):
    start_time = time.time()

    # Generate sample data
    embeddings, words = generate_sample_embeddings(n_samples=n_samples, embedding_dim=50)

    # Reduce dimensions to 3D using PCA (much faster than t-SNE)
    vectors_3d = reduce_dimensions(embeddings, method='pca', n_components=3)

    # Generate colors directly from 3D positions
    colors = generate_direct_rgb_colors(vectors_3d)

    # Create plot
    fig = plot_vectors_3d(vectors_3d, words, colors, method='pca')

    # Save as HTML file for interactive viewing
    html_file = "colorful_vectors_3d.html"
    fig.write_html(html_file)

    end_time = time.time()
    print(f"Visualization saved as '{html_file}' (completed in {end_time - start_time:.2f} seconds)")

    # Show the plot
    fig.show()

    return fig


# Run the main function with optimized parameters
if __name__ == "__main__":
    main(n_samples=50)  # Reduced number of samples for better performance