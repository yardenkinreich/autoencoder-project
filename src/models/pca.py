import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load data
# Assumes shape is (n_samples, width, height) or (n_samples, n_features)
data = np.load('data/processed/craters.npy')
original_shape = data.shape[1:]  # e.g., (n_pixels, n_pixels)
n_samples = 10

# Flatten images if they aren't already
X = data.reshape(n_samples, -1)

# 2. Preprocessing
# PCA is sensitive to scale; standardizing is best practice
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA with 95% Variance Threshold
# Setting n_components to a float between 0 and 1 selects the 
# number of components that explain that % of variance.
pca = PCA(n_components=0.95, svd_solver='full')
X_pca = pca.fit_transform(X_scaled)

print(f"Original features: {X.shape[1]}")
print(f"Components retained for 95% variance: {pca.n_components_}")

# 4. Visualize Reconstruction of specific components (Eigen-craters)
def plot_pca_components(pca, original_shape, n_to_show=5):
    plt.figure(figsize=(15, 3))
    for i in range(n_to_show):
        plt.subplot(1, n_to_show, i + 1)
        # Reshape the component back to image dimensions
        component_img = pca.components_[i].reshape(original_shape)
        plt.imshow(component_img, cmap='gray')
        plt.title(f"PC {i+1}\nVar: {pca.explained_variance_ratio_[i]:.2%}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def compare_reconstruction(X_scaled, X_pca, pca, scaler, original_shape, n_examples=5):
    # Inverse transform: PCA space -> Scaled space -> Original Pixel space
    X_reconstructed_scaled = pca.inverse_transform(X_pca)
    X_reconstructed = scaler.inverse_transform(X_reconstructed_scaled)
    
    # Back to original pixel values for comparison
    X_original = scaler.inverse_transform(X_scaled)

    plt.figure(figsize=(12, 6))
    for i in range(n_examples):
        # Original Image
        plt.subplot(2, n_examples, i + 1)
        plt.imshow(X_original[i].reshape(original_shape), cmap='magma')
        plt.title(f"Original {i+1}")
        plt.axis('off')

        # Reconstructed Image (95% Variance)
        plt.subplot(2, n_examples, i + 1 + n_examples)
        plt.imshow(X_reconstructed[i].reshape(original_shape), cmap='magma')
        plt.title(f"95% Recon {i+1}")
        plt.axis('off')
        
    plt.suptitle("Comparison: Original vs. 95% Variance Reconstruction", fontsize=16)
    plt.tight_layout()
    plt.show()



plot_pca_components(pca, original_shape)

compare_reconstruction(X_scaled, X_pca, pca, scaler, original_shape)

