import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.train.train import ConvAutoencoder  
import argparse

def save_reconstructions(model_path, npy_path, device="cpu", filename="models/reconstructions.png", num_images=8):
    # Load data
    data = np.load(npy_path)  # shape: (N, 100, 100)
    data = data.astype(np.float32)
    data = data.reshape(-1, 1, 100, 100)  # reshape to (N, 1, 100, 100)
    data = data[:num_images]
    data = torch.tensor(data)
    
    # Create dataset and loader
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=num_images, shuffle=True)

    # Load model
    model = ConvAutoencoder(latent_dim=args.latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Take a batch
    inputs = next(iter(loader))[0]
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(inputs)

    # Plot originals and reconstructions
    fig, axes = plt.subplots(2, num_images, figsize=(num_images*2, 4))
    fig.suptitle("Original Images and Reconstructions", fontsize=16)
    for i in range(num_images):
        axes[0, i].imshow(inputs[i].cpu().squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(outputs[i].cpu().squeeze(), cmap="gray")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help="Path to crater npy file")
    parser.add_argument('--model', required=True, help="Path to trained model")
    parser.add_argument('--device', default="cpu", help="Device to run the model on")
    parser.add_argument('--latent_dim', type=int, default=6, help="Dimensionality of the latent space")
    parser.add_argument('--file_outq', default="models/reconstructions.png", help="Path to save the reconstruction image")
    parser.add_argument('--num_images', type=int, default=8, help="Number of images to reconstruct and display")
    args = parser.parse_args()


    save_reconstructions(
        model_path=args.model,
        npy_path=args.input,
        device=args.device,
        filename=args.file_outq,
        num_images=args.num_images
    )
