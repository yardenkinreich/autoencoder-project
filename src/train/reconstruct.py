import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.train.train import ConvAutoencoder  
import argparse

def save_reconstructions(model_path, npy_path, autoencoder_model="cnn",
                         device="cpu", filename="models/reconstructions.png", num_images=8):
    import timm
    
    # Load data
    data = np.load(npy_path)  # shape: (N, H, W) or (N, H, W, C)
    data = data.astype(np.float32)
    
    if autoencoder_model == "cnn":
        data = data.reshape(-1, 1, 100, 100)  # (N, 1, H, W)
    elif autoencoder_model == "mae":
        # Ensure 3 channels for MAE
        if data.ndim == 3:
            data = np.stack([data]*3, axis=-1)
        data = data.reshape(-1, 3, 224, 224)  # NCHW for ViT
    
    data = data[:num_images]
    data = torch.tensor(data)

    # Create dataset and loader
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=num_images, shuffle=True)

    # Load model
    if autoencoder_model == "cnn":
        model = ConvAutoencoder(latent_dim=6).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    elif autoencoder_model == "mae":
        model = timm.create_model("mae_vit_base_patch16_dec512d8b", pretrained=False)
        # Adjust for grayscale if needed
        if data.shape[1] == 1:
            model.patch_embed.proj = nn.Conv2d(1, model.patch_embed.proj.out_channels,
                                               kernel_size=16, stride=16)
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Take a batch
    inputs = next(iter(loader))[0].to(device)
    with torch.no_grad():
        if autoencoder_model == "cnn":
            outputs = model(inputs)
        elif autoencoder_model == "mae":
            _, outputs, _ = model(inputs)  # MAE forward returns loss, pred, mask

    # Plot originals and reconstructions
    fig, axes = plt.subplots(2, num_images, figsize=(num_images*2, 4))
    fig.suptitle(f"Original Images and Reconstructions ({autoencoder_model.upper()})", fontsize=16)
    
    for i in range(num_images):
        if autoencoder_model == "mae":
            # MAE outputs are [B, C, H, W] with 3 channels
            orig_img = inputs[i].cpu().permute(1,2,0)  # CHW -> HWC
            recon_img = outputs[i].cpu().permute(1,2,0)
            axes[0, i].imshow(orig_img.squeeze(), cmap="gray" if orig_img.shape[2]==1 else None)
            axes[1, i].imshow(recon_img.squeeze(), cmap="gray" if recon_img.shape[2]==1 else None)
        else:
            axes[0, i].imshow(inputs[i].cpu().squeeze(), cmap="gray")
            axes[1, i].imshow(outputs[i].cpu().squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--autoencoder_model', type=str, choices=['cnn','mae'], default='cnn')
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
