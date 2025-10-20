import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.train.train import ConvAutoencoder  
import argparse
import os
from transformers import ViTMAEForPreTraining, AutoImageProcessor
import torch.nn.functional as F


def save_reconstructions(model_path, npy_path, autoencoder_model="cnn",
                         device="cpu",latent_dim=6 ,filename="models/reconstructions.png", num_images=8, freeze_until=-2):
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if autoencoder_model == "cnn":
    
        # Load data
        data = np.load(npy_path)  # shape: (N, H, W) or (N, H, W, C)
        data = data.astype(np.float32)
        data = data.reshape(-1, 1, 100, 100)  # (N, 1, H, W)
        
        data = data[:num_images]
        data = torch.tensor(data)

        # Create dataset and loader
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=num_images, shuffle=True)

        model = ConvAutoencoder(latent_dim=6).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        model.to(device)
        model.eval()

        # Take a batch
        inputs = next(iter(loader))[0].to(device)
        with torch.no_grad():
            outputs = model(inputs)

        # Plot originals and reconstructions
        fig, axes = plt.subplots(2, num_images, figsize=(num_images*2, 4))
        fig.suptitle(f"Original Images and Reconstructions ({autoencoder_model.upper()})", fontsize=16)
        
        for i in range(num_images):
            axes[0, i].imshow(inputs[i].cpu().squeeze(), cmap="gray")
            axes[1, i].imshow(outputs[i].cpu().squeeze(), cmap="gray")
            axes[0, i].axis("off")
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    
    if autoencoder_model == "mae":

        # Load data
        file_size = os.path.getsize(npy_path)
        N = file_size // (224 * 224 * 3 * 4)
        craters = np.memmap(
            npy_path,
            dtype=np.float32,
            mode="r",
            shape=(N, 224, 224, 3)
        )
        craters = craters.transpose(0, 3, 1, 2)  # NHWC -> NCHW
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(len(craters), size=num_images, replace=False)
        craters_subset = craters[sample_indices]

        dataset = TensorDataset(torch.from_numpy(craters_subset))
        loader = DataLoader(dataset, batch_size=num_images, shuffle=True)

        model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
        processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")

        # Freeze encoder except last N blocks
        for param in model.parameters():
            param.requires_grad = False
        for param in model.decoder.parameters():
            param.requires_grad = True
        for param in model.vit.encoder.layer[freeze_until:].parameters():
            param.requires_grad = True

        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Successfully loaded MAE model weights from {model_path}")
        except Exception as e:
            print(f"Error loading state_dict for MAE model: {e}")
            return

        model.to(device)
        model.eval()

        # Take a batch
        inputs = next(iter(loader))[0].to(device)


        with torch.no_grad():
            outputs = model(inputs, mask_ratio=0.75)
            logits = outputs.logits   # [B, num_patches, patch_dim]
            mask = outputs.mask       # [B, num_patches] boolean tensor

        patch_size = model.config.patch_size  # usually 16
        h = w = 224 // patch_size             # number of patches per dim

        # reconstruct images from patch logits
        recon = logits.reshape(-1, h, w, patch_size, patch_size, 3)
        recon = recon.permute(0, 5, 1, 3, 2, 4)   # [B, 3, h, ps, w, ps]
        recon = recon.reshape(-1, 3, 224, 224)    # [B, 3, H, W]

        # ---- PLOT ----
        fig, axes = plt.subplots(3, num_images, figsize=(num_images*2, 6))
        fig.suptitle(f"Original, Masked Input, and Reconstruction ({autoencoder_model.upper()})", fontsize=14)

        for i in range(num_images):
            orig_img = inputs[i].cpu().permute(1, 2, 0).numpy()  # HWC
            recon_img = recon[i].cpu().permute(1, 2, 0).clamp(0,1).numpy()

            # Build visible mask (True = visible)
            mask_bool = mask[i].bool().cpu().numpy().reshape(h, w)
            mask_img = np.kron(mask_bool, np.ones((patch_size, patch_size))).astype(bool)  # upscale to image size

            # Plot originals, overlay missing patches in red, reconstruction
            axes[0, i].imshow(orig_img)
            axes[0, i].axis("off")

            axes[1, i].imshow(orig_img)                      # original as background
            axes[1, i].imshow(~mask_img, cmap="Reds", alpha=0.3)  # overlay red where masked
            axes[1, i].axis("off")

            axes[2, i].imshow(recon_img)
            axes[2, i].axis("off")

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
    parser.add_argument('--freeze_until', type=int, default=2, help="For MAE: number of encoder transformer blocks to freeze from the end (negative number)") 
    args = parser.parse_args()


    save_reconstructions(
        model_path=args.model,
        npy_path=args.input,
        device=args.device,
        filename=args.file_outq,
        latent_dim=args.latent_dim,
        num_images=args.num_images,
        freeze_until=args.freeze_until,
        autoencoder_model=args.autoencoder_model
    )
