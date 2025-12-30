import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.train.train import ConvAutoencoder  
import argparse
import os
from transformers import ViTMAEForPreTraining
import torch.nn.functional as F
from src.helper_functions import *
import sys
sys.path.append(os.path.abspath("src/models/mae"))
from src.models.mae.models_mae import *


def save_reconstructions(model_path, npy_path, autoencoder_model="cae",
                         device="cpu",latent_dim=6 , filename="models/reconstructions.png", num_images=8, freeze_until=-2, pretrained_model='facebook/vit-mae-large', mask_ratio=0.75):
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if autoencoder_model == "cae":
        # Load data
        file_size = os.path.getsize(npy_path)
        N = file_size // (224 * 224 * 1 * 4)
        craters = np.memmap(
            npy_path,
            dtype=np.float32,
            mode="r",
            shape=(N, 1, 224, 224)
        )

        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(len(craters), size=num_images, replace=False)
        craters_subset = craters[sample_indices]

        dataset = TensorDataset(torch.from_numpy(craters_subset))
        loader = DataLoader(dataset, batch_size=num_images, shuffle=True)

        model = ConvAutoencoder(latent_dim=latent_dim).to(device)
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
            shape=(N, 3, 224, 224)
        )

        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(len(craters), size=num_images, replace=False)
        craters_subset = craters[sample_indices]

        dataset = TensorDataset(torch.from_numpy(craters_subset))
        loader = DataLoader(dataset, batch_size=num_images, shuffle=True)

        # Load pretrained MAE
        model = mae_vit_base_patch16()

        try:
            state_dict = torch.load(model_path, map_location="cpu")
            msg = model.load_state_dict(state_dict)
            print(f"Successfully loaded MAE model weights from {model_path}")
            print(f"Loaded pretrained MAE weights: {msg}")

        except Exception as e:
            print(f"Error loading state_dict for MAE model: {e}")
            return

        model.to(device) 
        model.eval()

        # Take a batch
        inputs = next(iter(loader))[0].to(device)

        with torch.no_grad():
            torch.manual_seed(42)
            _ , pred, mask = model(inputs, mask_ratio)

        print("pred shape:", pred.shape)
        print("mask shape:", mask.shape)
        print("patch size:", model.patch_embed.patch_size)

        # reconstruct images from patch logits

        recon = model.unpatchify(pred)

        # De-normalize
        mean = torch.tensor([0.27261323, 0.27261323, 0.27261323], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.0973839, 0.0973839, 0.0973839], device=device).view(1, 3, 1, 1)

        recon = recon * std + mean
        inputs = inputs * std + mean

        # Clamp
        recon = recon.clamp(0, 1)
        inputs = inputs.clamp(0, 1)

        # Create mask image
        mask_img = unpatchify_mask(mask.cpu().numpy())  # (B, 1, H, W)

        # DEBUG: Check mask_img values
        print(f"mask_img shape: {mask_img.shape}")
        print(f"mask_img unique: {np.unique(mask_img)}")
        print(f"mask_img mean: {mask_img.mean()}")

        masked_recon = recon.clone()
        for i in range(len(masked_recon)):
            # mask_img[i] is already a tensor, shape (1, H, W)
            mask_single = mask_img[i]  # (1, 224, 224)
            
            # Repeat to 3 channels: (3, 224, 224)
            mask_3ch = mask_single.repeat(3, 1, 1)
            
            # Where mask==0 (visible), set to gray (0.5)
            masked_recon[i] = torch.where(
                mask_3ch > 0.5,
                masked_recon[i],  # Keep reconstruction for masked patches
                torch.full_like(masked_recon[i], 0.5)  # Gray out visible patches
            )

        fig, axes = plt.subplots(5, num_images, figsize=(num_images*2, 10))  # 5 rows now
        fig.suptitle("MAE Reconstruction Analysis", fontsize=14)

        for i in range(num_images):
            orig_img = inputs[i].cpu().permute(1, 2, 0).numpy()[:, :, 0]
            recon_img = recon[i].cpu().permute(1, 2, 0).numpy()[:, :, 0]
            masked_recon_img = masked_recon[i].cpu().permute(1, 2, 0).numpy()[:, :, 0]
            
            # Row 0: Original
            axes[0, i].imshow(orig_img, cmap="gray")
            axes[0, i].set_title("Original", fontsize=9)
            axes[0, i].axis("off")

            # Row 1: Full Reconstruction
            axes[1, i].imshow(recon_img, cmap="gray")
            axes[1, i].set_title("Full Recon", fontsize=9)
            axes[1, i].axis("off")

            # Row 2: ONLY Masked Reconstructions (visible = gray)
            axes[2, i].imshow(masked_recon_img, cmap="gray")
            axes[2, i].set_title("Masked Only", fontsize=9)
            axes[2, i].axis("off")

            # Row 3: Mask overlay
            axes[3, i].imshow(orig_img, cmap="gray")
            axes[3, i].imshow(mask_img[i, 0].cpu().numpy(), cmap="Reds", alpha=0.3)
            axes[3, i].set_title("Masked Patches", fontsize=9)
            axes[3, i].axis("off")

            # Row 4: Composite
            mask_img_bw = mask_img[i, 0].cpu().numpy()
            composite_img = np.where(mask_img_bw > 0.5, recon_img, orig_img)
            axes[4, i].imshow(composite_img, cmap="gray")
            axes[4, i].set_title("Composite", fontsize=9)
            axes[4, i].axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filename)
        plt.close()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--autoencoder_model', type=str, choices=['cae','mae'], default='cae')
    parser.add_argument('--input', required=True, help="Path to crater npy file")
    parser.add_argument('--model', required=True, help="Path to trained model")
    parser.add_argument('--device', default="cpu", help="Device to run the model on")
    parser.add_argument('--latent_dim', type=int, default=6, help="Dimensionality of the latent space")
    parser.add_argument('--file_outq', default="models/reconstructions.png", help="Path to save the reconstruction image")
    parser.add_argument('--num_images', type=int, default=8, help="Number of images to reconstruct and display")
    parser.add_argument('--freeze_until', type=int, default=2, help="For MAE: number of encoder transformer blocks to freeze from the end (negative number)") 
    parser.add_argument('--pretrained_model', type=str, default='facebook/vit-mae-large', help="Pretrained model name for MAE")
    parser.add_argument('--mask_ratio', type=float, default=0.75, help="Masking ratio for MAE reconstruction")  
    args = parser.parse_args()


    save_reconstructions(
        model_path=args.model,
        npy_path=args.input,
        device=args.device,
        filename=args.file_outq,
        latent_dim=args.latent_dim,
        num_images=args.num_images,
        freeze_until=args.freeze_until,
        autoencoder_model=args.autoencoder_model,
        pretrained_model=args.pretrained_model,
        mask_ratio=args.mask_ratio
    )
