import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt
from src.models.autoencoder import ConvAutoencoder
import timm
#from transformers import ViTMAEForPreTraining
import torch.optim as optim
from src.helper_functions import *
import torchvision.transforms as T
import sys
sys.path.append(os.path.abspath("src/models/mae"))
from src.models.mae.models_mae import *

def main(args):

    # --- Reproducibility ---
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.autoencoder_model == 'cnn':
        # Load data
        craters = np.load(args.input)
        craters = craters.astype(np.float32)
        craters = craters.reshape(-1, 1, 100, 100) # Reshape from flattened to 1x100x100
        num_samples = args.num_samples if args.num_samples is not None else len(craters)
        num_samples = min(num_samples, len(craters))
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(len(craters), size=num_samples, replace=False)
        craters_subset = craters[sample_indices]

        dataset = TensorDataset(torch.from_numpy(craters_subset))

        # --- Train/val split (deterministic) ---
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # Model, loss, optimizer
        model = ConvAutoencoder(latent_dim=args.latent_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Training loop
        train_losses = []
        val_losses = []
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                inputs = batch[0].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_train_loss = running_loss / train_size
            train_losses.append(epoch_train_loss)

            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch[0].to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)
                    running_val_loss += loss.item() * inputs.size(0)
            epoch_val_loss = running_val_loss / val_size
            val_losses.append(epoch_val_loss)

            scheduler.step()
            current_lr = optimizer.get_lr()

            print(f"Epoch [{epoch+1}/{args.epochs}] - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}", 
                f"LR: {current_lr:.2e}")

        # Save model and loss plot
        os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
        torch.save(model.state_dict(), args.model_output)

        plt.figure(figsize=(8,5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.savefig(args.loss_plot)
        plt.close()

        # Save latent vectors for clustering
        latent_list = []

        model.eval()
        with torch.no_grad():
            for i in range(0, len(craters), args.batch_size):
                batch = torch.from_numpy(craters[i:i+args.batch_size]).to(device)
                latent_batch = model.encode(batch).cpu().numpy()
                latent_list.append(latent_batch)

        os.makedirs(os.path.dirname(args.latent_output), exist_ok=True)
        latent_vectors = np.concatenate(latent_list, axis=0)
        np.save(args.latent_output, latent_vectors)


    elif args.autoencoder_model == 'mae':
        # Load data
        file_size = os.path.getsize(args.input)
        N = file_size // (224 * 224 * 3 * 4)
        craters = arr = np.memmap(
            args.input,
            dtype=np.float32,
            mode="r",
            shape=(N, 3, 224, 224)
            )
        num_samples = args.num_samples if args.num_samples is not None else len(craters)
        num_samples = min(num_samples, len(craters))
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(len(craters), size=num_samples, replace=False)
        craters_subset = craters[sample_indices]

        print(f"Data range: min={craters_subset.min()}, max={craters_subset.max()}, mean={craters_subset.mean()}")

        dataset = TensorDataset(torch.from_numpy(craters_subset))

        # --- Train/val split (deterministic) ---
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # Load pretrained MAE
        model = mae_vit_large_patch16()
        # Load pretrained weights
        checkpoint = torch.load("src/models/mae/mae_finetuned_vit_large.pth", map_location="cpu")

        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Loaded pretrained MAE weights: {msg}")

        for name, param in model.named_parameters():
            param.requires_grad = False  # freeze everything

        # Unfreeze only the last few encoder layers
        for blk in model.blocks[args.freeze_until:]:
            for param in blk.parameters():
                param.requires_grad = True

        # Optionally unfreeze the decoder (if you want to fine-tune reconstruction)
        for param in model.decoder_blocks.parameters():
            param.requires_grad = True

        model.to(device)

        # data_training augmentations
        train_transforms = T.Compose([
            T.RandomVerticalFlip(p=0.5),
        
            # This "wiggles" the image: rotates, scales, and shifts it a little.
            T.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        
            T.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        ])

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr, weight_decay=args.weight_decay)

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max = args.epochs,   # total number of epochs
            eta_min = args.min_lr       # final LR
            )

        train_losses, val_losses = [], []

        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        print(f"Mask ratio: {args.mask_ratio}")


        for epoch in range(args.epochs):
            model.train()
            running_train_loss = 0.0
            for batch in train_loader:
                imgs = batch[0].to(device)
                #inputs = {"pixel_values": imgs}
                #imgs = train_transforms(imgs)

                loss, _, _ = model(imgs, mask_ratio=args.mask_ratio)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item() * imgs.size(0)
            epoch_train_loss = running_train_loss / len(train_loader.dataset)
            train_losses.append(epoch_train_loss)

            # --- validation loop ---
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch[0].to(device)
                    #inputs = {"pixel_values": imgs}
                    
                    val_loss, _, _ = model(imgs, mask_ratio=args.mask_ratio)


                    running_val_loss += val_loss.item() * imgs.size(0)

                epoch_val_loss = running_val_loss / len(val_loader.dataset)
                val_losses.append(epoch_val_loss)
            
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}/{args.epochs} | "
                    f"Train Loss: {epoch_train_loss:.4f} | "
                    f"Val Loss: {epoch_val_loss:.4f} | "
                    f"LR: {current_lr:.2e}")

        # Save model and loss plot
        os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
        torch.save(model.state_dict(), args.model_output)

        plt.figure(figsize=(8,5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")

        # Annotate the last values
        final_train = train_losses[-1]
        final_val = val_losses[-1]
        plt.text(len(train_losses)-1, final_train, f"{final_train:.4f}", 
                color="blue", ha="right", va="bottom", fontsize=9)
        plt.text(len(val_losses)-1, final_val, f"{final_val:.4f}", 
                color="orange", ha="right", va="bottom", fontsize=9)

        plt.xlabel("Epoch")
        plt.ylabel("Reconstruction Loss (MSE)")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.savefig(args.loss_plot)
        plt.close()

        latent_list = []

        with torch.no_grad():
            model.eval()
            for i in range(0, len(craters_subset), args.batch_size):
                batch = torch.tensor(craters_subset[i:i+args.batch_size], device=device)
                #inputs = {"pixel_values": batch}

                # Forward encoder (with masking)
                latent, mask, ids_restore = model.forward_encoder(batch, mask_ratio=args.mask_ratio)

                latent_list.append(latent.cpu().numpy())

        # Concatenate all
        latent_vectors = np.concatenate(latent_list, axis=0)
        np.save(args.latent_output, latent_vectors)
        print(f"Saved latent vectors of shape {latent_vectors.shape}")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--autoencoder_model', type=str, choices=['cnn', 'mae'], default='cnn', help="Type of autoencoder model to use")
    parser.add_argument('--input', required=True, help="Path to crater npy file")
    parser.add_argument('--model_output', required=True, help="Path to save trained model")
    parser.add_argument('--loss_plot', required=True, help="Path to save loss curve plot")
    parser.add_argument('--latent_output', required=True, help="Path to save latent vectors")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--lr_patience', type=int, default=5, help="Epochs to wait before reducing LR")
    parser.add_argument('--min_lr', type=float, default=1e-8, help="Minimum learning rate")
    parser.add_argument('--lr_factor', type=float, default=0.5, help="Factor to reduce LR by")
    parser.add_argument('--num_samples', type=int, default=None, help="Number of craters to sample for training")
    parser.add_argument('--freeze_until', type=int, default=-2, help="For MAE: number of encoder transformer blocks to freeze from the end (negative number)") 
    parser.add_argument('--mask_ratio', type=float, default=0.75, help="Masking ratio for MAE training")
#    parser.add_argument('--pretrained_model', type=str, default='facebook/vit-mae-large', help="Pretrained model name for MAE")
    args = parser.parse_args()
    main(args) 