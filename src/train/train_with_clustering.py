import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt
from src.models.autoencoder import ConvAutoencoder
import timm
import torch.optim as optim
from src.helper_functions import *
import torchvision.transforms.v2 as T
import sys
sys.path.append(os.path.abspath("src/models/mae"))
from src.models.mae.models_mae import *


# ============ SwAV Components ============
class ClusterHead(nn.Module):
    """Small MLP to predict cluster assignments from CLS token"""
    def __init__(self, input_dim=1024, hidden_dim=2048, num_clusters=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_clusters)
        )
    
    def forward(self, x):
        return self.mlp(x)


@torch.no_grad()
def sinkhorn(scores, eps=0.05, niters=3):
    """
    Sinkhorn-Knopp algorithm for optimal transport.
    Converts cluster scores into balanced soft cluster assignments.
    
    This prevents collapse (all samples assigned to one cluster) by:
    1. Ensuring each sample is assigned to exactly one cluster (row normalization)
    2. Ensuring clusters are balanced (column normalization)
    
    Args:
        scores: [batch_size, num_clusters] raw cluster predictions
        eps: temperature for softmax
        niters: number of iterations for convergence
    
    Returns:
        Q: [batch_size, num_clusters] soft cluster assignments (pseudo-labels)
    """
    Q = torch.exp(scores / eps).t()  # [num_clusters, batch_size]
    Q /= Q.sum()  # normalize so sum = 1
    
    K, B = Q.shape  # num_clusters, batch_size
    
    for _ in range(niters):
        # Make each row sum to 1/K (balance across samples)
        Q /= Q.sum(dim=0, keepdim=True)
        Q /= K
        
        # Make each column sum to 1/B (balance across clusters)
        Q /= Q.sum(dim=1, keepdim=True)
        Q /= B
    
    return (Q * K).t()  # [batch_size, num_clusters]


def main(args):

    # --- Reproducibility ---
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.autoencoder_model == 'cae':
        # [cae code unchanged - keeping your original]
        craters = np.load(args.input)
        craters = craters.astype(np.float32)
        craters = craters.reshape(-1, 1, 100, 100)
        num_samples = args.num_samples if args.num_samples is not None else len(craters)
        num_samples = min(num_samples, len(craters))
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(len(craters), size=num_samples, replace=False)
        craters_subset = craters[sample_indices]

        dataset = TensorDataset(torch.from_numpy(craters_subset))

        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        model = ConvAutoencoder(latent_dim=args.latent_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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

        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # Load MAE model
        model = mae_vit_base_patch16()
        checkpoint = torch.load("src/models/mae/mae_finetuned_vit_large.pth", map_location="cpu")
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Loaded pretrained MAE weights: {msg}")

        for name, param in model.named_parameters():
            param.requires_grad = True

        model.to(device)

        # ============ ADD CLUSTERING HEAD ============
        cluster_head = ClusterHead(
            input_dim=1024,  # ViT-Large hidden dimension
            hidden_dim=2048,
            num_clusters=args.num_clusters
        ).to(device)
        
        # Initialize with small weights to prevent early collapse
        cluster_head.mlp[-1].weight.data.normal_(0, 0.01)
        cluster_head.mlp[-1].bias.data.zero_()

        # Data augmentations (IMPORTANT for SwAV!)
        train_transforms = T.Compose([
            T.RandomVerticalFlip(p=0.5),
            T.RandomErasing(p=0.2, scale=(0.02, 0.06), ratio=(0.5, 2.0)),
        ])

        # Optimizer now includes cluster head parameters
        optimizer = optim.AdamW(
            list(filter(lambda p: p.requires_grad, model.parameters())) + 
            list(cluster_head.parameters()),
            lr=args.lr, 
            weight_decay=args.weight_decay
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr
        )

        train_losses, val_losses = [], []
        train_recon_losses, train_cluster_losses = [], []

        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        print(f"Mask ratio: {args.mask_ratio}")
        print(f"Cluster weight: {args.cluster_weight}")
        print(f"Number of clusters: {args.num_clusters}")

        for epoch in range(args.epochs):
            
            running_train_loss = 0.0
            running_recon_loss = 0.0
            running_cluster_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                imgs = batch[0].to(device)
                

                if torch.isnan(imgs).any():
                        print(f"WARNING: NaN in input data at batch {batch_idx}")
                        continue
                
                # Create augmented view
                imgs_aug = train_transforms(imgs)
            
                if torch.isnan(imgs_aug).any():
                    print(f"WARNING: NaN after augmentation at batch {batch_idx}")
                    continue
                
                # ============ RECONSTRUCTION LOSS (with masking) ============
                loss_recon, _, _ = model(imgs, mask_ratio=args.mask_ratio)

                if torch.isnan(loss_recon):
                    print(f"NaN in reconstruction loss at batch {batch_idx}")
                    print(f"Input range: [{imgs.min():.4f}, {imgs.max():.4f}]")
                    print(f"Input mean: {imgs.mean():.4f}, std: {imgs.std():.4f}")
                    continue
                
                # ============ CLUSTERING LOSS (SwAV) ============
                # Get CLS tokens from FULL images (no masking for clustering!)
                with torch.no_grad():
                    latent1 = model.forward_encoder(imgs, mask_ratio=0.0)[0][:, 0, :]  # [batch, 1024]
                
                # Predict cluster assignments for original view
                cluster_scores1 = cluster_head(latent1)  # [batch, num_clusters]
                
                # Convert to soft pseudo-labels via Sinkhorn
                cluster_targets = sinkhorn(cluster_scores1)  # [batch, num_clusters]
                
                # Get features from augmented view
                latent2 = model.forward_encoder(imgs_aug, mask_ratio=0.0)[0][:, 0, :]
                cluster_scores2 = cluster_head(latent2)
                
                # SwAV loss: predict cluster assignment of view1 from features of view2
                loss_cluster = F.cross_entropy(cluster_scores2, cluster_targets)
                
                # ============ COMBINED LOSS ============
                loss = loss_recon + args.cluster_weight * loss_cluster
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_train_loss += loss.item() * imgs.size(0)
                running_recon_loss += loss_recon.item() * imgs.size(0)
                running_cluster_loss += loss_cluster.item() * imgs.size(0)
            
            epoch_train_loss = running_train_loss / len(train_loader.dataset)
            epoch_recon_loss = running_recon_loss / len(train_loader.dataset)
            epoch_cluster_loss = running_cluster_loss / len(train_loader.dataset)
            
            train_losses.append(epoch_train_loss)
            train_recon_losses.append(epoch_recon_loss)
            train_cluster_losses.append(epoch_cluster_loss)

            # Validation loop (reconstruction only)
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch[0].to(device)
                    val_loss, _, _ = model(imgs, mask_ratio=args.mask_ratio)
                    running_val_loss += val_loss.item() * imgs.size(0)

                epoch_val_loss = running_val_loss / len(val_loader.dataset)
                val_losses.append(epoch_val_loss)
            
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Total: {epoch_train_loss:.4f} | "
                  f"Recon: {epoch_recon_loss:.4f} | "
                  f"Cluster: {epoch_cluster_loss:.4f} | "
                  f"Val: {epoch_val_loss:.4f} | "
                  f"LR: {current_lr:.2e}")

        # Save model and cluster head
        os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
        torch.save({
            'model': model.state_dict(),
            'cluster_head': cluster_head.state_dict()
        }, args.model_output)

        # Plot losses
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Total Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Total Training and Validation Loss")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_recon_losses, label="Reconstruction")
        plt.plot(train_cluster_losses, label="Clustering")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Components")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(args.loss_plot)
        plt.close()

        # Extract latent vectors (NO MASKING for clustering evaluation!)
        latent_list = []
        with torch.no_grad():
            model.eval()
            for i in range(0, len(craters_subset), args.batch_size):
                batch = torch.tensor(craters_subset[i:i+args.batch_size], device=device)
                
                # CRITICAL: Extract with mask_ratio=0.0 for clustering!
                latent, _, _ = model.forward_encoder(batch, mask_ratio=0.0)
                cls_token = latent[:, 0, :]  # Just the CLS token
                
                latent_list.append(cls_token.cpu().numpy())

        latent_vectors = np.concatenate(latent_list, axis=0)
        np.save(args.latent_output, latent_vectors)
        print(f"Saved latent vectors of shape {latent_vectors.shape}")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--autoencoder_model', type=str, choices=['cae', 'mae'], default='cae')
    parser.add_argument('--input', required=True)
    parser.add_argument('--model_output', required=True)
    parser.add_argument('--loss_plot', required=True)
    parser.add_argument('--latent_output', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--lr_patience', type=int, default=5)
    parser.add_argument('--min_lr', type=float, default=1e-8)
    parser.add_argument('--lr_factor', type=float, default=0.5)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--freeze_until', type=int, default=-2)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    
    # NEW: SwAV parameters
    parser.add_argument('--num_clusters', type=int, default=10, 
                       help="Number of clusters for SwAV")
    parser.add_argument('--cluster_weight', type=float, default=0.5, 
                       help="Weight for clustering loss (0=off, 1.0=equal to reconstruction)")
    
    args = parser.parse_args()
    main(args)