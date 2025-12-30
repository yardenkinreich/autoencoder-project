import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from src.models.autoencoder import ConvAutoencoder
from src.helper_functions import *
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import sys
sys.path.append(os.path.abspath("src/models/mae"))
from src.models.mae.models_mae import *
#from src.models.mae import MAEWithBottleneck

STATE_LABELS = {
    1: "New Crater",
    2: "Semi New Crater",
    3: "Semi Old Crater",
    4: "Old Crater"
}
STATE_COLORS = {
    1: "tab:blue",
    2: "tab:green",
    3: "tab:orange",
    4: "tab:red"
}

def normalize_image_to_crater_stats(img_path):
    """
    Normalize crater images using the statistics from your training data
    """
    # Load grayscale image
    img = Image.open(img_path).convert("L")

    img = img.resize((224, 224))

    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Replicate to 3 channels
    img_3ch = np.stack([img_array, img_array, img_array], axis=0)  # (3, 224, 224)
    
    # Normalize with crater dataset statistics
    mean = np.array([0.27261323, 0.27261323, 0.27261323]).reshape(3, 1, 1)
    std = np.array([0.0973839, 0.0973839, 0.0973839]).reshape(3, 1, 1)
    
    img_normalized = (img_3ch - mean) / std
    
    return torch.from_numpy(img_normalized)

def load_images(imgs_dir):
    files = sorted([f for f in os.listdir(imgs_dir) if f.endswith(".png")])
    states = np.array([int(f.split("_")[1].split(".")[0]) for f in files])

    if args.autoencoder_model == 'cae':
        imgs_list = []
        for f in files:
            img_path = os.path.join(imgs_dir, f)
            
            # Load grayscale image
            img = Image.open(img_path).convert("L")
            img = img.resize((224, 224))
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Apply flip_crater
            img_array = flip_crater(img_array).copy()

            # Convert to tensor and add channel dimension
            imgs_list.append(torch.from_numpy(img_array).unsqueeze(0))  # shape (1, 224, 224)
                
        # Stack all images into a single tensor
        imgs = torch.stack(imgs_list).float()  # shape (N, 1, 224, 224)

    elif args.autoencoder_model == 'mae':
        imgs_list = []
        for f in files:
            img_path = os.path.join(imgs_dir, f)
            
            # Load grayscale image
            img = Image.open(img_path).convert("L")
            img = img.resize((224, 224))
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Apply flip_crater
            img_array = flip_crater(img_array)
            
            # Replicate to 3 channels
            img_3ch = np.stack([img_array, img_array, img_array], axis=0)  # (3, 224, 224)
            '''
            # Normalize with crater dataset statistics
            mean = np.array([0.27261323, 0.27261323, 0.27261323]).reshape(3, 1, 1)
            std = np.array([0.0973839, 0.0973839, 0.0973839]).reshape(3, 1, 1)
            
            img_3ch = (img_3ch - mean) / std
            '''
            imgs_list.append(torch.from_numpy(img_3ch))
                
        imgs = torch.stack(imgs_list).float()
    
    return imgs, states, files  # Returns (N, C, H, W) Tensor



def encode_images(inputs, model_path, bottleneck, device, out_latents, 
out_states, states = None,mask_ratio=0.75, freeze_until=-2,autoencoder_model="mae", pretrained_model='facebook/vit-mae-large',is_dataloader=False):

    if autoencoder_model == 'cae':
        model = ConvAutoencoder(latent_dim=bottleneck)
        model.load_state_dict(torch.load(model_path, map_location=device))
        encoder = model.encoder.to(device).eval()

        inputs = inputs.to(device)
        with torch.no_grad():
            latents = encoder(inputs).cpu().numpy()

        np.save(out_latents, latents)
        np.save(out_states, states)
        print(f"Saved latents to {out_latents}, states to {out_states}")
    
    elif autoencoder_model == 'mae':
        # Load pretrained MAE
        model = mae_vit_base_patch16()

        try:
            state_dict = torch.load(model_path, map_location="cpu")
            msg = model.load_state_dict(state_dict, strict=False) # strict=False is often safer for MAE fine-tuning
            print(f"Successfully loaded MAE model weights from {model_path}")

        except Exception as e:
            print(f"Error loading state_dict for MAE model: {e}")
            return

        model.to(device) # Move model to GPU
        model.eval()
        
        if is_dataloader:
            latents_list = []
            with torch.no_grad():
                for i, batch in enumerate(inputs):
                    if i % 100 == 0:
                        print(f"Processing batch {i}/{len(inputs)}")
                    
                    x = batch.to(device)
                    z, _, _ = model.forward_encoder(x, mask_ratio=0)
                    latents_list.append(z[:, 0, :].cpu())
            
            latents = torch.cat(latents_list, dim=0).numpy()
        else:
            # Single tensor path
            inputs = inputs.to(device)
            with torch.no_grad():
                z, _, _ = model.forward_encoder(inputs, mask_ratio=0)
                latents = z[:, 0, :].cpu().numpy()

        np.save(out_latents, latents)
        if states is not None:
            np.save(out_states, states)
        print(f"Saved latents to {out_latents}")



def plot_dots(latents_path, states_path, out_png, technique, model_name):
    latents = np.load(latents_path)
    states = np.load(states_path).squeeze()

    if technique == "pca":
        pca = PCA(n_components=2)
        coords = pca.fit_transform(latents)

        x_axis, y_axis = coords[:, 0], coords[:, 1]

        explained_var = pca.explained_variance_ratio_
        x_label = f"PC1 ({explained_var[0]*100:.1f}% variance)"
        y_label = f"PC2 ({explained_var[1]*100:.1f}% variance)"

        total_var = explained_var.sum() * 100
        print(f"[PCA] Total variance explained (PC1+PC2): {total_var:.2f}%")

        fig, ax = plt.subplots(figsize=(10, 7))

    elif technique == "tsne":
        coords = TSNE(n_components=2).fit_transform(latents)
        x_axis, y_axis = coords[:, 0], coords[:, 1]

        x_label = "t-SNE Component 1"
        y_label = "t-SNE Component 2"

        fig, ax = plt.subplots(figsize=(10, 7))

    else:
        raise ValueError(f"Unknown technique {technique}")

    ax.set_title(
        f"{technique.upper()} Projection of Latent Space "
        f"Colored by Ground-Truth Deformation State"
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    for s in np.unique(states):
        mask = states == s
        ax.scatter(
            x_axis[mask],
            y_axis[mask],
            label=STATE_LABELS.get(s, f"state {s}"),
            c=STATE_COLORS.get(s, "gray"),
            alpha=0.7,
            s=20,
        )

    ax.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    print(f"Saving plot to {out_png}")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_imgs(latents_path, imgs_dir, out_png, technique, model_name):
    latents = np.load(latents_path)

    if technique == "pca":
        coords = PCA(n_components=2).fit_transform(latents)
    elif technique == "tsne":
        coords = TSNE(n_components=2).fit_transform(latents)

    files = sorted([f for f in os.listdir(imgs_dir) if f.endswith(".png")])

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(f"{technique} on {model_name} Latent Image Clustering")
    ax.set_xlabel(f"{technique} Component 1")
    ax.set_ylabel(f"{technique} Component 2")

    for (x, y), fname in zip(coords, files):
        img = Image.open(os.path.join(imgs_dir, fname)).convert("L")
        im = OffsetImage(img, zoom=0.2, cmap="gray")
        ab = AnnotationBbox(im, (x, y), frameon=False)
        ax.add_artist(ab)
    ax.update_datalim(coords)
    ax.autoscale()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # encode
    enc = subparsers.add_parser("encode")
    enc.add_argument("--imgs-dir", required=True)
    enc.add_argument("--model", required=True)
    enc.add_argument("--autoencoder-model", choices=["cae", "mae"], default="cae", required=True)
    enc.add_argument("--bottleneck", type=int, default=6)
    enc.add_argument("--out-latents", required=True)
    enc.add_argument("--out-states", required=True)
    enc.add_argument("--freeze-until", type=int, default=-2)
    enc.add_argument("--pretrained-model", type=str, default='facebook/vit-mae-large')
    enc.add_argument("--mask-ratio", type=float, default=0.75)

    # plot dots
    dots = subparsers.add_parser("plot-dots")
    dots.add_argument("--latents", required=True)
    dots.add_argument("--states", required=True)
    dots.add_argument("--out-png", required=True)
    dots.add_argument("--technique", choices=["pca", "tsne"], default="pca")
    dots.add_argument("--model-name", required=True)

    # plot imgs
    imgs = subparsers.add_parser("plot-imgs")
    imgs.add_argument("--latents", required=True)
    imgs.add_argument("--imgs-dir", required=True)
    imgs.add_argument("--out-png", required=True)
    imgs.add_argument("--technique", choices=["pca", "tsne"], default="pca")
    imgs.add_argument("--model-name", required=True)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.command == "encode":
        inputs, states, _ = load_images(args.imgs_dir)
        encode_images(inputs, args.model, args.bottleneck, device,
                      args.out_latents, args.out_states, states,args.mask_ratio, args.freeze_until, args.autoencoder_model, args.pretrained_model)
    elif args.command == "plot-dots":
        plot_dots(args.latents, args.states, args.out_png, args.technique, args.model_name)
    elif args.command == "plot-imgs":
        plot_imgs(args.latents, args.imgs_dir, args.out_png, args.technique, args.model_name)
    else:
        parser.print_help()

