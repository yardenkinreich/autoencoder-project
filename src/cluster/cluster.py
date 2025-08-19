import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from torchvision import transforms
from src.models.autoencoder import ConvAutoencoder
from src.helper_functions import *

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

def load_images(imgs_dir):
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])
    files = sorted([f for f in os.listdir(imgs_dir) if f.endswith(".png")])
    states = np.array([int(f.split("_")[1].split(".")[0]) for f in files])

    imgs = torch.stack([
        transform(Image.fromarray(
            flip_crater(np.array(Image.open(os.path.join(imgs_dir, f)).convert("L")))
        ))
        for f in files
    ]).float()

    return imgs, states, files  # (N,1,H,W)


def encode_images(inputs, model_path, bottleneck, device, out_latents, out_states, states):
    model = ConvAutoencoder(latent_dim=bottleneck)
    model.load_state_dict(torch.load(model_path, map_location=device))
    encoder = model.encoder.to(device).eval()

    inputs = inputs.to(device)
    with torch.no_grad():
        latents = encoder(inputs).cpu().numpy()

    np.save(out_latents, latents)
    np.save(out_states, states)
    print(f"Saved latents to {out_latents}, states to {out_states}")


def plot_dots(latents_path, states_path, out_png, technique, model_name):
    latents = np.load(latents_path)
    states = np.load(states_path)

    if technique == "pca":
        coords = PCA(n_components=2).fit_transform(latents)
    elif technique == "tsne":
        coords = TSNE(n_components=2).fit_transform(latents)
    else:
        raise ValueError(f"Unknown technique {technique}")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title(f"{technique} on {model_name} Latent Clustering with Known Labels")
    ax.set_xlabel(f"{technique} Component 1")
    ax.set_ylabel(f"{technique} Component 2")

    for s in np.unique(states):
        mask = states == s
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   label=STATE_LABELS.get(s, f"state {s}"),
                   c=STATE_COLORS.get(s, "gray"),
                   alpha=0.7)
    ax.legend()
    plt.savefig(out_png, dpi=200)
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
    enc.add_argument("--bottleneck", type=int, default=6)
    enc.add_argument("--out-latents", required=True)
    enc.add_argument("--out-states", required=True)

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
                      args.out_latents, args.out_states, states)
    elif args.command == "plot-dots":
        plot_dots(args.latents, args.states, args.out_png, args.technique, args.model_name)
    elif args.command == "plot-imgs":
        plot_imgs(args.latents, args.imgs_dir, args.out_png, args.technique, args.model_name)
    else:
        parser.print_help()
