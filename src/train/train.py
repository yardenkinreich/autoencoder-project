import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import matplotlib.pyplot as plt
from src.models.autoencoder import ConvAutoencoder

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load data
    craters = np.load(args.input)
    craters = craters.astype(np.float32)
    craters = craters.reshape(-1, 1, 100, 100) # Reshape from flattened to 1x100x100
    dataset = TensorDataset(torch.from_numpy(craters))

    # Train/val split
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model, loss, optimizer
    model = ConvAutoencoder(latent_dim=args.latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    train_losses = []
    val_losses = []

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

        print(f"Epoch [{epoch+1}/{args.epochs}] - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

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
    model.eval()
    with torch.no_grad():
        all_data = torch.tensor(craters).to(device)
        latent_vectors = model.encode(all_data).cpu().numpy()
    os.makedirs(os.path.dirname(args.latent_output), exist_ok=True)
    np.save(args.latent_output, latent_vectors)
    print(f"Saved latent vectors to {args.latent_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help="Path to crater npy file")
    parser.add_argument('--model_output', required=True, help="Path to save trained model")
    parser.add_argument('--loss_plot', required=True, help="Path to save loss curve plot")
    parser.add_argument('--latent_output', required=True, help="Path to save latent vectors")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val_split', type=float, default=0.2)
    args = parser.parse_args()
    main(args)