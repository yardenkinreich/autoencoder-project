import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=6):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),   # 100 -> 50
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 50 -> 25
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(32 * 25 * 25, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 25 * 25),
            nn.ReLU(True),
            nn.Unflatten(1, (32, 25, 25)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # 25 -> 50
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # 50 -> 100
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
