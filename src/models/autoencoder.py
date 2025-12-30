import torch
import torch.nn as nn

import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 224 -> 112
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 112 -> 56
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 56 -> 28
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),# 28 -> 14
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(128*14*14, latent_dim)           # 128*14*14=25088
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128*14*14),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 14, 14)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 14 -> 28
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   # 28 -> 56
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),   # 56 -> 112
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),    # 112 -> 224
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

