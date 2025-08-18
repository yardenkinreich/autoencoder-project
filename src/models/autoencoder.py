import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=6):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 100x100 -> 50x50
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 50x50 -> 25x25
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 25x25 -> 13x13
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64*13*13, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64*13*13),
            nn.ReLU(True),
            nn.Unflatten(1, (64, 13, 13)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # 13x13 -> 25x25
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # 25x25 -> 50x50
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # 50x50 -> 100x100
            nn.Sigmoid()  # normalize output 0-1
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
