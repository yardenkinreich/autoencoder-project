"""
dino_craters_dataset.py
────────────────────────
DINOv2 dataset class for crater crops, reading directly from the
preprocessed memmap (.dat) — the same format src/train/train.py and
reconstruct.py already use for MAE/CAE. No PNG/bytes round-trip: every
existing decoder in the vendored dinov2 fork (decoders.py) expects encoded
image bytes and the default one hardcodes .convert(mode="RGB"), which would
force a lossy uint8 requantization and a fake RGB conversion of our
carefully globally-scaled float32 single-channel data. This overrides
__getitem__ directly instead, skipping that decoder machinery entirely.

Registered in the dinov2 dataset string registry (loaders.py) as "Craters".
"""

import os
from typing import Any, Tuple

import numpy as np
import torch

from src.models.dinov2.dinov2.data.datasets.extended import ExtendedVisionDataset


class Craters(ExtendedVisionDataset):
    """
    root: path to a craters_*.dat memmap file (e.g. the DINO-specific
    wide-FOV, single-branch export — see preprocess_2.py --clean_offset).
    """

    def __init__(self, root, num_channels: int = 1, transform=None,
                target_transform=None, transforms=None):
        super().__init__(root=root, transforms=transforms,
                         transform=transform, target_transform=target_transform)
        self._num_channels = num_channels
        self._data = self._load_memmap(root, num_channels)

    @staticmethod
    def _load_memmap(path: str, num_channels: int) -> np.memmap:
        """Infer N from file size, mirroring src/train/train.py's load_memmap."""
        file_size = os.path.getsize(path)
        total_floats = file_size // 4          # float32 = 4 bytes
        size = 128
        n_pixels = num_channels * size * size
        if total_floats % n_pixels != 0:
            raise ValueError(
                f"Cannot infer crater memmap shape from {path}: "
                f"{total_floats} floats with {num_channels} channel(s), "
                f"expected a multiple of {n_pixels} (128x128)."
            )
        n = total_floats // n_pixels
        return np.memmap(path, dtype=np.float32, mode="r",
                         shape=(n, num_channels, size, size))

    def get_image_data(self, index: int) -> torch.Tensor:
        return torch.from_numpy(np.array(self._data[index]))   # (C,H,W) float32, copied out of the memmap

    def get_target(self, index: int) -> Any:
        return -1   # self-supervised: no labels

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = self.get_image_data(index)
        target = self.get_target(index)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self) -> int:
        return len(self._data)
