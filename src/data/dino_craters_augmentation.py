"""
dino_craters_augmentation.py
──────────────────────────────
DINOv2 multi-crop augmentation for craters — analogous to CellAugmentationDINO
(dinov2/data/cell_dino/augmentations.py), but for our single-channel,
already globally-scaled [0,1] float32 data instead of raw multi-channel
fluorescence microscopy.

Neither of the fork's existing augmentation classes fit as-is:
  • DataAugmentationDINO (stock): ColorJitter(saturation=..., hue=...)
    requires 3-channel RGB and errors on true grayscale; RandomSolarize
    assumes an 8-bit [0,255] range, not our [0,1] floats.
  • CellAugmentationDINO: Div255() would re-divide data that's already
    scaled to [0,1], and its protein-channel transforms are
    microscopy-specific (nothing to select/drop on 1 channel).

So this reuses the channel/range-agnostic pieces of CellAugmentationDINO's
transform library (RandomBrightness, RandomContrast — tensor-based, loop
over channels, work fine at 1 channel) and drops the rest.

Deliberately skips SelfNormalizeNoDiv() (which CellAugmentationDINO does
use): a per-crop mean/std normalization would wash out the absolute-contrast
cue that — per preprocess_2.py's own normalize_global() docstring —
distinguishes fresh (high local contrast) from degraded (washed-out)
craters. Same reasoning this project already used to move away from
per-sample min-max normalization for MAE.
"""

from torchvision import transforms

from src.models.dinov2.dinov2.data.transforms import GaussianBlur, MaybeToTensor
from src.models.dinov2.dinov2.data.cell_dino.transforms import RandomBrightness, RandomContrast


class CraterAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=128,
        local_crops_size=64,
        in_chans=1,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        # our .dat files are always 1-channel on disk; in_chans>1 (e.g. 3, to
        # match a stock ImageNet-pretrained checkpoint we're warm-starting
        # from) replicates the single channel rather than requiring a
        # separate re-preprocessed 3-channel dataset. RandomResizedCrop
        # below already resizes to global/local_crops_size regardless of the
        # raw 128px input, so no separate resize step is needed either.
        self.in_chans = in_chans

        def geometric(size, scale):
            return transforms.Compose([
                MaybeToTensor(),   # no-op — get_image_data() already returns a tensor
                transforms.RandomResizedCrop(
                    size, scale=scale, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),   # craters have no preferred orientation
            ])

        self.geometric_global = geometric(global_crops_size, global_crops_scale)
        self.geometric_local  = geometric(local_crops_size, local_crops_scale)

        # Mild photometric jitter + blur. No color jitter (nothing to jitter
        # on a single grayscale channel), no solarize (assumes 0-255 range).
        self.photometric = transforms.Compose([
            RandomBrightness(p=0.2),
            RandomContrast(p=0.2),
            GaussianBlur(p=0.3, radius_min=0.1, radius_max=1.0),
        ])

    def __call__(self, image):
        if self.in_chans > 1 and image.shape[0] == 1:
            image = image.repeat(self.in_chans, 1, 1)

        output = {}

        global_crop_1 = self.photometric(self.geometric_global(image))
        global_crop_2 = self.photometric(self.geometric_global(image))
        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        output["local_crops"] = [
            self.photometric(self.geometric_local(image))
            for _ in range(self.local_crops_number)
        ]
        output["offsets"] = ()
        return output
