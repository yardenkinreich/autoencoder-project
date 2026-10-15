"""
dino_craters_augmentation.py
──────────────────────────────
DINOv2 multi-crop augmentation for craters — analogous to CellAugmentationDINO
(dinov2/data/cell_dino/augmentations.py), but for our single-channel,
already globally-scaled [0,1] float32 data instead of raw multi-channel
fluorescence microscopy.

Neither of the fork's existing augmentation classes fit as-is:
  • DataAugmentationDINO (stock): ColorJitter(saturation=..., hue=...)
    requires 3-channel RGB and errors on true grayscale.
  • CellAugmentationDINO: Div255() would re-divide data that's already
    scaled to [0,1], and its protein-channel transforms are
    microscopy-specific (nothing to select/drop on 1 channel).

So this reuses the channel/range-agnostic pieces of CellAugmentationDINO's
transform library (RandomBrightness, RandomContrast — tensor-based, loop
over channels, work fine at 1 channel) as the grayscale-appropriate stand-in
for ColorJitter's brightness/contrast (saturation/hue genuinely don't apply -
no color channels to have a saturation/hue).

RandomSolarize, unlike ColorJitter, does NOT actually require an 8-bit
[0,255] range - torchvision's implementation inverts using whatever scale
the threshold is given in, so it works on our [0,1] floats directly once
the threshold (stock's 128/255) is rescaled to 128/255 ≈ 0.502. An earlier
version of this module dropped it on the mistaken assumption that it was
color/range-specific like ColorJitter - confirmed working on grayscale data
by e.g. Stillwagon et al.'s DINOCell (continued DINOv2 pretraining on
grayscale microscopy, keeping solarize unmodified). This turned out to
matter: without it (and without stock's per-crop-slot blur/solarize
asymmetry below), a from-stock DINO finetune measurably collapsed onto
brightness as its dominant learned feature (near-perfect correlation with
PC1, see docs/dino_mae_proto_finetune_status.md) - brightness is a
crop-invariant statistic by construction, so with weak/uniform photometric
jitter it's a free shortcut DINO's self-distillation objective has every
incentive to exploit instead of learning real crater structure.

Deliberately skips SelfNormalizeNoDiv() (which CellAugmentationDINO does
use): a per-crop mean/std normalization would wash out the absolute-contrast
cue that — per preprocess_2.py's own normalize_global() docstring —
distinguishes fresh (high local contrast) from degraded (washed-out)
craters. Same reasoning this project already used to move away from
per-sample min-max normalization for MAE.
"""

# Bumped whenever this module's augmentation recipe changes materially -
# there's no config toggle for old-vs-new (the fix lives in code, not a
# YAML flag), so this is the durable marker that lets a run_manifest.json /
# the dashboard tell which augmentation recipe actually produced a given
# checkpoint, independent of directory naming. "v1" = the original
# symmetric brightness/contrast-only recipe (no solarize, flat blur
# probability); "v2" = this file's current stock-DINOv2-style asymmetric
# recipe (solarize on global crop 2, per-slot blur probabilities).
AUGMENTATION_RECIPE_VERSION = "v4_rangefree_photometric"
# v4: brightness/contrast/solarize no longer assume [0,1] pixels. v3 used
# cell_dino's RandomBrightness/RandomContrast, i.e. torchvision
# adjust_brightness/adjust_contrast, which CLAMP float images to [0,1] - on
# the robust/asinh-normalized Kaguya catalogs (values centred on 0) that
# clipped ~64-71% of every crop's pixels (all shadow -> 0, all bright -> 1),
# on every view of every Kaguya finetune up to 2026-10-14. [0,1]-scaled WAC
# data was unaffected apart from the clip at 1.

# GLOBAL-CROP GEOMETRY (crops.global_crops_mode)
# ---------------------------------------------
# "random"       stock DINO: both global crops are RandomResizedCrop over
#                crops.global_crops_scale - the two teacher views are two
#                random, mostly-overlapping windows.
# "context_zoom" the two teacher views are HARD-WIRED to two fixed scales:
#                  global crop 1 = the whole image (the crater in its full
#                                  context, image_fov_diameters wide)
#                  global crop 2 = a centre crop exactly zoom_diameters wide
#                                  (1.0 = rim to rim: the crater itself)
#                so the global/global DINO term asks context <-> crater to
#                agree, instead of two near-identical random windows. Only
#                flips vary; the photometric per-slot asymmetry is unchanged.
#                Assumes the crater is centred, which every crop_crater /
#                build_catalog crop is (by construction, +- registration).
GLOBAL_CROPS_MODES = ("random", "context_zoom")


def recipe_version(global_crops_mode: str = "random") -> str:
    """The recipe marker a run manifest records: the photometric recipe plus,
    when not stock-random, the global-crop geometry."""
    if global_crops_mode == "random":
        return AUGMENTATION_RECIPE_VERSION
    return f"{AUGMENTATION_RECIPE_VERSION}+{global_crops_mode}"


import torch
from torchvision import transforms
from torchvision.transforms import functional as TF

from src.models.dinov2.dinov2.data.transforms import GaussianBlur, MaybeToTensor
import numpy as np


class RangeFreeBrightness(torch.nn.Module):
    """Brightness jitter for data on ANY scale: an additive shift of
    (factor-1) x the crop's own std, factor ~ max(N(1, sd), 0.5) - the same
    factor distribution as v3's cell_dino RandomBrightness - and one factor for
    all channels (replicated grayscale must not get a colour tint). Never clamps."""

    def __init__(self, sd: float = 0.2):
        super().__init__()
        self.sd = sd

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        f = max(float(np.random.normal(1, self.sd)), 0.5)
        return img + (f - 1.0) * img.std()


class RangeFreeContrast(torch.nn.Module):
    """Contrast jitter about the crop's own mean: (x - mean) * factor + mean,
    same factor distribution as v3, one factor for all channels. Never clamps."""

    def __init__(self, sd: float = 0.2):
        super().__init__()
        self.sd = sd

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        f = max(float(np.random.normal(1, self.sd)), 0.5)
        m = img.mean()
        return (img - m) * f + m


class RangeFreeSolarize(torch.nn.Module):
    """Solarize relative to the crop's own range: with probability p, pixels
    above the midpoint of [min, max] are mirrored (lo + hi - x) - stock DINO's
    threshold 128/255 on [0,1] data, without assuming that range."""

    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = p

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if np.random.rand() >= self.p:
            return img
        lo, hi = img.min(), img.max()
        return torch.where(img >= (lo + hi) / 2, lo + hi - img, img)


class CraterAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=128,
        local_crops_size=64,
        in_chans=1,
        global_crops_mode="random",
        image_fov_diameters=2.0,
        zoom_diameters=1.0,
    ):
        if global_crops_mode not in GLOBAL_CROPS_MODES:
            raise ValueError(f"global_crops_mode={global_crops_mode!r}, expected one of {GLOBAL_CROPS_MODES}")
        if global_crops_mode == "context_zoom" and not 0 < zoom_diameters <= image_fov_diameters:
            raise ValueError(f"zoom_diameters={zoom_diameters} must be in (0, image_fov_diameters="
                             f"{image_fov_diameters}] - the zoom has to fit inside the image")
        self.global_crops_mode = global_crops_mode
        # Linear fraction of the image side the zoom view keeps: the image is
        # image_fov_diameters crater diameters wide (2.0 for the "wide"
        # offset-1.0 data, WAC dino_wide and the Kaguya catalogs alike), so
        # a 1-diameter zoom is the central half.
        self.zoom_fraction = zoom_diameters / image_fov_diameters
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

        # context_zoom: fixed windows, only flips random. Resize is
        # antialiased - native-resolution catalog crops are often several
        # times larger than global_crops_size, and an aliased downsample
        # would put spurious texture into exactly the views the teacher sees.
        flips = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])
        resize = transforms.Resize((global_crops_size, global_crops_size),
                                   interpolation=transforms.InterpolationMode.BICUBIC,
                                   antialias=True)
        self.context_view = transforms.Compose([MaybeToTensor(), resize, flips])
        self.zoom_view = transforms.Compose([MaybeToTensor(), self._center_zoom, resize, flips])

        # Base photometric jitter - grayscale-appropriate stand-in for stock
        # DINO's ColorJitter(brightness/contrast/saturation/hue) (see module
        # docstring). Applied to every crop, same as stock's color_jittering
        # step.
        base_photometric = transforms.Compose([
            RangeFreeBrightness(0.2),
            RangeFreeContrast(0.2),
        ])

        # Per-crop-slot EXTRA transforms, matching stock DataAugmentationDINO's
        # own asymmetric structure exactly (dinov2/data/augmentations.py):
        # global crop 1 is ALWAYS blurred and NEVER solarized; global crop 2
        # is RARELY blurred but SOMETIMES solarized instead; local crops are
        # blurred half the time. This asymmetry - not just "add solarize" on
        # its own - is what actually makes DINO's two global/teacher views
        # look meaningfully different from each other, which is the real
        # mechanism that stops brightness/contrast from being a trivial,
        # free-to-match invariant between them. A flat, identical probability
        # for every crop (what this module used before) loses that asymmetry
        # even with solarize added back in.
        #
        # GaussianBlur's own p ARGUMENT IS INVERTED in this environment -
        # measured directly (see src/train/diet_cp.py's build_augmentation()
        # docstring for the full derivation): GaussianBlur(p=X) actually
        # blurs with probability (1-X), not X, because its `keep_p = 1 - p`
        # compensation (dinov2/data/transforms.py) was written for a
        # torchvision RandomApply behavior that this project's installed
        # torchvision does not have. AUGMENTATION_RECIPE_VERSION was bumped
        # to v3 because of this fix - the v2 recipe that job ep5_augv2
        # actually trained under had global_transfo1 NEVER blurred (not
        # ALWAYS) and global_transfo2 blurred ~90% of the time (not rarely),
        # the reverse of what's documented immediately above and what was
        # reported when that experiment "didn't fix" the brightness
        # collapse - that conclusion was drawn against the wrong recipe and
        # needs re-running under this corrected version before it can be
        # trusted. Not touching dinov2/data/transforms.py itself (DINOv2
        # stays unmodified); compensating at each call site instead, with
        # the intended probability stated explicitly.
        solarize_threshold = 128 / 255  # stock's 128 (8-bit), rescaled to our [0,1] floats
        global_transfo1_extra = GaussianBlur(p=1.0 - 1.0, radius_min=0.1, radius_max=1.0)  # intended: always (1.0)
        global_transfo2_extra = transforms.Compose([
            GaussianBlur(p=1.0 - 0.1, radius_min=0.1, radius_max=1.0),  # intended: rare (0.1)
            RangeFreeSolarize(p=0.2),
        ])
        local_transfo_extra = GaussianBlur(p=1.0 - 0.5, radius_min=0.1, radius_max=1.0)  # intended: half (0.5)

        self.global_transfo1 = transforms.Compose([base_photometric, global_transfo1_extra])
        self.global_transfo2 = transforms.Compose([base_photometric, global_transfo2_extra])
        self.local_transfo = transforms.Compose([base_photometric, local_transfo_extra])

    def _center_zoom(self, image: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]
        side = max(1, int(round(min(h, w) * self.zoom_fraction)))
        return TF.center_crop(image, [side, side])

    def __call__(self, image):
        if self.in_chans > 1 and image.shape[0] == 1:
            image = image.repeat(self.in_chans, 1, 1)

        output = {}

        if self.global_crops_mode == "context_zoom":
            # slot 1 (always-blurred, never solarized) = context,
            # slot 2 (rarely blurred, sometimes solarized) = crater zoom
            global_crop_1 = self.global_transfo1(self.context_view(image))
            global_crop_2 = self.global_transfo2(self.zoom_view(image))
        else:
            global_crop_1 = self.global_transfo1(self.geometric_global(image))
            global_crop_2 = self.global_transfo2(self.geometric_global(image))
        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        output["local_crops"] = [
            self.local_transfo(self.geometric_local(image))
            for _ in range(self.local_crops_number)
        ]
        output["offsets"] = ()
        return output
