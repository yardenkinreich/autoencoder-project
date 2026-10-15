"""
labeled_views.py — teacher-style augmented views for the LABELLED batches of
the auxiliary losses (prototypical, ordinal SupCon).

Each labelled crater is shown as the SAME two global views the DINO teacher
sees (src/data/dino_craters_augmentation.py, global_crops_mode=context_zoom):

  view 1  context (the whole 2xD image)  + global_transfo1: brightness/contrast
                                           jitter, ALWAYS Gaussian-blurred
  view 2  zoom (centred 1xD crater)       + global_transfo2: jitter, rarely
                                           blurred, sometimes solarized
  both    random flips (from the view) and an independent random 90-degree
          rotation (labelled views only - craters have no preferred
          orientation, and the label must not depend on it)

so the labelled signal is learned under the same appearance changes the
self-distillation is made invariant to, instead of on one fixed,
unaugmented rendering per crater (what the prototypical loss used before).

The pool images should be stored at ~2x the global crop size (448 for 224):
the zoom view is the central half of the image, so at 448 it is a native
224 crop rather than an upsampled 112.
"""
from __future__ import annotations

import torch

from src.data.dino_craters_augmentation import CraterAugmentationDINO


class TeacherStyleViews:
    def __init__(self, crops_cfg, in_chans: int = 3, rot90: bool = True):
        self.aug = CraterAugmentationDINO(
            crops_cfg.global_crops_scale, crops_cfg.local_crops_scale, 0,
            global_crops_size=crops_cfg.global_crops_size,
            local_crops_size=crops_cfg.local_crops_size, in_chans=in_chans,
            global_crops_mode="context_zoom",
            image_fov_diameters=crops_cfg.get("image_fov_diameters", 2.0),
            zoom_diameters=crops_cfg.get("zoom_diameters", 1.0))
        self.rot90 = rot90

    def _rot(self, v: torch.Tensor) -> torch.Tensor:
        if not self.rot90:
            return v
        return torch.rot90(v, int(torch.randint(0, 4, (1,))), dims=(-2, -1))

    def __call__(self, x: torch.Tensor):
        """x: (B, C, H, W) labelled images -> (context views, zoom views),
        each (B, C, S, S) at the global crop size, same order as x."""
        a, b = [], []
        for img in x:
            a.append(self._rot(self.aug.global_transfo1(self.aug.context_view(img))))
            b.append(self._rot(self.aug.global_transfo2(self.aug.zoom_view(img))))
        return torch.stack(a), torch.stack(b)
