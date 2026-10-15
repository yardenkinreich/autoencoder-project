"""
Tests for the context_zoom global-crop mode of CraterAugmentationDINO.
Run with: PYTHONPATH=$(pwd) python -m pytest src/test/test_dino_augmentation.py
"""
import numpy as np
import pytest
import torch

from src.data.dino_craters_augmentation import CraterAugmentationDINO, recipe_version


def _ring_image(side=384, fov=2.0):
    """A bright rim of diameter side/fov centred in a side x side image -
    i.e. a crater cut at fov diameters, as crop_crater/build_catalog do."""
    yy, xx = np.mgrid[:side, :side] - (side - 1) / 2.0
    r = np.hypot(xx, yy)
    rim = side / fov / 2.0
    img = np.exp(-((r - rim) ** 2) / (2 * 3.0 ** 2)).astype(np.float32)
    return torch.from_numpy(img)[None]


def _rim_radius(view):
    """Radius (px) at which the view's radial profile peaks."""
    a = view[0].numpy()
    n = a.shape[0]
    yy, xx = np.mgrid[:n, :n] - (n - 1) / 2.0
    r = np.hypot(xx, yy).round().astype(int)
    cnt = np.bincount(r.ravel())
    prof = np.bincount(r.ravel(), a.ravel()) / np.maximum(cnt, 1)
    prof[cnt == 0] = np.nan                  # even n: no pixel sits at r=0
    return int(np.nanargmax(prof[: n // 2 + 2]))


def _aug(**kw):
    return CraterAugmentationDINO([0.32, 1.0], [0.05, 0.32], 2, global_crops_size=224,
                                  local_crops_size=98, **kw)


def test_context_zoom_views_have_fixed_scales():
    aug = _aug(global_crops_mode="context_zoom", image_fov_diameters=2.0, zoom_diameters=1.0)
    aug.global_transfo1 = aug.global_transfo2 = lambda x: x     # geometry only
    for _ in range(5):                                           # only flips are random
        out = aug(_ring_image())
        g1, g2 = out["global_crops"]
        assert g1.shape == g2.shape == (1, 224, 224)
        assert out["global_crops_teacher"] == [g1, g2]
        # context: rim diameter is half the frame -> radius ~ 224/4 = 56
        assert abs(_rim_radius(g1) - 56) <= 2
        # zoom at 1 diameter: rim touches the frame -> radius ~ 112
        assert _rim_radius(g2) >= 108
        assert len(out["local_crops"]) == 2 and out["local_crops"][0].shape == (1, 98, 98)


def test_zoom_fraction_follows_fov_and_zoom():
    aug = _aug(global_crops_mode="context_zoom", image_fov_diameters=3.0, zoom_diameters=1.5)
    aug.global_transfo1 = aug.global_transfo2 = lambda x: x
    g2 = aug(_ring_image(side=450, fov=3.0))["global_crops"][1]
    # 1.5-diameter zoom of a crater: rim radius = 224 / 1.5 / 2 ~ 75
    assert abs(_rim_radius(g2) - 75) <= 2


def test_zoom_must_fit_inside_image():
    with pytest.raises(ValueError):
        _aug(global_crops_mode="context_zoom", image_fov_diameters=2.0, zoom_diameters=2.5)
    with pytest.raises(ValueError):
        _aug(global_crops_mode="bogus")


def test_three_channel_replication_and_ragged_input():
    aug = _aug(global_crops_mode="context_zoom", in_chans=3)
    out = aug(torch.rand(1, 301, 299))                 # native catalog crops are ragged
    assert all(g.shape == (3, 224, 224) for g in out["global_crops"])


def test_recipe_version_marks_mode():
    assert recipe_version("random") == recipe_version()
    assert recipe_version("context_zoom").endswith("+context_zoom")


def test_teacher_style_views_for_labelled_batches():
    from omegaconf import OmegaConf
    from src.train.labeled_views import TeacherStyleViews
    crops = OmegaConf.create({"global_crops_scale": [0.32, 1.0], "local_crops_scale": [0.05, 0.32],
                              "global_crops_size": 224, "local_crops_size": 98,
                              "image_fov_diameters": 2.0, "zoom_diameters": 1.0})
    v = TeacherStyleViews(crops, in_chans=3)
    v.aug.global_transfo1 = v.aug.global_transfo2 = lambda x: x      # geometry only
    x = _ring_image(side=448).repeat(3, 1, 1)[None].repeat(4, 1, 1, 1)
    a, b = v(x)
    assert a.shape == b.shape == (4, 3, 224, 224)
    for i in range(4):        # rot90/flips keep a centred ring centred
        assert abs(_rim_radius(a[i, :1]) - 56) <= 2 and _rim_radius(b[i, :1]) >= 108


def test_balanced_episode_sampler():
    import numpy as np
    from src.train.prototypical_loss import sample_balanced_episode_idx
    y = np.array([0] * 27 + [1] * 57 + [2] * 114 + [3] * 318)
    s, q = sample_balanced_episode_idx(y, 4, 10, 32, np.random.RandomState(0))
    assert np.bincount(y[s]).tolist() == [10, 10, 10, 10]
    assert np.bincount(y[q]).tolist() == [32, 32, 32, 32]
    assert not set(s) & set(q)


def test_photometric_jitter_never_clamps_normalized_data():
    # robust/asinh-normalized Kaguya crops are centred on 0; v3's torchvision
    # adjust_brightness/contrast clamped them to [0,1]
    from src.data.dino_craters_augmentation import (RangeFreeBrightness, RangeFreeContrast,
                                                    RangeFreeSolarize)
    torch.manual_seed(0)
    x = torch.randn(3, 64, 64) * 2.0
    x[1:] = x[:1]                                   # replicated grayscale
    for t in (RangeFreeBrightness(), RangeFreeContrast()):
        y = t(x.clone())
        assert (y < 0).float().mean() > 0.2 and (y > 1).float().mean() > 0.1, t
        assert torch.equal(y[0], y[1]) and torch.equal(y[1], y[2]), t   # no colour tint
    y = RangeFreeSolarize(p=1.0)(x.clone())       # mirror above the crop's own midpoint
    mid = (x.min() + x.max()) / 2
    assert torch.isclose(y.min(), x.min()) and y.max() <= mid + 1e-5
    assert y.unique().numel() > 1000              # nothing collapsed to a constant
    aug = _aug(global_crops_mode="context_zoom", in_chans=3)
    out = aug(x[:1].clone())
    g = out["global_crops"][0]
    assert (g < 0).float().mean() > 0.2, "global crop was clamped"
