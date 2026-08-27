"""
dino_backbone.py
──────────────────
Load a trained DINOv2 checkpoint's teacher backbone for inference
(embedding extraction / clustering) — parallel to src/models/mae_bottleneck.py's
load_mae() for MAE.

Confirmed against dinov2/eval/setup.py's own build_model_for_eval(): DINOv2's
official evaluation convention uses the EMA-smoothed TEACHER network (not the
student) for downstream tasks. The plain backbone's forward() already returns
the final-block, normalized CLS token through an identity head — verified
against get_intermediate_layers(n=1, return_class_token=True), which applies
the same final norm() before extracting the class token, so forward() IS one
of DINOv2's own official evaluation feature configurations (n_last_blocks=1,
avgpool=False), not an approximation of it. The CLS token is DINOv2's
documented choice for whole-image/global embeddings (retrieval,
classification-style tasks) as opposed to patch tokens (dense/spatial tasks
like segmentation) - matches our use case (cluster craters by overall
degradation state).

Loads the checkpoint train_dino.py explicitly saves via do_test() at the end
of training (eval/final/teacher_checkpoint.pth) — NOT the main
FSDPCheckpointer output (model_final.rank_N.pth). That one uses FSDP's
LOCAL_STATE_DICT format, meant for resuming training on the same sharding
topology (its key names reflect FSDP-internal block-chunking, not the
model's natural structure) - not a portable, loadable-elsewhere checkpoint.
do_test()'s format ({"teacher": model.teacher.state_dict()}, no FSDP
state-dict-type context manager) is what DINOv2's own load_pretrained_weights()
is built for, so this reuses that helper directly instead of reimplementing
prefix-stripping.
"""

import torch
import torch.nn.functional as F

from src.models.dinov2.dinov2.models.vision_transformer import vit_small
from src.models.dinov2.dinov2.utils.utils import load_pretrained_weights

# must match configs/dino_craters.yaml's student config (from-scratch runs)
DINO_VIT_KWARGS = dict(
    img_size=128,
    patch_size=8,
    in_chans=1,
    channel_adaptive=False,
)

# img_size this project has always paired with each patch_size - patch8/
# in_chans1 is the from-scratch convention (configs/dino_craters.yaml, 128px
# .dat crop resolution); patch14/in_chans3 is the checkpoint-matching
# finetune convention (configs/dino_craters_finetune.yaml, 224px global
# crops - chosen to match the stock ImageNet checkpoint's own architecture,
# see that config's own comment). Both happen to produce the same 16x16=256
# patch grid (128/8=16, 224/14=16), so pos_embed's shape alone can't tell
# these two architectures apart - only patch_embed.proj.weight's kernel
# shape (which directly encodes patch_size/in_chans) can, which is what
# _detect_dino_arch() below reads.
_DINO_IMG_SIZE_BY_PATCH_SIZE = {8: 128, 14: 224}


def _detect_dino_arch(checkpoint_path: str, checkpoint_key: str = "teacher") -> dict:
    """Auto-detect patch_size/in_chans/img_size from a checkpoint's own
    patch_embed.proj.weight shape, instead of assuming DINO_VIT_KWARGS's
    from-scratch convention unconditionally - every "dino"-family checkpoint
    used to share that one architecture (from-scratch training only), so the
    hardcoded kwargs were never wrong until configs/dino_craters_finetune.yaml
    introduced a second one (patch14/in_chans3, to match the stock
    checkpoint's own shape for finetuning). Confirmed the actual failure
    mode directly: loading a patch14/in_chans3 checkpoint with
    DINO_VIT_KWARGS's patch8/in_chans1 raised "size mismatch for
    patch_embed.proj.weight: ... torch.Size([384, 3, 14, 14]) ...
    torch.Size([384, 1, 8, 8])" - pos_embed itself didn't also mismatch
    (see _DINO_IMG_SIZE_BY_PATCH_SIZE's comment on why), so this is the one
    key that actually needs inspecting. Same auto-detect philosophy
    build_model()'s CAE/MAE branches already use for their own
    checkpoint-specific shape drift (see cluster.py). Returns {} (no
    override, DINO_VIT_KWARGS wins) if the checkpoint has no patch_embed key
    at all - lets the load below raise its own clear error rather than
    silently guessing."""
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = raw[checkpoint_key] if checkpoint_key is not None and checkpoint_key in raw else raw
    sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
    w = sd.get("patch_embed.proj.weight")
    if w is None:
        return {}
    _, in_chans, patch_size, _ = w.shape
    detected = {"patch_size": int(patch_size), "in_chans": int(in_chans)}
    if int(patch_size) in _DINO_IMG_SIZE_BY_PATCH_SIZE:
        detected["img_size"] = _DINO_IMG_SIZE_BY_PATCH_SIZE[int(patch_size)]
    return detected


def load_dino_backbone(checkpoint_path: str, device="cpu", **vit_kwargs):
    kwargs = {**DINO_VIT_KWARGS, **_detect_dino_arch(checkpoint_path), **vit_kwargs}
    model = vit_small(**kwargs)
    # strict=False inside load_pretrained_weights: the "teacher" checkpoint
    # also has dino_head.* keys (from the ModuleDict backbone+dino_head both
    # live under), which a bare backbone-only model doesn't have and doesn't
    # need - those just get reported as unexpected/ignored, not an error.
    load_pretrained_weights(model, checkpoint_path, "teacher")
    model.to(device).eval()
    return model


# Official Meta ImageNet/LVD-142M-pretrained ViT-S/14 - completely stock
# architecture, no channel/patch-size customization at all. This is the
# "use DINOv2 as-is" path: after diagnosing that training vit_small from
# scratch on crater crops collapses (near-identical embeddings for genuinely
# different craters, confirmed via direct cosine-similarity testing), and
# confirming the DINOv2 paper itself never trains small architectures from
# scratch ("we distill them from our largest model... instead of training
# them from scratch"), this is the validated alternative: frozen pretrained
# features, no training at all - matches the geological CT-scan paper's
# strongest simple baseline (non-fine-tuned DINOv2 "demonstrates strong
# performance... even outside its original training distribution").
#
# Verified directly (not just assumed): stock-loaded this way, cosine
# similarity between different real crater crops is 0.75-0.88 (genuine
# differentiation) vs. 0.976-0.9998 for our from-scratch-trained backbone
# (collapsed). See session diagnostics for the isolated test.
STOCK_DINO_CHECKPOINT = "data/raw/pretrained/dinov2_vits14_pretrain.pth"
STOCK_DINO_VIT_KWARGS = dict(
    img_size=518,          # native resolution the checkpoint's pos_embed was trained at
    patch_size=14,
    in_chans=3,
    block_chunks=0,        # flat "blocks.N.*" naming, matches the checkpoint's keys
    init_values=1.0,        # enables LayerScale (ls1/ls2.gamma) - the checkpoint has these
)
STOCK_DINO_FORWARD_SIZE = 224   # actual resolution fed at inference (pos_embed
                                # interpolates automatically - see interpolate_pos_encoding)


def load_stock_pretrained_dino(checkpoint_path: str = STOCK_DINO_CHECKPOINT, device="cpu"):
    model = vit_small(**STOCK_DINO_VIT_KWARGS)
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    msg = model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def encode_stock_pretrained_dino(model, imgs: torch.Tensor) -> torch.Tensor:
    """
    imgs: (B, 1, H, W) float, whatever scale the rest of the pipeline uses
    (global-scaled [0,1] or similar) - channel-replicated to 3 and resized
    to STOCK_DINO_FORWARD_SIZE here, so callers don't need architecture-
    specific preprocessing beyond what every other encoder already does.

    L2-normalized (unit norm per sample) before returning - matches DINOv2's
    OWN official eval code exactly: dinov2/eval/utils.py's ModelWithNormalize
    (nn.functional.normalize(..., dim=1, p=2)), which they wire into knn.py
    (their k-NN classifier) but NOT linear.py (their linear-probe script).
    k-NN and KMeans are both Euclidean-distance-based, so the same rationale
    applies directly here: without normalizing, whichever samples have
    larger raw embedding magnitude dominate distance calculations regardless
    of direction - confirmed as the likely cause of a confound (clustering
    correlating with Sobel edge-magnitude) found when clustering un-
    normalized embeddings on the full crater dataset.
    """
    x = F.interpolate(imgs, size=(STOCK_DINO_FORWARD_SIZE, STOCK_DINO_FORWARD_SIZE),
                      mode="bicubic", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    with torch.no_grad():
        out = model(x, is_training=True)
        feats = out["x_norm_clstoken"]
        feats = F.normalize(feats, dim=1, p=2)
    return feats
