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

from src.models.dinov2.dinov2.models.vision_transformer import vit_small
from src.models.dinov2.dinov2.utils.utils import load_pretrained_weights

# must match configs/dino_craters.yaml's student config
DINO_VIT_KWARGS = dict(
    img_size=128,
    patch_size=8,
    in_chans=1,
    channel_adaptive=False,
)


def load_dino_backbone(checkpoint_path: str, device="cpu", **vit_kwargs):
    kwargs = {**DINO_VIT_KWARGS, **vit_kwargs}
    model = vit_small(**kwargs)
    # strict=False inside load_pretrained_weights: the "teacher" checkpoint
    # also has dino_head.* keys (from the ModuleDict backbone+dino_head both
    # live under), which a bare backbone-only model doesn't have and doesn't
    # need - those just get reported as unexpected/ignored, not an error.
    load_pretrained_weights(model, checkpoint_path, "teacher")
    model.to(device).eval()
    return model
