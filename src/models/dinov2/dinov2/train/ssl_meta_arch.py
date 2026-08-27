# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging
import math

import numpy as np
import torch
from torch import nn

from src.models.dinov2.dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from src.models.dinov2.dinov2.models import build_model_from_cfg
from src.models.dinov2.dinov2.layers import DINOHead
from src.models.dinov2.dinov2.utils.utils import has_batchnorms
from src.models.dinov2.dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from src.models.dinov2.dinov2.fsdp import get_fsdp_wrapper, ShardedGradScaler, get_fsdp_modules, reshard_fsdp_model

from src.models.dinov2.dinov2.models.vision_transformer import BlockChunk


try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")


logger = logging.getLogger("dinov2")


def _resize_pretrained_pos_embed(state_dict: dict, target_pos_embed: torch.Tensor) -> None:
    """Bicubic-interpolate a checkpoint's pos_embed (patch tokens only -
    the CLS token position is kept as-is) to match the target model's own
    pos_embed shape, in place on state_dict.

    load_state_dict(strict=False) only tolerates missing/unexpected KEYS,
    not a shape mismatch on a key present in both - loading a checkpoint
    trained at a different resolution (e.g. finetuning this project's
    224px crops from the stock checkpoint's native 518px) needs this done
    explicitly before the load. The model's own interpolate_pos_encoding()
    (vision_transformer.py) doesn't help here - that only ever interpolates
    FROM the model's own already-loaded pos_embed at forward-pass time to
    match a given input's resolution; it's never invoked during checkpoint
    loading itself."""
    ckpt_pos_embed = state_dict.get("pos_embed")
    if ckpt_pos_embed is None or ckpt_pos_embed.shape == target_pos_embed.shape:
        return
    n_ckpt = ckpt_pos_embed.shape[1] - 1
    n_target = target_pos_embed.shape[1] - 1
    m_ckpt = int(math.sqrt(n_ckpt))
    m_target = int(math.sqrt(n_target))
    assert m_ckpt * m_ckpt == n_ckpt and m_target * m_target == n_target, (
        f"pos_embed patch grid isn't square: checkpoint has {n_ckpt} patch tokens, "
        f"target model has {n_target}")
    dim = ckpt_pos_embed.shape[-1]
    cls_pos_embed = ckpt_pos_embed[:, :1]
    patch_pos_embed = ckpt_pos_embed[:, 1:].reshape(1, m_ckpt, m_ckpt, dim).permute(0, 3, 1, 2)
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.float(), size=(m_target, m_target), mode="bicubic", antialias=True
    ).to(ckpt_pos_embed.dtype)
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, m_target * m_target, dim)
    state_dict["pos_embed"] = torch.cat([cls_pos_embed, patch_pos_embed], dim=1)
    logger.info(f"OPTIONS -- pretrained weights: interpolated pos_embed "
               f"{m_ckpt}x{m_ckpt} -> {m_target}x{m_target} patches")


class SSLMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None

        student_model_dict = dict()
        teacher_model_dict = dict()

        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        if cfg.student.pretrained_weights:
            chkpt = torch.load(cfg.student.pretrained_weights)
            logger.info(f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}")
            # DINOv2's own training checkpoints (FSDPCheckpointer) wrap the
            # state dict under "model"; the publicly released stock
            # inference checkpoints (e.g. dinov2_vits14_pretrain.pth) are
            # the flat state dict itself - support both rather than
            # assuming the training-checkpoint format.
            state_dict = chkpt["model"] if "model" in chkpt else chkpt
            _resize_pretrained_pos_embed(state_dict, student_backbone.pos_embed)
            missing, unexpected = student_backbone.load_state_dict(state_dict, strict=False)
            logger.info(f"OPTIONS -- pretrained weights: {len(missing)} missing, "
                       f"{len(unexpected)} unexpected keys (strict=False)")

        self.embed_dim = embed_dim
        self.dino_out_dim = cfg.dino.head_n_prototypes

        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head

        # Prototypical-loss auxiliary supervision (see src/train/
        # prototypical_loss.py's module docstring) - entirely optional,
        # absent from ssl_default_config.yaml's schema, so cfg.get(...)
        # rather than cfg.proto.* keeps every from-scratch/plain-finetune
        # config (no "proto:" section at all) working unmodified.
        proto_cfg = cfg.get("proto", None)
        self.do_proto = bool(proto_cfg) and proto_cfg.get("enabled", False)
        if self.do_proto:
            from src.train.prototypical_loss import LabeledEpisodePool

            logger.info("OPTIONS -- PROTO -- enabled")
            logger.info(f"OPTIONS -- PROTO -- loss_weight: {proto_cfg.loss_weight}")
            logger.info(f"OPTIONS -- PROTO -- n_support_per_class: {proto_cfg.n_support_per_class}")
            self.proto_loss_weight = proto_cfg.loss_weight
            self.proto_n_support = proto_cfg.n_support_per_class
            self.proto_pool = LabeledEpisodePool(
                dat_path=proto_cfg.dat_path, metadata_csv=proto_cfg.metadata_csv,
                split_csv=proto_cfg.split_csv, class_names=list(proto_cfg.class_names),
                crop_size=proto_cfg.get("crop_size", 224), in_chans=cfg.student.in_chans,
                device="cuda",
            )
            self._proto_rng = np.random.RandomState(proto_cfg.get("seed", 0))

        logger.info("OPTIONS -- DINO")
        if self.do_dino:
            logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
            logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
            logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
            logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
            self.dino_loss_weight = cfg.dino.loss_weight
            dino_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=cfg.dino.head_n_prototypes,
                hidden_dim=cfg.dino.head_hidden_dim,
                bottleneck_dim=cfg.dino.head_bottleneck_dim,
                nlayers=cfg.dino.head_nlayers,
            )
            self.dino_loss = DINOLoss(self.dino_out_dim)
            if self.do_koleo:
                logger.info("OPTIONS -- DINO -- applying KOLEO regularization")
                self.koleo_loss = KoLeoLoss()

        else:
            logger.info("OPTIONS -- DINO -- not using DINO")

        if self.do_dino or self.do_ibot:
            student_model_dict["dino_head"] = dino_head()
            teacher_model_dict["dino_head"] = dino_head()

        logger.info("OPTIONS -- IBOT")
        logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}")
        if self.do_ibot:
            self.ibot_loss_weight = cfg.ibot.loss_weight
            assert max(cfg.ibot.mask_ratio_min_max) > 0, "please provide a positive mask ratio tuple for ibot"
            assert cfg.ibot.mask_sample_probability > 0, "please provide a positive mask probability for ibot"
            self.ibot_out_dim = cfg.ibot.head_n_prototypes if self.ibot_separate_head else cfg.dino.head_n_prototypes
            self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
            if self.ibot_separate_head:
                logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
                logger.info(f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}")
                logger.info(f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}")
                logger.info(f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}")
                ibot_head = partial(
                    DINOHead,
                    in_dim=embed_dim,
                    out_dim=cfg.ibot.head_n_prototypes,
                    hidden_dim=cfg.ibot.head_hidden_dim,
                    bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                    nlayers=cfg.ibot.head_nlayers,
                )
                student_model_dict["ibot_head"] = ibot_head()
                teacher_model_dict["ibot_head"] = ibot_head()
            else:
                logger.info("OPTIONS -- IBOT -- head shared with DINO")

        self.need_to_synchronize_fsdp_streams = True

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        logger.info(f"Student and Teacher are built: they are both {cfg.student.arch} network.")

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def forward_backward(self, images, teacher_temp):
        n_global_crops = 2
        assert n_global_crops == 2
        n_local_crops = self.cfg.crops.local_crops_number

        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        local_crops = images["collated_local_crops"].cuda(non_blocking=True)

        # DIAGNOSTIC ONLY - std of the raw INPUT crops across the batch dim.
        # Rules out a data-loading bug (e.g. every sample being a duplicate
        # of the same crater) as the cause of near-zero teacher logit std -
        # if this is healthy but the logit std is still ~0, the problem is in
        # the backbone/head, not the data pipeline.
        self._diag_input_std = global_crops.detach().std(dim=0).mean()

        masks = images["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = images["mask_indices_list"].cuda(non_blocking=True)
        n_masked_patches_tensor = images["n_masked_patches"].cuda(non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = images["upperbound"]
        masks_weight = images["masks_weight"].cuda(non_blocking=True)

        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

        do_dino = self.do_dino
        do_ibot = self.do_ibot

        # loss scales
        ibot_loss_scale = 1.0 / n_global_crops

        # teacher output
        _diag_raw_teacher_logits = []  # DIAGNOSTIC ONLY - see use below

        @torch.no_grad()
        def get_teacher_output():
            x, n_global_crops_teacher = global_crops, n_global_crops
            teacher_backbone_output_dict = self.teacher.backbone(x, is_training=True)
            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
            teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops_teacher)
            # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
            teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))
            ibot_teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]
            _dim = ibot_teacher_patch_tokens.shape[-1]
            n_cls_tokens = teacher_cls_tokens.shape[0]

            if do_ibot and not self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound + n_cls_tokens, _dim)
                buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[n_cls_tokens : n_cls_tokens + n_masked_patches],
                )
                tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
                teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
                masked_teacher_patch_tokens_after_head = tokens_after_head[
                    n_cls_tokens : n_cls_tokens + n_masked_patches
                ]
            elif do_ibot and self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[:n_masked_patches],
                )
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(buffer_tensor_teacher)[
                    :n_masked_patches
                ]
            else:
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_ibot_softmaxed_centered = None

            _diag_raw_teacher_logits.append(teacher_cls_tokens_after_head.detach())  # DIAGNOSTIC ONLY

            if self.cfg.train.centering == "centering":
                teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])
                self.dino_loss.update_center(teacher_cls_tokens_after_head)
                if do_ibot:
                    masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.unsqueeze(0)
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.softmax_center_teacher(
                        masked_teacher_patch_tokens_after_head[:, :n_masked_patches], teacher_temp=teacher_temp
                    )
                    masked_teacher_ibot_softmaxed_centered = masked_teacher_ibot_softmaxed_centered.squeeze(0)
                    self.ibot_patch_loss.update_center(masked_teacher_patch_tokens_after_head[:n_masked_patches])

            elif self.cfg.train.centering == "sinkhorn_knopp":
                teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])

                if do_ibot:
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                        masked_teacher_patch_tokens_after_head,
                        teacher_temp=teacher_temp,
                        n_masked_patches_tensor=n_masked_patches_tensor,
                    )

            else:
                raise NotImplementedError

            return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered

        teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered = get_teacher_output()
        reshard_fsdp_model(self.teacher)

        loss_dict = {}

        # DIAGNOSTIC ONLY (not backpropagated) - mean entropy of the teacher's
        # centered/softmaxed output distribution over dino.head_n_prototypes.
        # If this collapses toward 0, the teacher's predictions have become
        # near-one-hot/degenerate (representation collapse) regardless of what
        # total_loss shows; if it stays near log(head_n_prototypes), the
        # teacher output is still close to uniform/uninformative.
        with torch.no_grad():
            _p = teacher_dino_softmaxed_centered_list.flatten(0, 1)
            loss_dict["diag_teacher_entropy"] = (
                -(_p * torch.log(_p.clamp_min(1e-12))).sum(dim=-1).mean()
            )
            # DIAGNOSTIC ONLY - std of the RAW (pre-centering, pre-softmax)
            # teacher logits ACROSS the batch dimension, averaged over
            # prototypes. Near-zero here means the backbone+head are
            # producing near-identical logits for different input crops -
            # i.e. the collapse is upstream of the loss/centering math
            # entirely, in representation learning itself.
            _raw = _diag_raw_teacher_logits[0]
            loss_dict["diag_teacher_logit_std_across_batch"] = _raw.std(dim=0).mean()
            loss_dict["diag_input_std_across_batch"] = self._diag_input_std

        loss_accumulator = 0  # for backprop
        student_global_backbone_output_dict, student_local_backbone_output_dict = self.student.backbone(
            [global_crops, local_crops], masks=[masks, None], is_training=True
        )

        inputs_for_student_head_list = []

        # 1a: local crops cls tokens
        student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

        # 1b: global crops cls tokens
        student_global_cls_tokens = student_global_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

        # 1c: global crops patch tokens
        if do_ibot:
            _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
            ibot_student_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(upperbound, _dim)
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
            )
            if not self.ibot_separate_head:
                inputs_for_student_head_list.append(buffer_tensor_patch_tokens.unsqueeze(0))
            else:
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(buffer_tensor_patch_tokens)[
                    :n_masked_patches
                ]

        # 2: run
        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list)
        outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))

        # 3a: local crops cls tokens
        student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3b: global crops cls tokens
        student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3c: global crops patch tokens
        if do_ibot and not self.ibot_separate_head:
            student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(0)[:n_masked_patches]

        if n_local_crops > 0:
            dino_local_crops_loss = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)

            # store for display
            loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_local_crops_loss

        # process global crops
        loss_scales = 2  # this is here since we process global crops together

        if do_dino:
            # compute loss
            dino_global_crops_loss = (
                self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head],
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_softmaxed_centered_list.flatten(0, 1)
                    ],  # these were chunked and stacked in reverse so A is matched to B
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )

            loss_dict["dino_global_crops_loss"] = dino_global_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_global_crops_loss

            student_cls_tokens = student_global_cls_tokens

            if self.do_koleo:
                koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
                )  # we don't apply koleo loss between cls tokens of a same image
                loss_accumulator += koleo_loss
                loss_dict["koleo_loss"] = (
                    koleo_loss / loss_scales
                )  # this is to display the same losses as before but we can remove eventually

        if do_ibot:
            # compute loss
            ibot_patch_loss = (
                self.ibot_patch_loss.forward_masked(
                    student_global_masked_patch_tokens_after_head,
                    masked_teacher_ibot_softmaxed_centered,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=masks_weight,
                )
                * loss_scales
                * ibot_loss_scale
            )

            # store for display
            loss_dict["ibot_loss"] = ibot_patch_loss / 2

            # accumulate loss
            loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        if self.do_proto:
            from src.train.prototypical_loss import prototypical_loss

            sx, sy, qx, qy = self.proto_pool.sample(self.proto_n_support, self._proto_rng)
            proto_loss = prototypical_loss(
                self.student.backbone, sx, sy, qx, qy, self.proto_pool.n_classes)
            # summed into the SAME loss_accumulator backprop_loss() below
            # calls .backward() on once - not a second, separate backward
            # pass, which FSDP's gradient hooks don't expect per step.
            # Kept small (proto.loss_weight, default 0.2 - see
            # configs/dino_craters_finetune.yaml's comment) - the proto
            # loss's episode is a much smaller, higher-variance signal than
            # the 50k-crater unlabeled pool the rest of the loss sees, so
            # it's meant as a gentle nudge, not an equal-weight objective.
            loss_accumulator += self.proto_loss_weight * proto_loss
            loss_dict["proto_loss"] = proto_loss.detach()

        self.backprop_loss(loss_accumulator)

        self.fsdp_synchronize_streams()

        return loss_dict

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            # ._streams is a private FSDP attribute this was written against
            # on an older PyTorch; newer FSDP (torch 2.7.1 here) doesn't
            # expose it. It's a stream-sharing perf optimization for
            # cross-shard async overlap, not correctness-critical -
            # torch.cuda.synchronize() above already covers correctness.
            if hasattr(self.teacher.backbone, "_streams"):
                self.student.dino_head._streams = (
                    self.teacher.dino_head._streams
                ) = self.student.backbone._streams = self.teacher.backbone._streams
            self.need_to_synchronize_fsdp_streams = False

    def update_teacher(self, m):
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for k in self.student.keys():
                for ms, mt in zip(get_fsdp_modules(self.student[k]), get_fsdp_modules(self.teacher[k])):
                    student_param_list += ms.params
                    teacher_param_list += mt.params
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def train(self):
        super().train()
        self.teacher.eval()

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        if has_batchnorms(self.student):
            raise NotImplementedError
        # below will synchronize all student subnetworks across gpus:
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
            student_model_cfg = self.cfg.compute_precision.student[k]
            self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student[k])
            teacher_model_cfg = self.cfg.compute_precision.teacher[k]
            self.teacher[k] = get_fsdp_wrapper(teacher_model_cfg, modules_to_wrap={BlockChunk})(self.teacher[k])
