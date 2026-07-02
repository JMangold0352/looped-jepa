from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa.models.encoder import VisionTransformerEncoder
from jepa.models.looped_predictor import LoopedPredictor
from jepa.models.predictor import VisionTransformerPredictor
from jepa.models.vit import VICRegProjector, rms_norm_factory, vicreg_variance_covariance


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, momentum: float) -> None:
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(momentum).add_(p.data, alpha=1.0 - momentum)


class IJEPA(nn.Module):
    """I-JEPA: context encoder + EMA target encoder + patch predictor.

    The student encoder sees only context patches. The EMA teacher encodes the
    full image and supplies stop-gradient targets. Only the student encoder is
    kept for downstream linear probing.
    """

    def __init__(
        self,
        encoder: VisionTransformerEncoder,
        target_encoder: VisionTransformerEncoder,
        predictor: VisionTransformerPredictor | LoopedPredictor,
        ema_momentum: float = 0.996,
        reg_enabled: bool = False,
        reg_var_coeff: float = 1.0,
        reg_cov_coeff: float = 0.04,
        reg_projector_dims: list[int] | None = None,
        encoder_dim: int = 192,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.ema_momentum = ema_momentum

        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.reg_enabled = reg_enabled
        self.reg_var_coeff = reg_var_coeff
        self.reg_cov_coeff = reg_cov_coeff
        self.reg_scale = 1.0
        self.projector: VICRegProjector | None = None
        if reg_enabled:
            dims = reg_projector_dims if reg_projector_dims else [encoder_dim]
            self.projector = VICRegProjector(encoder_dim, dims)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> IJEPA:
        enc_cfg = cfg["encoder"]
        pred_cfg = cfg["predictor"]
        grid_size = cfg["data"]["img_size"] // cfg["data"]["patch_size"]
        num_patches = grid_size ** 2

        encoder = VisionTransformerEncoder(
            img_size=cfg["data"]["img_size"],
            patch_size=cfg["data"]["patch_size"],
            embed_dim=enc_cfg["embed_dim"],
            depth=enc_cfg["depth"],
            num_heads=enc_cfg["num_heads"],
            mlp_ratio=enc_cfg.get("mlp_ratio", 4.0),
            dropout=enc_cfg.get("dropout", 0.0),
            drop_path=enc_cfg.get("drop_path", 0.0),
        )
        target_encoder = copy.deepcopy(encoder)

        if pred_cfg.get("ouro", False):
            base_predictor = VisionTransformerPredictor.ouro_ready(
                num_patches=num_patches,
                grid_size=grid_size,
                encoder_dim=enc_cfg["embed_dim"],
                predictor_dim=pred_cfg["embed_dim"],
                depth=pred_cfg["depth"],
                num_heads=pred_cfg["num_heads"],
            )
        else:
            norm_type = pred_cfg.get("norm", "layer").lower()
            norm_factory = rms_norm_factory if norm_type == "rms" else (lambda d: nn.LayerNorm(d))
            base_predictor = VisionTransformerPredictor(
                num_patches=num_patches,
                grid_size=grid_size,
                encoder_dim=enc_cfg["embed_dim"],
                predictor_dim=pred_cfg["embed_dim"],
                depth=pred_cfg["depth"],
                num_heads=pred_cfg["num_heads"],
                mlp_ratio=pred_cfg.get("mlp_ratio", 4.0),
                dropout=pred_cfg.get("dropout", 0.0),
                norm_factory=norm_factory,
                sandwich_norm=bool(pred_cfg.get("sandwich_norm", False)),
            )

        if pred_cfg.get("looped", False):
            predictor: VisionTransformerPredictor | LoopedPredictor = LoopedPredictor(
                base_predictor,
                max_loops=pred_cfg.get("max_loops", 4),
                use_exit_gate=pred_cfg.get("use_exit_gate", False),
            )
        else:
            predictor = base_predictor

        reg_cfg = cfg.get("regularizer", {}) or {}
        return cls(
            encoder=encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            ema_momentum=cfg["train"].get("ema_momentum_start", 0.996),
            reg_enabled=reg_cfg.get("enabled", False),
            reg_var_coeff=reg_cfg.get("var_coeff", 1.0),
            reg_cov_coeff=reg_cfg.get("cov_coeff", 0.04),
            reg_projector_dims=reg_cfg.get("projector") or None,
            encoder_dim=enc_cfg["embed_dim"],
        )

    def train(self, mode: bool = True) -> IJEPA:
        super().train(mode)
        self.target_encoder.eval()
        return self

    def num_trainable_params(self) -> int:
        """Trainable param count excluding the frozen EMA copy."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        images: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
        teacher_images: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        context_repr = self.encoder(images, context_indices)

        teacher_input = teacher_images if teacher_images is not None else images
        with torch.no_grad():
            all_tokens = self.target_encoder.forward_all_patches(teacher_input)
            idx = target_indices.unsqueeze(-1).expand(-1, -1, all_tokens.size(-1))
            target_repr = torch.gather(all_tokens, 1, idx)

        pred_out = self.predictor(context_repr, context_indices, target_indices)
        if isinstance(pred_out, tuple):
            pred_repr, exit_probs = pred_out
        else:
            pred_repr, exit_probs = pred_out, None

        pred_loss = F.smooth_l1_loss(pred_repr, target_repr)
        loss = pred_loss

        out: dict[str, torch.Tensor] = {
            "pred_loss": pred_loss.detach(),
            "pred_repr": pred_repr,
            "target_repr": target_repr,
        }

        if self.reg_enabled and self.projector is not None:
            z = self.projector(context_repr.mean(dim=1))
            var_loss, cov_loss = vicreg_variance_covariance(z)
            reg_loss = self.reg_var_coeff * var_loss + self.reg_cov_coeff * cov_loss
            loss = loss + self.reg_scale * reg_loss
            out["reg_var"] = var_loss.detach()
            out["reg_cov"] = cov_loss.detach()

        out["loss"] = loss
        if exit_probs is not None:
            out["exit_probs"] = exit_probs
        return out

    def update_target_encoder(self, momentum: float) -> None:
        update_ema(self.target_encoder, self.encoder, momentum)
