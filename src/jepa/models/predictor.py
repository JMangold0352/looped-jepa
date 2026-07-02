from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from jepa.models.vit import (
    Block,
    BlockStack,
    PatchEmbed,
    RotaryEmbedding2D,
    build_pos_embed,
    init_pos_embed,
    rms_norm_factory,
)


class VisionTransformerPredictor(nn.Module):
    """Narrow ViT predictor, structured as a reusable BlockStack for looped recurrence."""

    def __init__(
        self,
        num_patches: int = 64,
        grid_size: int = 8,
        encoder_dim: int = 192,
        predictor_dim: int = 96,
        depth: int = 4,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        norm_factory: Callable[[int], nn.Module] | None = None,
        sandwich_norm: bool = False,
        ffn_type: str = "mlp",
        use_rope: bool = False,
    ) -> None:
        super().__init__()
        norm_factory = norm_factory or (lambda d: nn.LayerNorm(d))
        self.num_patches = num_patches
        self.predictor_dim = predictor_dim

        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.pos_embed = build_pos_embed(num_patches, predictor_dim)
        init_pos_embed(self.pos_embed)

        self.ctx_proj = nn.Linear(encoder_dim, predictor_dim)

        rope = RotaryEmbedding2D(predictor_dim // num_heads, grid_size) if use_rope else None

        blocks = nn.ModuleList(
            [
                Block(
                    predictor_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    norm_factory=norm_factory,
                    sandwich_norm=sandwich_norm,
                    ffn_type=ffn_type,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )
        self.block_stack = BlockStack(blocks)
        self.norm = norm_factory(predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, encoder_dim)

    def build_sequence(
        self,
        context_repr: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble predictor input: context tokens + mask tokens at target positions."""
        b = context_repr.shape[0]
        pos = self.pos_embed.expand(b, -1, -1)
        seq_len = context_indices.shape[1] + target_indices.shape[1]
        x = context_repr.new_zeros(b, seq_len, self.predictor_dim)

        ctx_pos = torch.gather(pos, 1, context_indices.unsqueeze(-1).expand(-1, -1, self.predictor_dim))
        tgt_pos = torch.gather(pos, 1, target_indices.unsqueeze(-1).expand(-1, -1, self.predictor_dim))

        x[:, : context_indices.shape[1]] = self.ctx_proj(context_repr) + ctx_pos
        x[:, context_indices.shape[1] :] = self.mask_token.expand(b, target_indices.shape[1], -1) + tgt_pos
        return x

    def forward_stack(self, x: torch.Tensor) -> torch.Tensor:
        """One pass through the shared block stack (used by LoopedPredictor)."""
        x = self.block_stack(x)
        return self.norm(x)

    def forward(
        self,
        context_repr: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        x = self.build_sequence(context_repr, context_indices, target_indices)
        x = self.forward_stack(x)
        return self.output_proj(x[:, context_indices.shape[1] :])

    @classmethod
    def ouro_ready(
        cls,
        num_patches: int = 64,
        grid_size: int = 8,
        encoder_dim: int = 192,
        predictor_dim: int = 96,
        depth: int = 4,
        num_heads: int = 3,
    ) -> VisionTransformerPredictor:
        """RMSNorm + SwiGLU + 2D RoPE, used by the looped predictor path."""
        return cls(
            num_patches=num_patches,
            grid_size=grid_size,
            encoder_dim=encoder_dim,
            predictor_dim=predictor_dim,
            depth=depth,
            num_heads=num_heads,
            norm_factory=rms_norm_factory,
            sandwich_norm=True,
            ffn_type="swiglu",
            use_rope=True,
        )
