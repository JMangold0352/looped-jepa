from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from jepa.models.vit import Block, PatchEmbed, build_pos_embed, init_pos_embed


class VisionTransformerEncoder(nn.Module):
    """ViT encoder over a subset of patches or the full patch grid."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_factory: Callable[[int], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        norm_factory = norm_factory or (lambda d: nn.LayerNorm(d))
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = build_pos_embed(self.num_patches, embed_dim)
        init_pos_embed(self.pos_embed)

        # Linearly increasing stochastic-depth rate from 0 to ``drop_path``.
        dpr = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    norm_factory=norm_factory,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_factory(embed_dim)

    def forward(self, x: torch.Tensor, patch_indices: torch.Tensor) -> torch.Tensor:
        """Encode ``patch_indices`` from a batch of images (B, C, H, W)."""
        b = x.shape[0]
        tokens = self.patch_embed(x)
        pos = self.pos_embed.expand(b, -1, -1)
        idx = patch_indices.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
        gathered = torch.gather(tokens, 1, idx) + torch.gather(pos, 1, idx)

        for block in self.blocks:
            gathered = block(gathered)
        return self.norm(gathered)

    def forward_all_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Encode every patch; used by the EMA teacher and linear probe."""
        b = x.shape[0]
        tokens = self.patch_embed(x) + self.pos_embed.expand(b, -1, -1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)
