from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass
class MaskBatch:
    context_indices: list[torch.Tensor]
    target_indices: list[torch.Tensor]


class IJEPAMaskCollator:
    """I-JEPA-style multi-block masking on a square patch grid."""

    def __init__(
        self,
        grid_size: int = 8,
        num_target_blocks: int = 4,
        target_scale: tuple[float, float] = (0.15, 0.2),
        context_scale: tuple[float, float] = (0.85, 1.0),
        aspect_ratio: tuple[float, float] = (0.75, 1.5),
        min_context_patches: int = 10,
        fixed_context_patches: int | None = 32,
        fixed_target_patches: int | None = 16,
        context_patches_range: tuple[int, int] | None = None,
        target_patches_range: tuple[int, int] | None = None,
    ) -> None:
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        self.num_target_blocks = num_target_blocks
        self.target_scale = target_scale
        self.context_scale = context_scale
        self.aspect_ratio = aspect_ratio
        self.min_context_patches = min_context_patches
        self.fixed_context_patches = fixed_context_patches
        self.fixed_target_patches = fixed_target_patches
        self.context_patches_range = context_patches_range
        self.target_patches_range = target_patches_range

    def _sample_block(self, scale_range: tuple[float, float]) -> set[int]:
        g = self.grid_size
        target_area = random.uniform(*scale_range) * self.num_patches
        aspect = random.uniform(*self.aspect_ratio)
        h = max(1, min(g, int(round((target_area * aspect) ** 0.5))))
        w = max(1, min(g, int(round((target_area / aspect) ** 0.5))))
        top = random.randint(0, g - h)
        left = random.randint(0, g - w)
        indices: set[int] = set()
        for r in range(top, top + h):
            for c in range(left, left + w):
                indices.add(r * g + c)
        return indices

    def __call__(self, batch_size: int) -> MaskBatch:
        context_indices: list[torch.Tensor] = []
        target_indices: list[torch.Tensor] = []

        target_count = self.fixed_target_patches
        if self.target_patches_range is not None:
            target_count = random.randint(*self.target_patches_range)
        context_count = self.fixed_context_patches
        if self.context_patches_range is not None:
            context_count = random.randint(*self.context_patches_range)

        for _ in range(batch_size):
            targets: set[int] = set()
            for _ in range(self.num_target_blocks):
                targets |= self._sample_block(self.target_scale)

            if target_count is not None:
                pool = list(set(range(self.num_patches)) - targets)
                need = target_count - len(targets)
                if need > 0 and pool:
                    targets |= set(random.sample(pool, min(need, len(pool))))
                if len(targets) > target_count:
                    targets = set(sorted(targets)[:target_count])

            all_idx = set(range(self.num_patches))
            context = self._sample_block(self.context_scale) - targets
            if len(context) < self.min_context_patches:
                context = all_idx - targets

            if context_count is not None:
                available = all_idx - targets
                base = context if context else set(available)
                if len(base) >= context_count:
                    context = set(random.sample(sorted(base), context_count))
                else:
                    context = set(base)
                    remaining = sorted(available - context)
                    need = context_count - len(context)
                    if need > 0 and remaining:
                        context |= set(random.sample(remaining, min(need, len(remaining))))

            context_indices.append(torch.tensor(sorted(context), dtype=torch.long))
            target_indices.append(torch.tensor(sorted(targets), dtype=torch.long))

        return MaskBatch(context_indices=context_indices, target_indices=target_indices)
