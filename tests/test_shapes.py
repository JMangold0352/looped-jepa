from __future__ import annotations

import copy

import torch

from jepa.masking import IJEPAMaskCollator
from jepa.models.jepa import IJEPA
from jepa.models.looped_predictor import LoopedPredictor
from jepa.utils.config import load_config


def tiny_config() -> dict:
    cfg = load_config("configs/smoke_test.yaml")
    return cfg


def test_ijepa_forward_shapes() -> None:
    cfg = tiny_config()
    model = IJEPA.from_config(cfg)
    model.eval()

    b = 2
    images = torch.randn(b, 3, 32, 32)
    grid = cfg["data"]["img_size"] // cfg["data"]["patch_size"]
    collator = IJEPAMaskCollator(grid_size=grid, fixed_context_patches=32, fixed_target_patches=16)
    masks = collator(b)
    ctx = torch.stack(masks.context_indices)
    tgt = torch.stack(masks.target_indices)

    out = model(images, ctx, tgt)
    assert out["loss"].ndim == 0
    assert out["pred_repr"].shape == out["target_repr"].shape
    assert out["pred_repr"].shape[0] == b
    assert out["pred_repr"].shape[1] == 16


def test_looped_predictor() -> None:
    cfg = copy.deepcopy(tiny_config())
    cfg["predictor"]["looped"] = True
    cfg["predictor"]["ouro"] = True
    cfg["predictor"]["max_loops"] = 2
    cfg["predictor"]["use_exit_gate"] = True

    model = IJEPA.from_config(cfg)
    assert isinstance(model.predictor, LoopedPredictor)

    b = 2
    images = torch.randn(b, 3, 32, 32)
    grid = cfg["data"]["img_size"] // cfg["data"]["patch_size"]
    collator = IJEPAMaskCollator(grid_size=grid, fixed_context_patches=32, fixed_target_patches=16)
    masks = collator(b)
    ctx = torch.stack(masks.context_indices)
    tgt = torch.stack(masks.target_indices)

    out = model(images, ctx, tgt)
    assert "exit_probs" in out
    assert out["exit_probs"].shape[0] == b


def test_vicreg_regularizer_adds_to_loss() -> None:
    cfg = copy.deepcopy(tiny_config())
    cfg["regularizer"] = {
        "enabled": True,
        "type": "vicreg",
        "var_coeff": 1.0,
        "cov_coeff": 0.04,
        "projector": [128, 128],
    }

    model = IJEPA.from_config(cfg)
    assert model.reg_enabled and model.projector is not None
    model.train()  # BatchNorm in projector needs >1 sample

    b = 8
    images = torch.randn(b, 3, 32, 32)
    grid = cfg["data"]["img_size"] // cfg["data"]["patch_size"]
    collator = IJEPAMaskCollator(grid_size=grid, fixed_context_patches=32, fixed_target_patches=16)
    masks = collator(b)
    ctx = torch.stack(masks.context_indices)
    tgt = torch.stack(masks.target_indices)

    out = model(images, ctx, tgt)
    assert "reg_var" in out and "reg_cov" in out
    # total loss must include the regularizer on top of the prediction loss
    assert out["loss"].item() >= out["pred_loss"].item()
    out["loss"].backward()
    proj_grads = [p.grad for p in model.projector.parameters() if p.grad is not None]
    assert len(proj_grads) > 0


def test_config_base_inheritance() -> None:
    cfg = load_config("configs/image_jepa_cifar10_vicreg.yaml")
    # inherited from baseline
    assert cfg["encoder"]["embed_dim"] == 384
    assert cfg["data"]["patch_size"] == 4
    # overridden by the child config
    assert cfg["regularizer"]["enabled"] is True
    assert cfg["train"]["run_dir"] == "runs/cifar10_vicreg_v2"
    assert "_base_" not in cfg


def test_mask_collator_fixed_sizes() -> None:
    collator = IJEPAMaskCollator(grid_size=8, fixed_context_patches=32, fixed_target_patches=16)
    masks = collator(batch_size=4)
    for ctx, tgt in zip(masks.context_indices, masks.target_indices):
        assert ctx.numel() == 32
        assert tgt.numel() == 16
        assert len(set(ctx.tolist()) & set(tgt.tolist())) == 0


def test_mask_collator_ranges_vary_and_stack() -> None:
    collator = IJEPAMaskCollator(
        grid_size=8,
        context_patches_range=(24, 40),
        target_patches_range=(10, 22),
    )

    seen_ctx_sizes: set[int] = set()
    seen_tgt_sizes: set[int] = set()
    for _ in range(40):
        masks = collator(batch_size=4)
        # Uniform length within a batch -> stackable -> no overlap.
        ctx_sizes = {c.numel() for c in masks.context_indices}
        tgt_sizes = {t.numel() for t in masks.target_indices}
        assert len(ctx_sizes) == 1
        assert len(tgt_sizes) == 1
        torch.stack(masks.context_indices)
        torch.stack(masks.target_indices)
        for ctx, tgt in zip(masks.context_indices, masks.target_indices):
            assert 24 <= ctx.numel() <= 40
            assert 10 <= tgt.numel() <= 22
            assert len(set(ctx.tolist()) & set(tgt.tolist())) == 0
        seen_ctx_sizes |= ctx_sizes
        seen_tgt_sizes |= tgt_sizes

    # Sizes should vary across batches.
    assert len(seen_ctx_sizes) > 1
    assert len(seen_tgt_sizes) > 1
