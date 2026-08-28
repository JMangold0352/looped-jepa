"""Tests for released-weight registry and download helpers."""
from __future__ import annotations

from pathlib import Path

import pytest

from jepa.utils.weights import (
    RELEASED_WEIGHTS,
    WeightDownloadError,
    checkpoint_status,
    download_released_weight,
    ensure_checkpoint,
    get_released_weight,
    list_released_weights,
    resolve_download_urls,
)


def test_registry_has_three_models():
    keys = list_released_weights()
    assert keys == ["baseline_v3", "looped_v3", "sandwich_rmsnorm"]
    assert len(RELEASED_WEIGHTS) == 3


def test_registry_metrics_match_documentation():
    base = get_released_weight("baseline_v3")
    loop = get_released_weight("looped_v3")
    sand = get_released_weight("sandwich_rmsnorm")
    assert base.tuned_top1 == pytest.approx(77.23)
    assert loop.tuned_top1 == pytest.approx(75.13)
    assert sand.tuned_top1 == pytest.approx(78.28)
    assert base.feat_std == pytest.approx(0.1607)
    assert loop.feat_std == pytest.approx(0.1450)
    assert sand.feat_std == pytest.approx(0.0432)
    assert base.params == 9_816_960


def test_unknown_model_raises():
    with pytest.raises(KeyError, match="Unknown model"):
        get_released_weight("not_a_model")


def test_placeholder_urls_not_configured():
    for key in list_released_weights():
        hf, gdrive = resolve_download_urls(key)
        assert hf is None
        assert gdrive is None


@pytest.mark.parametrize("key", list_released_weights())
def test_download_raises_on_placeholder(key: str, tmp_path: Path):
    with pytest.raises(WeightDownloadError, match="No download URL configured"):
        download_released_weight(key, root=tmp_path, force=True)


def test_ensure_checkpoint_missing_without_pretrained(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Checkpoint missing"):
        ensure_checkpoint("baseline_v3", pretrained=False, checkpoint=None, root=tmp_path)


def test_checkpoint_status_reports_paths():
    st = checkpoint_status("baseline_v3")
    assert st["name"] == "baseline_v3"
    assert st["checkpoint"] == "checkpoints/baseline_v3/latest.pt"
    assert st["urls_configured"] is False


@pytest.mark.skipif(
    not Path("checkpoints/baseline_v3/latest.pt").is_file(),
    reason="local baseline checkpoint not present",
)
def test_load_ijepa_baseline_local_checkpoint():
    from jepa import load_ijepa

    model = load_ijepa("baseline_v3", pretrained=False, device="cpu")
    assert model.training is False
    assert model.num_trainable_params() == 9_816_960
