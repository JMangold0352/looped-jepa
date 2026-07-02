from __future__ import annotations

import torch

from jepa.models.looped_predictor import expected_loops_from_exit_probs


def test_expected_loops_deterministic_exit() -> None:
    exit_probs = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    expected = expected_loops_from_exit_probs(exit_probs)
    assert torch.allclose(expected, torch.tensor([1.0, 1.0]))


def test_expected_loops_never_exit() -> None:
    exit_probs = torch.tensor([[0.0, 0.0]])
    expected = expected_loops_from_exit_probs(exit_probs)
    assert torch.allclose(expected, torch.tensor([2.0]))


def test_expected_loops_mixed() -> None:
    exit_probs = torch.tensor([[0.5, 0.5]])
    expected = expected_loops_from_exit_probs(exit_probs)
    assert abs(expected.item() - 1.5) < 1e-5
