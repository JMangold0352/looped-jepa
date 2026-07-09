"""Tests for CLI helpers."""
from __future__ import annotations

import pytest

from jepa.utils.cli import require_file


def test_require_file_returns_resolved_path(tmp_path):
    f = tmp_path / "ok.yaml"
    f.write_text("x: 1\n")
    assert require_file(f, label="Config") == f.resolve()


def test_require_file_exits_when_missing(tmp_path):
    missing = tmp_path / "nope.pt"
    with pytest.raises(SystemExit) as exc:
        require_file(missing, label="Checkpoint", hint="train first")
    assert "Checkpoint not found" in str(exc.value)
    assert "train first" in str(exc.value)
