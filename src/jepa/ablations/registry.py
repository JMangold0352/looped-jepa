from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AblationVariant:
    """One train/eval configuration within an ablation suite."""

    name: str
    description: str
    predictor_overrides: dict[str, Any] = field(default_factory=dict)
    train_overrides: dict[str, Any] = field(default_factory=dict)
    existing_checkpoint: str | None = None


@dataclass(frozen=True)
class AblationSuite:
    """A group of variants that isolate a single design choice."""

    key: str
    title: str
    variants: tuple[AblationVariant, ...]


def _paths(slug: str) -> dict[str, str]:
    return {
        "run_dir": f"runs/ablations/{slug}",
        "checkpoint_dir": f"checkpoints/ablations/{slug}",
    }


ABLATION_SUITES: dict[str, AblationSuite] = {
    "loop_count": AblationSuite(
        key="loop_count",
        title="Loop count (1 vs 2 vs 4)",
        variants=(
            AblationVariant(
                name="loops_1",
                description="Single predictor pass (no recurrence, no exit gate).",
                predictor_overrides={"max_loops": 1, "use_exit_gate": False},
                train_overrides=_paths("loop_count/loops_1"),
            ),
            AblationVariant(
                name="loops_2",
                description="Two recurrence steps with exit gate (current default).",
                predictor_overrides={"max_loops": 2, "use_exit_gate": True},
                train_overrides=_paths("loop_count/loops_2"),
                existing_checkpoint="checkpoints/baseline_v3_looped/latest.pt",
            ),
            AblationVariant(
                name="loops_4",
                description="Four recurrence steps with exit gate.",
                predictor_overrides={"max_loops": 4, "use_exit_gate": True},
                train_overrides=_paths("loop_count/loops_4"),
            ),
        ),
    ),
    "entropy": AblationSuite(
        key="entropy",
        title="Exit-gate entropy regularization (on vs off)",
        variants=(
            AblationVariant(
                name="entropy_on",
                description="Exit gate + entropy penalty (beta=0.01).",
                predictor_overrides={"max_loops": 2, "use_exit_gate": True},
                train_overrides={**_paths("entropy/entropy_on"), "exit_entropy_beta": 0.01},
            ),
            AblationVariant(
                name="entropy_off",
                description="Exit gate without entropy penalty (beta=0).",
                predictor_overrides={"max_loops": 2, "use_exit_gate": True},
                train_overrides={**_paths("entropy/entropy_off"), "exit_entropy_beta": 0.0},
            ),
        ),
    ),
    "sandwich_norm": AblationSuite(
        key="sandwich_norm",
        title="Predictor normalization (LayerNorm vs sandwich RMSNorm)",
        variants=(
            AblationVariant(
                name="layernorm",
                description="Standard LayerNorm before attention and FFN.",
                predictor_overrides={
                    "max_loops": 2,
                    "use_exit_gate": True,
                    "norm": "layer",
                    "sandwich_norm": False,
                },
                train_overrides=_paths("sandwich_norm/layernorm"),
            ),
            AblationVariant(
                name="sandwich_rms",
                description="RMSNorm before and after each sub-layer (sandwich layout).",
                predictor_overrides={
                    "max_loops": 2,
                    "use_exit_gate": True,
                    "norm": "rms",
                    "sandwich_norm": True,
                },
                train_overrides=_paths("sandwich_norm/sandwich_rms"),
            ),
        ),
    ),
}


def suite_names() -> list[str]:
    return list(ABLATION_SUITES.keys())
