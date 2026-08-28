"""Registry and download helpers for released pretrained checkpoints."""
from __future__ import annotations

import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

def find_repo_root() -> Path:
    """Locate repository root (contains ``released_weights/`` and ``pyproject.toml``)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "released_weights").is_dir():
            return parent
    return here.parents[3]


ROOT = find_repo_root()
URLS_YAML = ROOT / "released_weights" / "urls.yaml"

PLACEHOLDER = "PLACEHOLDER_URL"

# Keys allowed to be missing when loading ckpt["model"] with strict=False:
# - Optimizer / training metadata (never in inference ckpt)
# - Optional probe or logging buffers if present in some exports
ALLOWED_MISSING_KEY_PREFIXES = ("optimizer", "probe", "scaler")


@dataclass(frozen=True)
class ReleasedWeight:
    """Metadata for one published checkpoint."""

    key: str
    config: str
    checkpoint: str
    tuned_top1: float
    feat_std: float
    params: int
    description: str

    @property
    def config_path(self) -> Path:
        return ROOT / self.config

    @property
    def checkpoint_path(self) -> Path:
        return ROOT / self.checkpoint


RELEASED_WEIGHTS: dict[str, ReleasedWeight] = {
    "baseline_v3": ReleasedWeight(
        key="baseline_v3",
        config="configs/image_jepa_cifar10_v3.yaml",
        checkpoint="checkpoints/baseline_v3/latest.pt",
        tuned_top1=77.23,
        feat_std=0.1607,
        params=9_816_960,
        description="v3 baseline I-JEPA (non-looped predictor)",
    ),
    "looped_v3": ReleasedWeight(
        key="looped_v3",
        config="configs/image_jepa_cifar10_v3_looped.yaml",
        checkpoint="checkpoints/baseline_v3_looped/latest.pt",
        tuned_top1=75.13,
        feat_std=0.1450,
        params=9_816_960,
        description="Default looped predictor (2-loop, LayerNorm, exit gate)",
    ),
    "sandwich_rmsnorm": ReleasedWeight(
        key="sandwich_rmsnorm",
        config="configs/image_jepa_cifar10_v3_looped_sandwich_rms.yaml",
        checkpoint="checkpoints/ablations/sandwich_norm/sandwich_rms/latest.pt",
        tuned_top1=78.28,
        feat_std=0.0432,
        params=9_816_960,
        description="Looped predictor with sandwich RMSNorm (best ablation)",
    ),
}

_ENV_PRIMARY = {
    "baseline_v3": "LOOPED_JEPA_WEIGHT_URL_BASELINE_V3",
    "looped_v3": "LOOPED_JEPA_WEIGHT_URL_LOOPED_V3",
    "sandwich_rmsnorm": "LOOPED_JEPA_WEIGHT_URL_SANDWICH_RMSNORM",
}

_ENV_GDRIVE = {
    "baseline_v3": "LOOPED_JEPA_WEIGHT_GDRIVE_BASELINE_V3",
    "looped_v3": "LOOPED_JEPA_WEIGHT_GDRIVE_LOOPED_V3",
    "sandwich_rmsnorm": "LOOPED_JEPA_WEIGHT_GDRIVE_SANDWICH_RMSNORM",
}


class WeightDownloadError(RuntimeError):
    """Raised when a checkpoint URL is missing or download fails."""


def list_released_weights() -> list[str]:
    return list(RELEASED_WEIGHTS.keys())


def get_released_weight(name: str) -> ReleasedWeight:
    try:
        return RELEASED_WEIGHTS[name]
    except KeyError as exc:
        known = ", ".join(RELEASED_WEIGHTS)
        raise KeyError(f"Unknown model {name!r}. Choose from: {known}") from exc


def _load_url_map() -> dict[str, dict[str, str]]:
    if not URLS_YAML.is_file():
        return {}
    with URLS_YAML.open() as f:
        data = yaml.safe_load(f) or {}
    return {str(k): dict(v or {}) for k, v in data.items()}


def _is_placeholder(url: str | None) -> bool:
    if not url:
        return True
    u = url.strip()
    return not u or u == PLACEHOLDER or u.upper().startswith("TODO")


def resolve_download_urls(name: str) -> tuple[str | None, str | None]:
    """Return (huggingface_or_primary_url, google_drive_fallback)."""
    spec = get_released_weight(name)
    env_primary = os.environ.get(_ENV_PRIMARY.get(spec.key, ""), "").strip()
    env_gdrive = os.environ.get(_ENV_GDRIVE.get(spec.key, ""), "").strip()

    yaml_entry = _load_url_map().get(spec.key, {})
    hf = env_primary or yaml_entry.get("huggingface") or yaml_entry.get("url")
    gdrive = env_gdrive or yaml_entry.get("google_drive")

    hf = None if _is_placeholder(hf) else hf
    gdrive = None if _is_placeholder(gdrive) else gdrive
    return hf, gdrive


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out)
        tmp.replace(dest)
    except (urllib.error.URLError, OSError) as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise WeightDownloadError(f"Failed to download {url} -> {dest}: {exc}") from exc


def download_released_weight(
    name: str,
    *,
    force: bool = False,
    root: Path | None = None,
) -> Path:
    """Download one released checkpoint if URLs are configured. Idempotent."""
    spec = get_released_weight(name)
    dest = (root or ROOT) / spec.checkpoint
    if dest.is_file() and not force:
        return dest.resolve()

    hf, gdrive = resolve_download_urls(name)
    if hf is None and gdrive is None:
        raise WeightDownloadError(
            f"No download URL configured for {name!r}. "
            f"Set URLs in {URLS_YAML.relative_to(ROOT)} or env vars "
            f"{_ENV_PRIMARY.get(name)} / {_ENV_GDRIVE.get(name)}. "
            f"See released_weights/README.md."
        )

    errors: list[str] = []
    for label, url in (("huggingface", hf), ("google_drive", gdrive)):
        if url is None:
            continue
        try:
            _download_file(url, dest)
            return dest.resolve()
        except WeightDownloadError as exc:
            errors.append(f"{label}: {exc}")

    raise WeightDownloadError(
        f"All download attempts failed for {name!r}.\n" + "\n".join(errors)
    )


def ensure_checkpoint(
    name: str,
    *,
    pretrained: bool,
    checkpoint: str | Path | None,
    root: Path | None = None,
) -> Path:
    """Resolve checkpoint path; optionally download when ``pretrained=True``."""
    spec = get_released_weight(name)
    repo = root or ROOT

    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path.resolve()

    path = repo / spec.checkpoint
    if path.is_file():
        return path.resolve()

    if not pretrained:
        raise FileNotFoundError(
            f"Checkpoint missing for {name!r}: {path}\n"
            f"Train with configs/{Path(spec.config).name} or download:\n"
            f"  ./scripts/download_weights.sh {name}"
        )

    return download_released_weight(name, root=repo)


def load_checkpoint_state(
    checkpoint_path: Path,
    *,
    map_location: Any = "cpu",
) -> dict[str, Any]:
    import torch

    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise WeightDownloadError(f"Unexpected checkpoint format in {checkpoint_path}")


def load_state_into_model(model: Any, state: dict[str, Any]) -> None:
    """Load weights; strict=False with documented allowed missing prefixes."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [
        k for k in missing
        if not any(k.startswith(p) for p in ALLOWED_MISSING_KEY_PREFIXES)
    ]
    if bad_missing:
        raise RuntimeError(
            "Checkpoint missing required keys (strict=False): "
            + ", ".join(bad_missing[:8])
            + (" ..." if len(bad_missing) > 8 else "")
        )
    if unexpected:
        # Log but do not fail: exports may include extra buffers.
        print(f"  [load_ijepa] ignoring unexpected keys: {unexpected[:5]}"
              + (" ..." if len(unexpected) > 5 else ""))


def checkpoint_status(name: str, *, root: Path | None = None) -> dict[str, Any]:
    """Return registry metadata plus whether the local checkpoint file exists."""
    spec = get_released_weight(name)
    path = (root or ROOT) / spec.checkpoint
    hf, gdrive = resolve_download_urls(name)
    return {
        "name": name,
        "checkpoint": str(path.relative_to(root or ROOT)),
        "present": path.is_file(),
        "huggingface_url": hf,
        "google_drive_url": gdrive,
        "urls_configured": hf is not None or gdrive is not None,
    }


def download_all(
    names: list[str] | None = None,
    *,
    force: bool = False,
    root: Path | None = None,
) -> list[Path]:
    """Download multiple registry entries; default is all three released models."""
    keys = names or list_released_weights()
    return [download_released_weight(k, force=force, root=root) for k in keys]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``scripts/download_weights.sh``."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Download released I-JEPA checkpoints into checkpoints/"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print registry entries and local checkpoint status",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when the checkpoint file already exists",
    )
    parser.add_argument(
        "models",
        nargs="*",
        help="Registry keys (default: all). Example: baseline_v3 looped_v3",
    )
    args = parser.parse_args(argv)

    if args.list:
        print(f"Registry ({len(RELEASED_WEIGHTS)} models) — root: {ROOT}\n")
        for key in list_released_weights():
            st = checkpoint_status(key)
            url_note = "URLs set" if st["urls_configured"] else "PLACEHOLDER (set urls.yaml or env)"
            present = "present" if st["present"] else "missing"
            spec = get_released_weight(key)
            print(
                f"  {key:18}  {present:7}  top1={spec.tuned_top1:.2f}%  "
                f"feat_std={spec.feat_std:.4f}  {url_note}"
            )
            print(f"    -> {st['checkpoint']}")
        return 0

    targets = args.models or list_released_weights()
    for key in targets:
        try:
            path = download_released_weight(key, force=args.force)
            print(f"OK  {key} -> {path}")
        except (KeyError, WeightDownloadError) as exc:
            print(f"ERROR  {key}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
