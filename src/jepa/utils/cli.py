"""Small helpers for command-line entry points."""
from __future__ import annotations

from pathlib import Path


def require_file(path: str | Path, *, label: str, hint: str | None = None) -> Path:
    """Exit with a clear message if ``path`` is not a readable file."""
    p = Path(path)
    if p.is_file():
        return p.resolve()

    resolved = p.resolve()
    lines = [f"Error: {label} not found: {p}"]
    if resolved != p:
        lines.append(f"  Resolved path: {resolved}")
    if hint:
        lines.append("")
        lines.append(hint)
    raise SystemExit("\n".join(lines))
