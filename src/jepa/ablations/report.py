from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def _cell(value: float | None, fmt: str) -> str:
    return format(value, fmt) if value is not None else "n/a"


def variant_row(entry: dict[str, Any]) -> str:
    if entry.get("status") == "missing_checkpoint":
        return f"| {entry['name']} | n/a | n/a | n/a | n/a | n/a | (not trained) |"
    loop = entry.get("loop_usage", {})
    stability = entry.get("training_stability", {})
    return (
        f"| {entry['name']} | {_fmt_pct(entry['top1_accuracy'])} | {entry['feat_std']:.4f} | "
        f"{_cell(loop.get('mean_loops_used'), '.2f')} | "
        f"{_cell(stability.get('tail_loss_std'), '.3f')} | "
        f"{_cell(stability.get('min_feat_std'), '.4f')} |"
    )


def render_suite_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['title']}",
        "",
        f"_Base config: `{payload['base_config']}`_",
        "",
        "| Variant | Tuned top-1 | feat_std | Mean loops (val) | Tail loss σ | Min feat_std (train) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in payload["variants"]:
        lines.append(variant_row(entry))

    lines.extend(["", "## Loop usage detail", ""])
    for entry in payload["variants"]:
        if entry.get("status") == "missing_checkpoint":
            lines.append(f"### {entry['name']}")
            lines.append("")
            lines.append("_Checkpoint not trained yet._")
            lines.append("")
            continue
        loop = entry.get("loop_usage", {})
        lines.append(f"### {entry['name']}")
        lines.append("")
        lines.append(f"- Mean loops used: **{loop.get('mean_loops_used', 0):.3f}**")
        if loop.get("exit_prob_per_loop"):
            probs = ", ".join(f"L{i+1}={p:.3f}" for i, p in enumerate(loop["exit_prob_per_loop"]))
            lines.append(f"- Mean exit prob per loop: {probs}")
        if loop.get("loops_used_histogram"):
            lines.append(f"- Histogram (rounded loops): `{loop['loops_used_histogram']}`")
        lines.append("")

    return "\n".join(lines)


def render_summary_markdown(all_payloads: list[dict[str, Any]]) -> str:
    lines = [
        "# Looped Predictor Ablation Summary",
        "",
        "All runs use the v3 training recipe (300 epochs, RandAugment, tuned linear probe).",
        "",
    ]
    for payload in all_payloads:
        lines.append(f"## {payload['title']}")
        lines.append("")
        lines.append(
            "| Variant | Tuned top-1 | feat_std | Mean loops |"
        )
        lines.append("| --- | --- | --- | --- |")
        for entry in payload["variants"]:
            if entry.get("status") == "missing_checkpoint":
                lines.append(f"| {entry['name']} | n/a | n/a | n/a |")
                continue
            loop = entry.get("loop_usage", {})
            lines.append(
                f"| {entry['name']} | {_fmt_pct(entry['top1_accuracy'])} | "
                f"{entry['feat_std']:.4f} | {loop.get('mean_loops_used', 0):.2f} |"
            )
        lines.append("")

    return "\n".join(lines)


def save_suite_results(payload: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    key = payload["suite"]
    json_path = out_dir / f"{key}.json"
    md_path = out_dir / f"{key}.md"
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(render_suite_markdown(payload))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def save_summary(all_payloads: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"suites": all_payloads}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "summary.md").write_text(render_summary_markdown(all_payloads))
    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"Wrote {out_dir / 'summary.md'}")
