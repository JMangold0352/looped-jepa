#!/usr/bin/env python3
"""I-JEPA v3 interactive demo: baseline vs looped predictor.

Run locally:
    uv sync --extra demo && source .venv/bin/activate
    python app.py                      # then open http://127.0.0.1:7860

This file doubles as the Hugging Face Spaces entry point.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import random

import gradio as gr
import torch

from demo import inference as inf
from demo import render as rd

CTX = inf.load_context()

# Checkpoint was trained with max_loops=2; 4-loop runs extrapolate the weight-shared stack.
LOOP_TRAINED_MAX = 2
LOOP_EXTRAPOLATION_NOTE = (
    f"> **Note:** This checkpoint was trained with `max_loops={LOOP_TRAINED_MAX}`. "
    f"Selecting **4 loops** runs two extra refinement steps beyond training, useful to "
    f"explore, but exit-gate stats only cover loops 1–{LOOP_TRAINED_MAX} and results may "
    f"be less reliable."
)

CUSTOM_CSS = """
.gradio-container {max-width: 1180px !important; margin: auto;}
#hero {text-align:center; padding: 8px 0 2px 0;}
#hero h1 {font-size: 2.05rem; margin-bottom: 2px; font-weight: 750;
  background: linear-gradient(90deg,#4C78A8,#F58518);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
#hero p {color:#5b6470; font-size:1.02rem; margin-top:2px;}
.badges {text-align:center; margin: 4px 0 10px 0;}
.badges span {display:inline-block; background:#f1f4f9; color:#39424e;
  border:1px solid #e2e8f0; border-radius:999px; padding:3px 12px;
  font-size:.8rem; margin:0 4px;}
.card {border:1px solid #e6e9ef; border-radius:16px; padding:14px 16px;
  background:#ffffff; box-shadow:0 1px 3px rgba(16,24,40,.05);}
.card h3 {margin:0 0 8px 0; font-size:1.05rem;}
.baseline-accent {border-top:4px solid #4C78A8;}
.looped-accent {border-top:4px solid #F58518;}
footer {visibility:hidden;}
"""

EMPTY_GALLERY: list = []


def _stats_markdown(base_res, loop_res, loops, probe_base, probe_loop) -> str:
    lines = ["### Summary", ""]
    lines.append(f"| Metric | Baseline | Looped ({loops} loop{'s' if loops > 1 else ''}) |")
    lines.append("| --- | ---: | ---: |")
    lines.append(
        f"| Final cosine to teacher | {base_res['final_cosine']:.3f} | {loop_res['final_cosine']:.3f} |"
    )
    per_loop = ", ".join(f"{c:.3f}" for c in loop_res["per_loop_cosine"])
    lines.append(f"| Per-loop cosine | n/a | {per_loop} |")
    if loop_res["expected_loops"] is not None:
        lines.append(f"| Expected exit depth | n/a | {loop_res['expected_loops']:.2f} loops |")
    if loop_res["exit_probs"] is not None:
        ep = ", ".join(f"{p:.2f}" for p in loop_res["exit_probs"])
        lines.append(f"| Exit-gate P(exit) | n/a | {ep} |")

    if loops > LOOP_TRAINED_MAX:
        lines += ["", LOOP_EXTRAPOLATION_NOTE, ""]

    if probe_base is not None:
        lines += ["", "### Linear-probe prediction (frozen features)", ""]
        lines.append("| Rank | Baseline | Looped |")
        lines.append("| --- | --- | --- |")
        for i in range(max(len(probe_base), len(probe_loop))):
            b = f"{probe_base[i][0]} ({probe_base[i][1]*100:.1f}%)" if i < len(probe_base) else ""
            l = f"{probe_loop[i][0]} ({probe_loop[i][1]*100:.1f}%)" if i < len(probe_loop) else ""
            lines.append(f"| {i+1} | {b} | {l} |")
    return "\n".join(lines)


def _loop_caveat(loops) -> str:
    """Show an inline warning when the user selects loops beyond training depth."""
    if int(loops) > LOOP_TRAINED_MAX:
        return LOOP_EXTRAPOLATION_NOTE
    return ""


def run(image, loops, mode, seed):
    if not CTX.available:
        msg = f"**Demo unavailable.** {CTX.message}\n\nTrain or place checkpoints, then relaunch."
        return (None, EMPTY_GALLERY, EMPTY_GALLERY, EMPTY_GALLERY, None, None, msg, "")
    if image is None:
        note = "Upload an image and press **Run comparison** to begin."
        return (None, EMPTY_GALLERY, EMPTY_GALLERY, EMPTY_GALLERY, None, None, note, _loop_caveat(loops))

    loops = int(loops)
    tensor = CTX.transform(image.convert("RGB")).unsqueeze(0).to(CTX.device)
    ctx_idx, tgt_idx = inf._sample_mask(CTX, int(seed))

    base_res = inf.run_model(CTX.baseline, tensor, ctx_idx, tgt_idx, CTX.grid_size, loops=1)
    loop_res = inf.run_model(CTX.looped, tensor, ctx_idx, tgt_idx, CTX.grid_size, loops=loops)

    img_t = tensor[0]
    original = rd.to_pil(rd.denorm(img_t))

    baseline_gallery = [
        (rd.masked_context(img_t, ctx_idx[0], CTX.grid_size), "Masked context (input)"),
        (rd.prediction_quality(img_t, tgt_idx[0], base_res["cos_per_patch"], CTX.grid_size), "Prediction quality"),
        (rd.attention_overlay(img_t, base_res["attn_grids"][0], CTX.grid_size), "Predictor attention"),
    ]
    looped_gallery = [
        (rd.masked_context(img_t, ctx_idx[0], CTX.grid_size), "Masked context (input)"),
        (rd.prediction_quality(img_t, tgt_idx[0], loop_res["cos_per_patch"], CTX.grid_size), f"Prediction quality (loop {loops})"),
        (rd.attention_overlay(img_t, loop_res["attn_grids"][-1], CTX.grid_size), f"Attention (loop {loops})"),
    ]
    evolution = [
        (rd.attention_overlay(img_t, g, CTX.grid_size), f"Loop {i + 1}")
        for i, g in enumerate(loop_res["attn_grids"])
    ]

    cosine_img = rd.cosine_curve(base_res["final_cosine"], loop_res["per_loop_cosine"])
    if loop_res["exit_probs"] is not None:
        exit_img = rd.exit_gate_bars(loop_res["exit_probs"])
    else:
        exit_img = rd.placeholder("Exit gate disabled\nfor this checkpoint")

    probe_base = probe_loop = None
    if mode == "Linear probe (CIFAR-10)":
        probe_base = inf.probe_predict(CTX, "baseline", tensor)
        probe_loop = inf.probe_predict(CTX, "looped", tensor)

    stats = _stats_markdown(base_res, loop_res, loops, probe_base, probe_loop)
    return (original, baseline_gallery, looped_gallery, evolution, cosine_img, exit_img, stats, _loop_caveat(loops))


def resample_seed():
    return random.randint(0, 10_000)


HOW_IT_WORKS = """
**I-JEPA** learns representations by predicting the *latent* embeddings of masked target
patches (red) from a visible context region (bright), scored against an EMA teacher, with no
pixel reconstruction. The **looped predictor** reuses the same predictor block stack for
multiple refinement steps (weight-shared recurrence) and uses a learned **exit gate** to
decide, per sample, how many loops to spend.

- **Masked context**: the patches the encoder actually sees.
- **Prediction quality**: each target patch tinted red→green by cosine similarity between
  the predicted and teacher latent (greener = better).
- **Predictor attention**: where the predictor looks in the context; for the looped model
  you can watch it evolve loop by loop.
- **Per-loop cosine**: how much each extra loop refines the prediction.
- **Exit gate**: the learned probability of stopping at each loop.
"""

RUN_LOCALLY = """
```bash
git clone https://github.com/JMangold0352/looped-jepa.git && cd looped-jepa
uv sync --extra demo          # or: pip install -e ".[demo]"
source .venv/bin/activate
python app.py                 # open http://127.0.0.1:7860
```

**Hugging Face Spaces:** create a Gradio Space, push this repo (`app.py` is the entry
point), and include `checkpoints/baseline_v3/latest.pt` and
`checkpoints/baseline_v3_looped/latest.pt` (or load them from the Hub). `requirements.txt`
lists the runtime deps.
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="I-JEPA v3: Baseline vs Looped") as demo:
        gr.HTML(
            '<div id="hero"><h1>I-JEPA v3 · Baseline vs Looped Predictor</h1>'
            '<p>Self-supervised vision under a 10M-parameter budget: watch the predictor '
            'refine its guess loop by loop.</p></div>'
            '<div class="badges"><span>ViT encoder · 384×5</span>'
            '<span>Baseline 77.2% probe</span><span>Weight-shared recurrence</span>'
            '<span>Learned exit gate</span></div>'
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label="Input image", height=260)
                loops_in = gr.Radio(
                    choices=[1, 2, 4],
                    value=2,
                    label="Looped predictor: number of loops",
                    info=f"Checkpoint trained with max_loops={LOOP_TRAINED_MAX}. "
                    "4 loops extrapolates beyond training.",
                )
                loop_caveat_out = gr.Markdown(value="", visible=True)
                mode_in = gr.Radio(
                    choices=["Visualization only", "Linear probe (CIFAR-10)"],
                    value="Visualization only",
                    label="Mode",
                    info="Linear-probe mode fits a frozen-feature classifier on first use (~1 min).",
                )
                with gr.Row():
                    seed_in = gr.Number(value=42, precision=0, label="Mask seed", scale=2)
                    resample_btn = gr.Button("🎲 Resample mask", scale=1)
                run_btn = gr.Button("Run comparison", variant="primary")
            with gr.Column(scale=1):
                original_out = gr.Image(label="Original (resized to 32×32)", height=260)
                stats_out = gr.Markdown("Upload an image and press **Run comparison**.")

        with gr.Row():
            with gr.Column():
                gr.HTML('<div class="card baseline-accent"><h3>🔵 v3 Baseline (single pass)</h3></div>')
                baseline_gallery = gr.Gallery(label="Baseline", columns=3, height=200, object_fit="contain")
            with gr.Column():
                gr.HTML('<div class="card looped-accent"><h3>🟠 v3 Looped predictor</h3></div>')
                looped_gallery = gr.Gallery(label="Looped", columns=3, height=200, object_fit="contain")

        gr.Markdown("#### Attention evolution across loops")
        evolution_gallery = gr.Gallery(label="Loop-by-loop predictor attention", columns=4, height=200, object_fit="contain")

        with gr.Row():
            cosine_out = gr.Image(label="Per-loop prediction quality", height=300)
            exit_out = gr.Image(label="Exit-gate behavior", height=300)

        with gr.Accordion("How it works", open=False):
            gr.Markdown(HOW_IT_WORKS)
        with gr.Accordion("Run locally / deploy", open=False):
            gr.Markdown(RUN_LOCALLY)

        outputs = [
            original_out, baseline_gallery, looped_gallery, evolution_gallery,
            cosine_out, exit_out, stats_out, loop_caveat_out,
        ]
        run_btn.click(run, inputs=[image_in, loops_in, mode_in, seed_in], outputs=outputs)
        loops_in.change(_loop_caveat, inputs=loops_in, outputs=loop_caveat_out)
        resample_btn.click(resample_seed, outputs=seed_in).then(
            run, inputs=[image_in, loops_in, mode_in, seed_in], outputs=outputs
        )
    return demo


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Launch the I-JEPA v3 demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    build_demo().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
