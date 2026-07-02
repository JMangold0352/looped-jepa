# Interactive Demo

An interactive Gradio demo that compares the v3 baseline and the v3 looped predictor side by side on any uploaded image, and shows how the looped predictor refines its prediction loop by loop

![demo](../visualizations/figures/01_mask_reconstruction.png)

## Features

- **Upload any image** (CIFAR-style or higher res; it's resized to 32×32).
- **Side-by-side comparison**: baseline (single pass) vs looped predictor.
- **Toggle loops** (1 / 2 / 4) for the looped model. The shipped checkpoint was trained
  with `max_loops=2`; selecting **4** extrapolates beyond training (the UI shows a caveat).
- **Visualizations**: original, masked context, per-patch prediction quality
  (cosine to the EMA teacher), and predictor attention, including how attention
  **evolves across loops**.
- **Adaptive-compute stats**: per-loop cosine curve, learned **exit-gate** probabilities,
  and expected exit depth.
- **Two modes**: *Visualization only* or *Linear probe (CIFAR-10)* class predictions on
  frozen features.
- **Resample mask** to see prediction under a different context/target split.

## Run locally

```bash
uv sync --extra demo        # or: pip install -e ".[demo]"
source .venv/bin/activate
python app.py               # open http://127.0.0.1:7860
```

Options: `python app.py --port 7861 --share` (`--share` creates a public Gradio link).

## Layout

```
app.py            # Gradio Blocks UI + entry point (also the HF Spaces entry point)
demo/inference.py # model loading + per-loop inference (cosine, attention, exit stats)
demo/render.py    # tensor/stats -> PIL images (panels + charts)
requirements.txt  # runtime deps for Hugging Face Spaces
```

## Deploy to Hugging Face Spaces

1. Create a new **Gradio** Space.
2. Push this repository (the Space uses `app.py` as its entry point and installs
   `requirements.txt`).
3. Make the checkpoints available at
   `checkpoints/baseline_v3/latest.pt` and `checkpoints/baseline_v3_looped/latest.pt`
   (commit with Git LFS, or download from the Hub at startup).

If checkpoints are missing the app still launches and shows a clear message instead of
crashing.
