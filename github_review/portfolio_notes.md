# Portfolio notes

*Why this project demonstrates research taste, implementation skill, and relevance to defense / autonomous systems roles.*

---

## Elevator pitch (30 seconds)

I built a **complete, reproducible Image-JEPA stack** on CIFAR-10 under a **10M-parameter**
budget, then asked a concrete research question: does a **weight-shared recurrent predictor**
with a learned **exit gate** improve latent world modeling? I ran the experiment honestly
(the default looped model **loses 2.1 pp** in-domain) and followed the negative result with
**seven full ablations**, **publication figures**, **aerial transfer**, and an **interactive
demo**. The best variant (sandwich-RMSNorm) **beats the baseline by +1.05 pp**; the looped
encoder **transfers +4 pp** to aerial imagery. The artifact chain is end-to-end: train →
ablate → visualize → transfer → demo → technical report.

---

## Research taste

What good research looks like here:

| Signal | Evidence in this repo |
| --- | --- |
| **Clear hypothesis** | Recurrent predictor at fixed param count; adaptive depth via exit gate |
| **Controlled comparison** | Same encoder, aug, schedule; only predictor architecture changes |
| **Honest reporting** | −2.10 pp headline loss reported without spin; postmortem in the technical report |
| **Ablation discipline** | Three suites (loop count, exit entropy, normalization), 300 epochs each |
| **Negative results documented** | v2 collapse, v4/v5 two-view failures in REPORT.md, showing what *doesn't* work |
| **Mechanistic follow-up** | Per-loop attention, cosine refinement, exit distributions, not just a single scalar |
| **Surprise reframed** | Sandwich RMSNorm (+78.28%) and transfer (+4 pp) emerge from structured inquiry |

A hiring manager or PI should see: **I don't cherry-pick**. I instrument experiments so the
story survives scrutiny.

---

## Implementation skill

| Layer | What was built |
| --- | --- |
| **Core ML** | ViT encoder, EMA teacher, I-JEPA masking, smooth-L1 latent loss, tuned linear probe |
| **Novel component** | `LoopedPredictor`: shared-weight recurrence, exit gate, entropy regularizer |
| **Training at scale** | 300-epoch runs, auto-resume, feat_std collapse diagnostics, ablation orchestration |
| **Eval harness** | Probe sweeps, loop-usage metrics, transfer experiments, scratch baselines |
| **Visualization** | 9+ publication figures at 300 DPI; per-loop deep dive; Gradio demo with loop toggle |
| **Engineering hygiene** | YAML config inheritance, pytest, verify script, model cards, CONTRIBUTING guide |
| **Documentation** | README, 3.5k-word technical report, reproduction guide, results index |

The codebase is **cloneable and runnable**: `python scripts/verify_install.py` → tests →
figures (`--fast`) → demo. Paths are relative; no hardcoded machine-specific roots in the
critical path.

---

## Relevance to defense & autonomous systems

| Theme | Connection |
| --- | --- |
| **Label-efficient perception** | SSL pretrain on cheap unlabeled imagery; small labeled head for mission classes |
| **Aerial / maritime ISR** | Transfer benchmark (EuroSAT proxy + Roboflow maritime path); +4 pp looped win |
| **Edge deployment** | <10M params; 32×32-native stack adaptable to embedded inference |
| **Adaptive compute** | Exit gate = per-sample depth; relevant when latency budgets vary by scene difficulty |
| **Interpretability** | Attention maps evolve loop-by-loop; exit stats and mask panels are inspectable |
| **World models / planning** | JEPA predicts in *latent* space, a stepping stone to action-conditioned dynamics |
| **Systems thinking** | Full pipeline from training through demo, not a notebook-only prototype |

This is **research code**, not a deployed product, but it reads like work from a lab that
ships artifacts, not just slides.

---

## Best artifacts to show in an interview

1. **[Technical report](docs/IJEPA_Looped_Predictor_Report.md)**: main narrative; send this link first
2. **[Gradio demo](app.py)**: live loop-by-loop attention; memorable in a screen share
3. **[Ablation figure](visualizations/figures/05_ablation_summary.png)**: one image tells the normalization story
4. **[Results index](results/README.md)**: shows organized experimental output
5. **[Model cards](model_cards/)**: professional documentation standard

**One-liner for a defense/autonomy role:**  
*"I studied whether recurrent latent predictors help compact JEPA encoders: honest negative
in-domain result, but +4 pp aerial transfer and a clear normalization ablation, with full
reproducibility and an interactive demo."*

---

## Suggested README / GitHub pin order

1. Hero GIF or static from Gradio demo (optional, not yet added)
2. Link to technical report
3. Results table (baseline vs looped vs best ablation vs transfer)
4. Figure gallery
5. `python app.py` quickstart

---

## What I would do next (shows forward thinking)

- Scale to 128×128 unlabeled drone video; re-test adaptive exit gate on harder inputs
- Action-conditioned JEPA for embodied planning horizons
- Deploy Gradio demo to Hugging Face Spaces with LFS checkpoints
- Fine-tune sandwich-RMSNorm checkpoint on Roboflow maritime with partial labels
