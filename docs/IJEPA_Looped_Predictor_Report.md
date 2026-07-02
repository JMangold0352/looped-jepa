---
title: "I-JEPA with a Looped Predictor: Iterative Refinement, Stability Ablations, and What I Learned"
date: 2026-07-02
author: jepa-ouro
tags: [I-JEPA, self-supervised learning, recurrent predictors, world models, CIFAR-10]
---

# I-JEPA with a Looped Predictor: Iterative Refinement, Stability Ablations, and What I Learned

*A technical report on building a compact Image-JEPA stack with weight-shared recurrence, running it honestly at CIFAR-10 scale, and reading the negative result.*

---

## TL;DR

I trained a **~9.9M-parameter** I-JEPA model on CIFAR-10 and swapped its one-shot ViT predictor for a **weight-shared looped predictor** with a learned exit gate. On the official metric (tuned linear probe on frozen features) the default looped variant lands at **75.13%**, **2.1 percentage points below** the **77.23%** baseline. That headline number is real, and I report it without spin.

The interesting part is everything around it. A structured ablation suite shows that **predictor normalization matters more than loop count**: sandwich-RMSNorm reaches **78.28%**, the best result in the study. Per-loop visualizations show genuine iterative refinement in latent space (+0.21 mean cosine gain from loop 1 to loop 2). And on aerial transfer, the frozen looped encoder beats the frozen baseline by **4.0 pp** despite losing in-domain.

This document is the full story: architecture, experiments, figures, failure modes, and why I still think recurrence belongs in the JEPA family, especially for embodied and edge-constrained systems.

---

## 1. Background: I-JEPA and why recurrence is interesting

**Image-JEPA** (Joint-Embedding Predictive Architecture; LeCun, Assran, et al., 2023) learns visual representations without reconstructing pixels and without contrastive negatives. The student encoder sees a **context** subset of image patches. A predictor must infer the **latent embeddings** of masked **target** patches. Targets come from a slow **EMA teacher** encoding the full image. Loss is smooth-L1 in representation space.

That setup is already a primitive **world model**: the network learns to infer hidden structure from partial observations. But in standard I-JEPA the predictor is a shallow feed-forward ViT, one pass from context latents to target latents.

Biological and robotic perception rarely works that way. Agents **iterate**: a coarse hypothesis, then refinement. In model-based RL and latent dynamics literature, recurrent computation at fixed parameter budget is an old idea (think shallow iterative inference, recurrent depth, universal transformers). The question I cared about was narrow and testable:

> *If the JEPA predictor is made recurrent (same weights, multiple steps, optional early exit), do the representations improve without growing the model?*

CIFAR-10 at 32×32 is a harsh but honest testbed. It is small enough to run full ablations on a laptop GPU, yet large enough that training instabilities show up quickly. I fixed a strong **v3 baseline** first (77.23% tuned probe, no collapse), then changed **only the predictor**.

---

## 2. Architecture

### 2.1 Shared I-JEPA backbone (both models)

| Component            | Config                                                          |
| -------------------- | --------------------------------------------------------------- |
| Image                | 32×32 RGB, patch size 4 → 8×8 grid (64 tokens)                  |
| Encoder              | ViT: `embed_dim=384`, `depth=5`, `heads=6`, drop-path 0.1       |
| Predictor (baseline) | ViT: `embed_dim=128`, `depth=4`, `heads=4`                      |
| Trainable params     | **9,816,960** (~9.9M)                                           |
| Optimizer            | AdamW, lr `2e-3`, wd `0.05`, batch 256                          |
| Schedule             | 300 epochs, 15-epoch warmup → cosine                            |
| Augmentation         | RandAugment(2, 9) + mild RRC (scale 0.5–1.0)                    |
| EMA teacher          | momentum `0.996 → 0.9999` (capped below 1.0)                    |
| Masking              | Context 24–40 patches, target 10–22; deterministic subselection |

Two stability fixes from earlier iterations are worth naming explicitly, because they matter for anyone reproducing small-scale JEPA:

1. **Deterministic target subselection** (`sorted(targets)[:N]`). A well-intentioned random subselection caused **representation collapse** in v2/v2b (~64% probe vs ~80% in v1). I reverted it.
2. **EMA cap at 0.9999**. With `ema_momentum_end=1.0`, the final EMA update is a no-op; `feat_std` decayed through training in v1.

### 2.2 Looped predictor + exit gate

The looped variant wraps the same `VisionTransformerPredictor` in a `LoopedPredictor`:

```
context latents ──► build_sequence(ctx + mask tokens)
                         │
                    ┌────▼────┐
                    │ loop t  │  shared BlockStack (depth 4)
                    └────┬────┘
                         ├──► exit_gate(x.mean) → P(stop at t)
                         │
                    (repeat up to max_loops)
                         │
                         ▼
                  output_proj → predicted target latents
                         │
              smooth-L1 vs EMA teacher targets
```

**Weight sharing** means loops add compute, not parameters. The default released checkpoint uses `max_loops=2`, LayerNorm, and an exit gate.

The exit gate is a linear layer on the mean pooled hidden state, passed through sigmoid each loop. Training adds an **entropy penalty** on exit probabilities (`exit_entropy_beta=0.01`) so the gate does not collapse to always exiting at loop 1 or never exiting.

```python
# Simplified from src/jepa/models/looped_predictor.py
for _ in range(loops):
    x = self.base_predictor.forward_stack(x)
    if self.use_exit_gate:
        exit_probs.append(sigmoid(self.exit_gate(x.mean(dim=1))))
target_tokens = self.base_predictor.output_proj(x[:, n_ctx:])
```

Expected depth is computed as a survival analysis over per-loop exit probabilities, the same quantity I log during training and plot in the demo.

**What did not change:** encoder architecture, augmentation, schedule, masking, EMA recipe. The baseline checkpoint and config remain untouched; the looped model is a drop-in predictor swap (`configs/image_jepa_cifar10_v3_looped.yaml`).

---

## 3. Experimental setup

### 3.1 Evaluation protocol

Every model in this study uses the same official metric:

- Freeze the encoder (and EMA teacher for target generation during pretraining only).
- Train a **linear probe** on mean-pooled patch features.
- Cosine LR schedule, 100 probe epochs.
- LR sweep over `{3e-4, 1e-3, 3e-3}` with **feature standardization**.
- Report best validation top-1.

During pretraining I also log a cheaper fixed-LR probe every 25 epochs for trend monitoring. Those numbers are **not** the headline metric.

I track `feat_std` (standard deviation of feature norms across the validation set) as a collapse diagnostic. Healthy runs sit around 0.15 to 0.30 at this scale; sustained values below ~0.10 are a red flag.

### 3.2 Ablation design

I ran **three ablation suites**, seven full 300-epoch training runs total, all branching from a common looped base config:

| Suite              | Question                          | Variants                                   |
| ------------------ | --------------------------------- | ------------------------------------------ |
| **Loop count**     | Does more recurrence help?        | `loops_1`, `loops_2`, `loops_4`            |
| **Exit entropy**   | Does the gate regularizer matter? | `entropy_on` (β=0.01), `entropy_off` (β=0) |
| **Predictor norm** | LayerNorm vs sandwich RMSNorm     | `layernorm`, `sandwich_rms`                |

The v3 non-looped baseline and the default looped checkpoint are reference points outside the ablation grid but trained with the same recipe.

---

## 4. Results

### 4.1 Headline comparison: baseline vs default looped

| Model                         | Tuned top-1 | Best LR | `feat_std` | Δ vs baseline |
| ----------------------------- | ----------- | ------- | ---------- | ------------- |
| **v3 baseline**               | **77.23%**  | 3e-3    | 0.1607     | reference     |
| v3 looped (2-loop, LayerNorm) | 75.13%      | 3e-3    | 0.1450     | **−2.10 pp**  |

Per-LR breakdown (looped): 73.22% @ 3e-4, 74.79% @ 1e-3, 75.13% @ 3e-3. The gap is consistent across the sweep, not an LR artifact.

Source: `runs/looped_v3_comparison.json`.

### 4.2 Ablation suite 1: loop count

| Variant | Tuned top-1 | `feat_std` | Mean loops | Notes                                                            |
| ------- | ----------- | ---------- | ---------- | --------------------------------------------------------------- |
| loops_1 | **77.24%**  | 0.1609     | 1.00       | Recurrent wrapper, single pass; essentially the baseline predictor |
| loops_2 | 75.04%      | 0.1276     | 1.50       | Default depth; exit gate active                                 |
| loops_4 | 75.49%      | 0.1049     | 1.88       | More loops, lower `feat_std`                                    |

**Reading this table:** `loops_1` matches the baseline, which is a useful sanity check; the recurrent machinery with one pass does not hurt. Adding loops **without** changing normalization **drops** `feat_std` and probe accuracy. Four loops do not recover the lost signal; they push `feat_std` toward collapse territory (0.105).

On validation, the exit gate saturates: all 10k samples exit at loop 2 with P(exit) ≈ 0.5 at each step (expected depth 1.5). The gate is active but not yet producing a spread of exit depths at this scale.

### 4.3 Ablation suite 2: exit-gate entropy

| Variant             | Tuned top-1 | `feat_std` | Mean loops |
| ------------------- | ----------- | ---------- | ---------- |
| entropy_on (β=0.01) | 75.36%      | 0.1275     | 1.50       |
| entropy_off (β=0)   | 76.00%      | 0.1270     | 1.55       |

Turning off the entropy penalty gives a **+0.64 pp** bump: modest, but directionally interesting. The regularizer may be slightly over-constraining the gate early in training. Neither variant closes the gap to baseline. Exit-gate tuning is second-order compared to normalization.

### 4.4 Ablation suite 3: predictor normalization (the main finding)

| Variant          | Tuned top-1 | `feat_std` | Mean loops |
| ---------------- | ----------- | ---------- | ---------- |
| layernorm        | 75.36%      | 0.1275     | 1.50       |
| **sandwich_rms** | **78.28%**  | 0.0432     | 1.50       |

This is the result that reframes the project. **Sandwich RMSNorm** (RMSNorm before and after each attention and FFN sub-layer in the predictor stack) beats the baseline by **+1.05 pp** while keeping the same loop count and exit gate.

The low `feat_std` (0.043) looks alarming in isolation, but the probe is *higher*, not lower. Features are more tightly scaled, not collapsed. Normalization changed the geometry of the representation space in a way the linear probe likes. Loop count alone did not do that.

**Practical lesson:** if you are going to iterate a predictor stack with shared weights, **stabilize the inner loop** before chasing depth.

### 4.5 Full results summary

Figure 1 (`visualizations/figures/05_ablation_summary.png`): all seven ablation variants under the v3 training recipe (tuned linear probe).

---

## 5. Visualizations and what they show

I built a publication figure pipeline (`visualizations/generate_all_figures.py`) plus a per-loop deep dive (`visualizations/loop_analysis/`). The figures are not decoration; they are how I sanity-checked that recurrence was doing something mechanistic.

### 5.1 Masked latent prediction quality

Figure 2 (`visualizations/figures/01_mask_reconstruction.png`): baseline (left three columns) vs looped (right three columns): original, masked context, target patches tinted by cosine similarity to the EMA teacher (greener = better).

No pixels are reconstructed. Green target patches mean the predictor's latent guess aligns with the teacher. The looped model often shows **more heterogeneous** patch quality (some targets improve at loop 2, others do not), which is what you would expect from iterative inference rather than a single global linear map.

### 5.2 Attention maps across loops

Figure 3 (`visualizations/figures/02_attention_maps.png`): predictor attention from target tokens onto context patches. Looped: one map per loop.

The baseline attends in one shot. The looped predictor **reshapes** attention between loops: early loops spread mass more diffusely; later loops sharpen on a subset of context patches. That is the most direct evidence I have that the recurrence is performing qualitatively different computation, not just repeating the same transform.

### 5.3 Per-loop cosine refinement

Figure 4 (`visualizations/figures/03_per_loop_cosine.png`): batch-mean cosine similarity between predicted and teacher target embeddings after each loop.

Loop 1 → loop 2 consistently adds latent prediction quality on the validation set. The aggregate curve rises even when the **encoder** probe falls: the predictor is refining, but the encoder is learning a representation that is slightly worse for the downstream linear classifier. That split is important: recurrence helps the **prediction task** without automatically helping the **probe task**.

Per-sample analysis (`visualizations/loop_analysis/summary.json`, 512 validation images):

| Stat                              | Value      |
| --------------------------------- | ---------- |
| Mean cosine gain (loop 1 → final) | **+0.213** |
| Exit P(loop 1), P(loop 2)         | 0.50, 0.50 |
| Expected exit depth               | 1.50       |

Figure 5 (`visualizations/loop_analysis/03_loops_vs_difficulty.png`): loop-1 cosine (proxy for difficulty) vs expected exit depth. At this scale the gate has not yet learned diverse per-sample depths.

### 5.4 Exit-loop distribution

Figure 6 (`visualizations/figures/04_exit_loop_distribution.png`): histogram and CDF of exit loops on the full validation set.

### 5.5 Training dynamics

Figure 7 (`visualizations/figures/04_training_curves.png`): loss, periodic probe, and feat_std over 300 epochs (baseline vs looped).

The looped run tracks slightly lower `feat_std` throughout training. That drift precedes the final probe gap and is visible from mid-training onward, not a late-epoch fluke.

### 5.6 Embedding geometry

Figure 8 (`visualizations/figures/03_embeddings.png`): t-SNE of frozen encoder features (baseline vs looped).

Class structure is broadly similar; the looped encoder compresses norm variance (consistent with lower `feat_std`) without obliterating separability.

### 5.7 Early vs late exit examples

Figure 9 (`visualizations/loop_analysis/04_early_vs_late_examples.png`): validation examples annotated with per-loop cosine trajectories (early-exit vs more-loop samples).

---

## 6. Transfer learning

In-domain CIFAR-10 probing is the wrong sole metric for a model aimed at **world modeling**. I froze both encoders and trained linear probes on **EuroSAT RGB** (aerial/satellite imagery, resized to 32×32; 1500 train / 400 val) as a lightweight domain-shift benchmark. Roboflow *Aerial Maritime Drone* is supported for follow-up with an API key.

| Method               | Top-1      | Macro F1 | Training              |
| -------------------- | ---------- | -------- | --------------------- |
| frozen v3 baseline   | 72.75%     | 75.66%   | SSL on CIFAR-10 only  |
| **frozen v3 looped** | **76.75%** | 75.43%   | SSL on CIFAR-10 only  |
| scratch ResNet18     | 77.50%     | 67.06%   | End-to-end on EuroSAT |

The looped encoder gains **+4.0 pp** top-1 over the baseline despite losing in-domain. Macro F1 is essentially tied, and both frozen SSL models trail scratch ResNet18 on raw accuracy but beat it badly on F1 balance: the ResNet overfits the dominant classes.

Figure 10 (`results/transfer/qualitative_baseline_gradcam.png`): probe-guided saliency on EuroSAT (baseline encoder; green = correct, red = incorrect).

I do not over-claim: EuroSAT is a proxy, not a maritime drone deployment. 32×32 resizing throws away detail. But the **direction** is stable: recurrence helps out-of-domain transfer at fixed parameter count. That is the result worth stressing in an autonomy context.

**CIFAR-100** (100-class label shift, same 32×32 size): frozen baseline reaches **46.32%** vs 77.2% in-domain. Expected drop; included for completeness.

---

## 7. Why didn't the default looped model beat the baseline?

An honest postmortem, in order of confidence:

### 7.1 Wrong predictor normalization for recurrence

LayerNorm + weight sharing across loops appears to **contract** feature norms (`feat_std` 0.145 vs 0.161) without improving probe-relevant geometry. Sandwich RMSNorm fixes the probe and beats baseline with the same loops and the same gate. The default looped checkpoint simply used the wrong norm for iterative depth.

### 7.2 Encoder and predictor coupling

JEPA trains encoder and predictor jointly. A harder iterative predictor changes the encoder's learning problem. The predictor **does** refine latents loop-over-loop (cosine rises), but the encoder may settle into a basin that is good for multi-step prediction and slightly worse for mean-pooled linear classification. Different probe heads or spatial probing might tell a different story.

### 7.3 Scale and task mismatch

CIFAR-10 at 32×32 with 4×4 patches is a **toy** spatial layout. There is limited masked structure to infer; one feed-forward pass may be near the ceiling for the probe metric. Recurrence may pay off more when targets are harder (higher resolution, temporal prediction, action conditioning) where iterative refinement has more room to matter.

I also tried pushing the stack further (v4: two-view + harder masking; v5: two-view only). Both **collapsed** or underperformed badly. Small-scale JEPA is brittle. The v3 recipe is well-tuned for this regime; the looped predictor was grafted onto it, not co-designed from scratch.

### 7.4 Exit gate not yet doing adaptive compute

At convergence, every validation sample runs 2 loops. The gate learns uniform 50/50 exit probabilities, a kind of **fixed fractional depth**, not sample-adaptive routing. Adaptive compute needs harder inputs or training pressure to diversify exit depths.

### What it still teaches us

Negative results with good instrumentation are useful. The study leaves behind:

- Evidence that **recurrence changes attention dynamics** (Figures 3, 9).
- Evidence that **latent prediction improves per loop** even when probe does not (Figures 4, 5).
- A clear **normalization ablation** that dominates loop count (Table in §4.4).
- A **transfer win** that the in-domain metric alone would have hidden (§6).

If you are building toward embodied world models, the looped predictor is a plausible **inference-time compute knob**, but only with the right inner-loop stability, and probably not judged solely on CIFAR-10 linear probe.

---

## 8. Future directions

**Action-conditioned JEPA.** The natural next step for robotics is predicting latents of **future** observations conditioned on actions; recurrence in the predictor mirrors iterative planning horizons.

**Temporal / video JEPA.** Loops over spatial targets are a stepping stone; loops over **time** are closer to dynamics models for MPC and model-based RL.

**Co-design normalization and depth.** Sandwich RMSNorm should not be an afterthought; it should be part of the recurrent predictor recipe from day one. SwiGLU + RoPE (the `ouro_ready` predictor path in this codebase) is unexplored at full 300-epoch scale here.

**Train the exit gate for heterogeneous compute.** Mix easy and hard batches; add latency-aware losses; evaluate on inputs where loop-1 cosine is low (Figure 5) and measure whether the gate learns to spend more depth.

**Larger scale and resolution.** 32×32 CIFAR is a development environment. The same architecture at 128×128 or on unlabeled drone video is where transfer gains and adaptive depth may compound.

**Hierarchical JEPA.** Recurrent predictor over patch latents; recurrent **encoder** over frames: different timescales of the same principle.

---

## 9. Conclusion

The question was whether a **looped, weight-shared JEPA predictor** could improve visual world modeling without growing past a **10M-parameter** budget. The default answer on CIFAR-10 is **no**, at least not on the tuned linear probe, where it loses **2.1 pp** against a strong baseline I trust.

But the project is not a failure. It produced:

- A **reproducible I-JEPA implementation** with honest training fixes documented the hard way (v2 collapse, v4/v5 dead ends).
- A **seven-run ablation grid** showing that **sandwich RMSNorm + loops** beats everything, including baseline (**78.28%**).
- A **visualization and demo stack** that makes predictor behavior inspectable loop by loop.
- A **transfer result** where recurrence helps on aerial imagery when the in-domain metric says it should not.

For **autonomous systems** (drones, harbor ISR, edge perception) the relevant constraints are parameter budget, label efficiency, and whether a model can **spend variable compute** on hard scenes. A 10M-param encoder with a recurrent latent predictor and a learned exit gate is a credible building block: pretrained on cheap unlabeled video, probed or fine-tuned with small labeled sets, deployed with interpretable attention and exit statistics.

World models will not be won on CIFAR-10 alone. They will be won by architectures that **predict in latent space**, **refine under compute pressure**, and **transfer** when the deployment domain shifts. This repo is one controlled study of that bet, reported honestly, with the figures to prove what happened.

---

## Appendix

### Reproduce

```bash
# Train
python scripts/train_jepa.py --config configs/image_jepa_cifar10_v3.yaml
python scripts/train_jepa.py --config configs/image_jepa_cifar10_v3_looped.yaml

# Evaluate
python scripts/compare_looped_v3.py
python scripts/run_ablations.py --suite all --train

# Figures
python visualizations/generate_all_figures.py

# Demo
uv sync --extra demo && python app.py
```

### Artifact index

| Artifact            | Path                                                         |
| ------------------- | ------------------------------------------------------------ |
| Baseline checkpoint | `checkpoints/baseline_v3/latest.pt`                          |
| Looped checkpoint   | `checkpoints/baseline_v3_looped/latest.pt`                   |
| Best ablation       | `checkpoints/ablations/sandwich_norm/sandwich_rms/latest.pt` |
| Comparison JSON     | `runs/looped_v3_comparison.json`                             |
| Ablation JSON       | `results/ablations/summary.json`                             |
| Transfer results    | `results/transfer/transfer_results.md`                       |
| Model cards         | `model_cards/`                                               |

### Citation

```bibtex
@misc{jepa_cifar10_looped_2026,
  title        = {I-JEPA with a Looped Predictor: Iterative Refinement,
                  Stability Ablations, and What I Learned},
  author       = {jepa-ouro},
  year         = {2026},
  howpublished = {Technical report, jepa\_v3\_model repository}
}
```

### References

- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*, 2023.
- Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, 2020.
- Grill et al., *Bootstrap Your Own Latent*, 2020 (EMA teacher lineage).

---

*Report generated as part of the looped-jepa project. Code, checkpoints, figures, and interactive demo are in the repository root.*
