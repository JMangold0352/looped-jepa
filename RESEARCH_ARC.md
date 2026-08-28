# Research arc

## 1. Claim

Weight-shared recurrent predictors in JEPA need **different normalization** than a single-pass feed-forward predictor.

On CIFAR-10 (frozen encoder, tuned linear probe, 300-epoch pretrain):

| Predictor | Tuned top-1 | Δ vs baseline | feat_std |
| --- | ---: | ---: | ---: |
| v3 baseline (feed-forward) | **77.23%** | — | 0.1607 |
| Looped + LayerNorm + exit gate (default) | 75.13% | **−2.10 pp** | 0.1450 |
| Looped + **sandwich RMSNorm** (best ablation) | **78.28%** | **+1.05 pp** | 0.0432 |

Recurrence alone (LayerNorm, 2 loops) is **net-negative** in-domain. Sandwich RMSNorm is the intervention that makes recurrence **net-positive** on the same encoder and loop budget. Normalization dominates raw loop count in the ablation suite ([`results/ablations/summary.md`](results/ablations/summary.md)).

The exit gate is **not** adaptive yet: ~50% / ~50% exit mass at loops 1 and 2, mean depth **1.5** ([`visualizations/loop_analysis/summary.json`](visualizations/loop_analysis/summary.json)).

---

## 2. Domain shift (already measured)

Same looped encoder, frozen, linear probe:

| Setting | Baseline | Looped | Δ |
| --- | ---: | ---: | ---: |
| CIFAR-10 in-domain | 77.23% | 75.13% | **−2.1 pp** |
| EuroSAT transfer (aerial proxy) | 72.75% | 76.75% | **+4.0 pp** |

That is the current evidence: recurrence **hurts** on 32×32 object classes with the default norm, and **helps** when the test distribution is aerial / more structured (EuroSAT RGB, 1500/400 split). EuroSAT is a proxy, not a deployment benchmark ([`results/transfer/transfer_results.json`](results/transfer/transfer_results.json)).

---

## 3. Child project: AeroJEPA

[**AeroJEPA**](https://github.com/JMangold0352/aerojepa) takes this looped-predictor idea into **egocentric drone video** (synthetic pretrain + Wilds fine-tune, ~3–5M params). Numbers from the AeroJEPA README ([Key results](https://github.com/JMangold0352/aerojepa#key-results)):

| Finding | Number | Source |
| --- | --- | --- |
| Looped vs feed-forward (masked objective) | **+0.7 pp** latent cosine (0.961 vs 0.954) | [`model_cards/aerojepa_base.md`](https://github.com/JMangold0352/aerojepa/blob/main/model_cards/aerojepa_base.md) |
| Future rollout @ h=4 | **flat ~0.97** (world_model: 0.973) | same |
| Per-loop cosine | **0.87 → 0.96 → 0.98**; mean loops **1.75** | same |
| Action counterfactuals | true ≈ zero ≈ shuffle ≈ **0.994** cosine | not causal yet |
| Closed-loop L-turn stress | scale ×1.25 → **0% success** (10 seeds) | [`visualizations/closed_loop/stress_suite.json`](https://github.com/JMangold0352/aerojepa/blob/main/visualizations/closed_loop/stress_suite.json) |

Video recurrence **does** improve latent alignment; long-horizon future loss **plateaus**; action-conditioning **does not** pass true/zero/shuffle tests; **closed-loop L-turns fail** under the current planner stack.

---

## 4. Thesis-shaped questions (next 12 months)

I am prioritizing three, not ten:

1. **Scale vs modality:** Is the CIFAR penalty a **scale effect** (200 classes / 64px) or a **static-image effect**? TinyImageNet configs are in [`results/scale/README.md`](results/scale/README.md); not trained yet.
2. **Adaptive compute:** Can the exit gate be trained to spend extra loops on **hard aerial / turbulent frames** instead of ~50/50 on CIFAR and ~1.75 fixed steps on video?
3. **Causal actions:** Can action-conditioned AeroJEPA **separate** true, zero, and shuffled controls on latent rollouts (currently ≈0.994 for all three)?

---

## 5. What this repo is not

- **Not** a foundation model or general vision backbone at ImageNet scale.
- **Not** onboard flight software or a certified autonomy stack.
- **Not** multi-seed ImageNet or a claim that looped predictors always beat feed-forward.
- **Not** proof that the exit gate learns adaptive depth (it does not, on current checkpoints).

This repo **is** a controlled ~10M-parameter I-JEPA stack, an honest negative in-domain result, an ablation that isolates normalization, a transfer signal, and the parent study for AeroJEPA.
