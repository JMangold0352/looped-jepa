# Latent-space findings (CIFAR-10)

From [`visualize_latents.ipynb`](visualize_latents.ipynb) on frozen encoder features.

- Classes stay separated in t-SNE/UMAP for both baseline and looped. Looped norms are tighter (`feat_std` 0.145 vs 0.161) but this is not collapse — silhouette and nearest-centroid accuracy drop modestly (−2.1 pp probe gap).
- Worst nearest-centroid pairs come from the val confusion matrix (cat↔dog, automobile↔truck, bird↔airplane; exact order in the notebook). Looped mistakes overlap baseline mistakes; recurrence does not add a new failure mode.
- Per-loop cosine to EMA targets still rises (~+0.21 mean gain). The probe gap is not from a predictor that stops refining. Figures: [`03_per_loop_cosine.png`](../visualizations/figures/03_per_loop_cosine.png), [`02_cosine_l1_by_loop_per_class.png`](../visualizations/loop_analysis/02_cosine_l1_by_loop_per_class.png).
- Sandwich RMSNorm: low `feat_std` (0.0432) with **higher** probe (78.28%). Tighter scaling, not dead features. Default looped LayerNorm lowers both spread and probe.
- Exit gate on CIFAR val: ~50/50 at loops 1 and 2, mean depth 1.5 ([`01_exit_distribution.png`](../visualizations/loop_analysis/01_exit_distribution.png)). Fixed effective depth on this checkpoint.

The in-domain gap tracks **predictor normalization and feature geometry**, not missing class structure in the encoder.
