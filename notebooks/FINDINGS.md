# Latent-space findings (CIFAR-10 baseline vs looped vs sandwich)

Bullets from `notebooks/visualize_latents.ipynb` — quantitative, no assumed class collapse.

- **No global collapse:** Frozen-encoder t-SNE/UMAP keeps ten CIFAR classes separated for both baseline and looped; the looped map is slightly tighter (lower norm variance) but not a single blob. Silhouette and nearest-centroid accuracy drop modestly for looped, consistent with the −2.1 pp probe gap rather than representation death.
- **Confusion is data-driven, not anecdotal:** Worst nearest-centroid pairs come from the measured confusion matrix (e.g. cat↔dog, automobile↔truck, bird↔airplane — exact ranking printed in the notebook from val features). Looped errors concentrate on the same fine-grained pairs; recurrence does not invent a new failure mode.
- **Predictor loops still align latents:** Per-loop cosine to EMA teacher targets rises across loops (~+0.21 mean gain); the gap is not “the predictor fails to refine.” See existing figure [`../visualizations/figures/03_per_loop_cosine.png`](../visualizations/figures/03_per_loop_cosine.png) and per-class bars in [`../visualizations/loop_analysis/02_cosine_l1_by_loop_per_class.png`](../visualizations/loop_analysis/02_cosine_l1_by_loop_per_class.png).
- **Low `feat_std` on sandwich is tighter scaling, not collapse:** Sandwich-RMSNorm compresses L2 norm spread (`feat_std` 0.0432) while **raising** tuned probe to 78.28%. Norm histograms overlap in support; classes remain separable. Default looped LayerNorm lowers both spread and probe.
- **Exit gate ≈ uniform:** Expected depth ~1.5 with ~50% / ~50% exit mass at loops 1 and 2 ([`../visualizations/loop_analysis/01_exit_distribution.png`](../visualizations/loop_analysis/01_exit_distribution.png)). Adaptive compute is claimed in the architecture, not yet learned on CIFAR — recurrence cost is effectively fixed-depth.

**Bottom line:** I did not find collapse; the in-domain gap is **predictor normalization + geometry** (LayerNorm looped hurts probe-relevant scaling; sandwich fixes it), not missing class structure in the encoder.
