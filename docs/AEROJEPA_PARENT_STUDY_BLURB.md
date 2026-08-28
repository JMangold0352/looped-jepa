## Parent study: looped-jepa

Static-image predecessor: [**looped-jepa**](https://github.com/JMangold0352/looped-jepa) (~9.9M I-JEPA on CIFAR-10). Default looped predictor (LayerNorm, 2 loops): **75.13%** tuned probe vs **77.23%** baseline (**−2.10 pp**). Best ablation (sandwich RMSNorm): **78.28%** (**+1.05 pp**). Frozen EuroSAT transfer: looped **76.75%** vs baseline **72.75%** (**+4.0 pp**). Full arc: [RESEARCH_ARC.md](https://github.com/JMangold0352/looped-jepa/blob/main/RESEARCH_ARC.md).

AeroJEPA reuses the weight-shared looped predictor + exit gate on egocentric video; recurrence here gains **+0.7 pp** latent cosine over feed-forward, but action counterfactuals and hard L-turn closed-loop tests still fail.
