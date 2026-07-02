# Model Card: Transfer / Downstream Probing

Frozen-encoder transfer results for the v3 encoders. The encoders are pretrained on
CIFAR-10 (self-supervised) and evaluated **without any fine-tuning of the backbone**: a
linear probe is trained on top of frozen features.

| Field | Value |
| --- | --- |
| Backbones | v3 baseline & v3 looped (frozen) |
| Protocol | Frozen encoder + tuned linear probe |
| Scripts | `scripts/transfer_probe.py`, `scripts/transfer_roboflow.py` |
| Results | [`results/transfer/`](../results/transfer/) |

---

## 1. Aerial imagery transfer (primary)

**Dataset**: EuroSAT RGB used as an aerial/satellite proxy (1500 train / 400 val, 10
classes). This stands in for the Roboflow *Aerial Maritime Drone* dataset, which runs
directly once a `ROBOFLOW_API_KEY` is set (see below).

| Method | Top-1 | Macro F1 | Notes |
| --- | ---: | ---: | --- |
| frozen v3 baseline | 72.75% | 75.66% | tuned linear probe on frozen CIFAR-10 encoder |
| **frozen v3 looped** | **76.75%** | 75.43% | tuned linear probe on frozen looped encoder |
| scratch ResNet18 | 77.50% | 67.06% | trained end-to-end on transfer data |

**Takeaways**

- The **looped encoder transfers ~4 pp better** than the baseline; recurrence helps
  out-of-domain generalization even when it trails on in-domain CIFAR-10.
- A frozen 10M-parameter self-supervised (SSL) encoder is within ~1 pp of a from-scratch ResNet18 while training
  only a linear head, and yields a much higher macro-F1 than the scratch model (better
  class balance / calibration).

Qualitative saliency overlays: `results/transfer/qualitative_baseline_gradcam.png`.

## 2. CIFAR-100 linear probe (label-space shift)

| Metric | Value |
| --- | ---: |
| Val top-1 | 46.32% |
| `feat_std` | 0.1528 |
| Reference | 77.2% in-domain (CIFAR-10) → 46.3% (CIFAR-100) |

The drop is expected: 10× more classes and a different label space on the same 32×32
distribution, evaluated with a frozen backbone.

## 3. Limitations

- EuroSAT is a **proxy** for maritime drone imagery; absolute numbers will shift on the
  real Roboflow dataset.
- Aerial images are resized to 32×32 to match the pretraining resolution; this discards
  high-frequency detail and understates achievable transfer accuracy.
- Linear-probe protocol only; full fine-tuning would likely close the gap to (or exceed)
  the scratch baseline.

## 4. Defense & autonomy relevance

Aerial and maritime perception underpins autonomous surface and airborne platforms: harbor
monitoring, coastal intelligence, surveillance, and reconnaissance (ISR), and search-and-rescue
all rely on robust object cues under variable altitude and lighting. Self-supervised pretraining
on cheap unlabeled imagery reduces the labeled-data burden for mission-specific adaptation, and
the looped encoder's transfer advantage suggests recurrence is a useful inductive bias for
domain shift. The compact footprint keeps inference viable on edge and embedded autonomy hardware.

## 5. How to run

Aerial transfer (EuroSAT proxy):

```bash
python scripts/transfer_roboflow.py --source eurosat
```

Real Roboflow Aerial Maritime Drone dataset:

```bash
export ROBOFLOW_API_KEY="..."   # free at https://app.roboflow.com/settings/api
python scripts/transfer_roboflow.py --download \
  --workspace demm --project aerial-maritime-drone-dataset --version 1 \
  --roboflow-format yolov8 --data-dir data/transfer/aerial_maritime
```

YOLO detection exports are auto-converted to image-level classification (dominant object
class per image). Alternatively export **folder** format from the Roboflow UI and pass
`--export-url <zip-link>`.

CIFAR-100 probe:

```bash
python scripts/transfer_probe.py \
  --dataset cifar100 \
  --checkpoint checkpoints/baseline_v3/latest.pt \
  --out runs/transfer_cifar100.json
```
