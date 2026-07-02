# Transfer Learning: Roboflow Aerial Maritime

**Dataset**: EuroSAT RGB (satellite aerial proxy; use Roboflow maritime with --download when API key set)
**Classes**: 10 (AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake)
**Train / val images**: 1500 / 400

## Results

| Method | Top-1 (%) | Macro F1 (%) | Notes |
| --- | ---: | ---: | --- |
| frozen_v3_baseline | 72.75 | 75.66 | Tuned linear probe, CIFAR-10 pretrained encoder |
| frozen_v3_looped | 76.75 | 75.43 | Tuned linear probe, looped CIFAR-10 encoder |
| scratch_resnet18 | 77.50 | 67.06 | ResNet18 trained from scratch on transfer data |

## Relevance to Defense & Autonomy

Aerial maritime perception is a core enabler for autonomous surface and airborne platforms: harbor monitoring, search-and-rescue, and coastal ISR all depend on robust object cues (vessels, docks, vehicles) under variable altitude and lighting. Self-supervised encoders pretrained on cheap unlabeled imagery can reduce labeled-data requirements for mission-specific fine-tuning, a practical constraint in defense and autonomy programs where expert annotation is scarce and deployment timelines are tight.

## Qualitative

Saved `results/transfer/qualitative_baseline_gradcam.png`; green/red titles show correct vs incorrect predictions with probe-guided Grad-CAM overlays.

## Running on Roboflow Aerial Maritime Drone (recommended)

```bash
export ROBOFLOW_API_KEY="..."   # free at https://app.roboflow.com/settings/api
python scripts/transfer_roboflow.py --download \
  --workspace demm --project aerial-maritime-drone-dataset --version 1 \
  --roboflow-format yolov8 --data-dir data/transfer/aerial_maritime
```

YOLO detection exports are auto-converted to image-level classification (dominant object class per image).
Alternatively, export **folder** format from the Roboflow UI and pass `--export-url <zip-link>`.
