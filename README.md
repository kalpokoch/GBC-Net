# GBC-Net: Gallbladder Carcinoma CT Classification

Script-based, config-driven PyTorch code for three experiments on gallbladder CT slices:

| Experiment | Entry point | Config | Ported from |
|---|---|---|---|
| **Binary classifier** (cancer vs. normal): ConvNeXt-Tiny + CBAM + MSAM, 5-fold CV | `gbcnet.train_binary` | `configs/binary_convnext_cbam_msam.yaml` | `Aug21/BinaryClassification/TrainNotebook/ConvNeXtTiny + CBAM + MSAM - 5Fold CV.ipynb` |
| **Backbone ablation**: DenseNet121 / ResNet50 / EfficientNet-B0 / ConvNeXt-Tiny (plain, +CBAM), 3-fold CV | `gbcnet.ablation` | `configs/ablation_backbones.yaml` | `Aug21/BinaryClassification/Ablation/Model Comparison ….ipynb` |
| **Attribute classifier** (7 radiological findings, multi-label): BiomedCLIP @336, dual-scale, 3 seeds × 5 grouped folds | `gbcnet.train_attributes` | `configs/attributes_biomedclip.yaml` | `Aug21/AttributeClassification/train_v3_multiseed_maskedloss.ipynb` |

Model definitions keep the notebooks' module names, so existing `*_best.pth` checkpoints load directly.

---

## 1. Repository structure

```text
GBC-Net/
├── configs/
│   ├── binary_convnext_cbam_msam.yaml
│   ├── ablation_backbones.yaml          # inherits the binary config (_base_)
│   └── attributes_biomedclip.yaml
├── scripts/
│   ├── prepare_binary_split.py          # builds data/manifests/binary_split_manifest.csv
│   └── audit_split_integrity.py         # path / exact-hash / near-duplicate leakage audit
├── src/gbcnet/
│   ├── config.py                        # YAML + _base_ inheritance + --set overrides
│   ├── data/binary.py                   # grayscale dataset, albumentations transforms, sampler
│   ├── data/attributes.py               # dual-scale (full + RUQ crop) dataset, TTA, label masking
│   ├── models/attention.py              # CBAM, MSAM
│   ├── models/convnext.py               # ConvNeXt-Tiny (+CBAM, +MSAM), 1-channel stem
│   ├── models/baselines.py              # DenseNet121, ResNet50, EfficientNet-B0
│   ├── models/registry.py               # name -> factory + optimizer param groups + Grad-CAM layer
│   ├── models/biomedclip.py             # BiomedCLIP encoder + MLP head
│   ├── splits.py                        # stratified k-fold, grouped multilabel k-fold
│   ├── losses.py                        # focal loss, MixUp, masked asymmetric loss
│   ├── metrics.py  plots.py  engine.py  utils.py
│   ├── train_binary.py  ablation.py  evaluate_binary.py  train_attributes.py
│   └── explain.py  predict.py  count_params.py  benchmark.py
├── tests/test_core.py
├── requirements.txt  environment.yml  pyproject.toml
└── data/   outputs/                     # git-ignored (patient data, results, checkpoints)
```

## 2. Installation

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .
pytest                             # fast CPU sanity checks
```

## 3. Data

**No patient data is committed.** Image paths contain patient and clinician names, so `data/`,
`outputs/`, images, CSV manifests/labels and checkpoints are all git-ignored.

Expected layout (paths can be changed in the configs or with `--set data.image_dir=...`):

```text
data/
├── dataset_masked/                # cancer/<contributor>/.../*.jpg|png, non-cancer/*.png
├── manifests/binary_split_manifest.csv
└── labels/attribute_labels_llm.csv          # and attribute_labels_ruleBased.csv
```

Build the binary manifest. To reproduce the split behind the reported results, pin the
100-image test set the 5-fold notebook saved (every other image becomes `train_val`):

```bash
python scripts/prepare_binary_split.py --image-dir data/dataset_masked \
  --pinned-test ../Aug21/BinaryClassification/Output_ConvNeXtTiny_CBAM_MSAM_5Fold/results/test_set.csv \
  --out data/manifests/binary_split_manifest.csv

python scripts/audit_split_integrity.py --manifest data/manifests/binary_split_manifest.csv \
  --image-dir data/dataset_masked
```

## 4. Running

Every config value can be overridden with `--set key.subkey=value` (YAML-parsed).

```bash
# Proposed model, 5-fold CV (multi-hour GPU job)
python -m gbcnet.train_binary --config configs/binary_convnext_cbam_msam.yaml

# Quick end-to-end check (minutes)
python -m gbcnet.train_binary --set cv.n_splits=2 train.epochs=2 train.warmup_epochs=1 \
  train.min_epochs=1 transforms.image_size=128 output.dir=outputs/smoke

# Backbone sweep, then comparison against the proposed run (read from disk, not retrained)
python -m gbcnet.ablation --config configs/ablation_backbones.yaml
python -m gbcnet.ablation --config configs/ablation_backbones.yaml --aggregate-only

# Re-evaluate a finished run (per-fold + fold-ensemble, bootstrap 95% CIs)
python -m gbcnet.evaluate_binary --run-dir outputs/binary_convnext_cbam_msam

# Grad-CAM++ for one fold
python -m gbcnet.explain --run-dir outputs/binary_convnext_cbam_msam --fold 1 --n 16

# Attribute classifier (smoke test first, then the full 3 x 5 run)
python -m gbcnet.train_attributes --set "seeds=[0]" cv.n_folds=2 train.epochs=2 train.patience=2 \
  train.freeze_warmup_epochs=1 output.dir=outputs/smoke_attr
python -m gbcnet.train_attributes --config configs/attributes_biomedclip.yaml

# Utilities
python -m gbcnet.predict --image path/to/slice.png --checkpoints outputs/.../trained_models/*_best.pth
python -m gbcnet.count_params
python -m gbcnet.benchmark --device cuda --out benchmark_gpu.json
```

Each run writes `config_resolved.yaml`, `environment.json` (library versions) and a log file to its
output directory, alongside checkpoints, per-fold metrics/predictions and figures.

## 5. Differences from the notebooks

| | Notebook | This repo |
|---|---|---|
| Decision threshold | Binary: F1-tuned on the test set. Attributes: per-label F1-tuned on OOF predictions | **Fixed at 0.5** for both tasks (`gbcnet.metrics.DECISION_THRESHOLD`); no threshold optimization |
| albumentations 2.x | 1.x arguments (`var_limit`, `alpha_affine`, `max_holes`, …) are silently ignored on 2.x, with warnings suppressed | Mapped explicitly to the 2.x API (`data/binary.py`); the installed version is logged per run |
| Proposed model in the ablation | Checkpoints re-loaded and re-inferred | Saved fold predictions read from the proposed run directory |
| Configuration | `CONFIG` dicts in cells | YAML + `--set` overrides; resolved config saved with each run |
| Bootstrap CIs | — | Pooled-metric 95% CIs (`eval.bootstrap`) |

Verification on this machine: loading the Aug21 fold checkpoints into the ported model gives
test AUCs within 0.002–0.007 of the notebook's recorded values (bf16 vs. fp32 inference alone moves
them by ~0.003), and the attribute pipeline reproduces the Aug21 run's 639 patient groups exactly.

## 6. Known limitations

- **Binary split is image-level.** There is no patient ID; slices from one patient can fall in
  both `train_val` and `test`. `audit_split_integrity.py` flags only near-identical slices.
- **Attribute results are out-of-fold with no held-out test set.**
- The masked-image intensity mismatch between classes (see `Aug6/KNOWLEDGE_BASE.md` §6) is a
  potential confound for the binary task.
