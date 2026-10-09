# GBC-Net: Gallbladder Carcinoma CT Classification

Script-based, config-driven PyTorch code for three experiments on gallbladder CT slices:

- **Binary classifier** (cancer vs. normal): ConvNeXt-Tiny + CBAM + MSAM, 5-fold CV —
  `gbcnet.train_binary`, `configs/binary_convnext_cbam_msam.yaml`
- **Backbone ablation**: DenseNet121 / ResNet50 / EfficientNet-B0 / ConvNeXt-Tiny (plain, +CBAM), 3-fold CV —
  `gbcnet.ablation`, `configs/ablation_backbones.yaml`
- **Attribute classifier** (7 radiological findings, multi-label): BiomedCLIP @336, dual-scale, 3 seeds × 5 grouped folds —
  `gbcnet.train_attributes`, `configs/attributes_biomedclip.yaml`

---

## 1. Repository structure

```text
GBC-Net/
├── configs/
│   ├── binary_convnext_cbam_msam.yaml
│   ├── ablation_backbones.yaml          # inherits the binary config (_base_)
│   └── attributes_biomedclip.yaml
├── scripts/
│   ├── mask_dataset.py                  # raw slices -> body-masked dataset_masked/ + mapping.csv
│   ├── prepare_binary_split.py          # builds data/manifests/binary_split_manifest.csv
│   └── audit_split_integrity.py         # path / exact-hash / near-duplicate leakage audit
├── src/gbcnet/
│   ├── config.py                        # YAML + _base_ inheritance + --set overrides
│   ├── preprocessing.py                 # body masking (threshold, morphology, largest contour)
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
├── raw/                           # original slices
│   ├── GBCA/<contributor>/<batch>/.../*.jpg|png
│   └── NORMAL GALL BLADDER 851/*.png
├── dataset_masked/                # generated: cancer/<contributor>/.../, non-cancer/
├── manifests/binary_split_manifest.csv
└── labels/attribute_labels_llm.csv          # and attribute_labels_ruleBased.csv
```

**Step 1 — body masking.** Each slice is thresholded at gray level 20, cleaned with a
morphological close (x2) and open (x1) using a 15 px elliptical kernel, reduced to its largest
filled contour, and everything outside that body mask is set to black. Folder structure and
filenames are preserved. The script also writes `mapping.csv`, `mask_stats.json`, and reports the
cancer vs. normal body-only intensity gap (a potential acquisition shortcut).

```bash
# Inspect the masks on a few random slices first (writes data/mask_preview.png)
python scripts/mask_dataset.py --data-dir data/raw --out-dir data/dataset_masked --preview 4

# Mask the full dataset
python scripts/mask_dataset.py --data-dir data/raw --out-dir data/dataset_masked
```

**Step 2 — split manifest.** Draw a stratified test set, or pin an existing one (a CSV with a
`relative_path` column; every other image becomes `train_val`):

```bash
python scripts/prepare_binary_split.py --image-dir data/dataset_masked \
  --test-per-class 50 --seed 42 --out data/manifests/binary_split_manifest.csv
# or: --pinned-test path/to/test_set.csv

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

## 5. Design notes

- **Decision threshold** is fixed at 0.5 for both tasks (`gbcnet.metrics.DECISION_THRESHOLD`);
  no threshold optimization is performed.
- **Augmentation** works with albumentations 1.x and 2.x: arguments are mapped to the installed
  API (`data/binary.py`), and the library version is logged with every run.
- **Ablation comparison** reads the proposed model's saved fold predictions instead of re-running it.
- **Uncertainty**: pooled test metrics come with percentile-bootstrap 95% CIs (`eval.bootstrap`).

## 6. Known limitations

- **Binary split is image-level.** There is no patient ID; slices from one patient can fall in
  both `train_val` and `test`. `audit_split_integrity.py` flags only near-identical slices.
- **Attribute results are out-of-fold with no held-out test set.**
- **Class intensity mismatch.** Masked cancer and normal slices differ in mean body intensity
  (see the report written by `scripts/mask_dataset.py`), a potential confound for the binary task.
