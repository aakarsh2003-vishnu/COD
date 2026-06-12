# MCOD Foundation Model Robustness

Compact code release for MCOD preprocessing, multispectral 8-band dataset adaptation, and baseline evaluation experiments using SINet-V2, CamoFormer, and ZoomNet.

This repository intentionally excludes local datasets, generated image outputs, model checkpoints, and prediction folders. The tracked files are the reusable code, lightweight evaluation tables, and scripts needed to reproduce the pipeline once the MCOD data and model weights are available locally.

## Repository Contents

- `process_dataset.py` - builds normalized MCOD spectral views, false-colour images, grayscale NIR views, PCA projections, all-8 tensors, and masks.
- `scripts/generate_mcod_manifest.py` - creates the MCOD manifest used by preprocessing and training.
- `scripts/complete_tasks_6_to_8.py` - generates verification reports, figures, tables, and metric summaries from processed data.
- `scripts/evaluate_cod_predictions.py` - evaluates prediction folders with common camouflaged object detection metrics.
- `SINet-V2/` - SINet-V2 training/testing code adapted for MCOD and 8-band inputs.
- `CamoFormer/` - CamoFormer training/testing code adapted for MCOD and 8-band inputs.
- `ZoomNet/` - ZoomNet training/testing code adapted for MCOD RGB and 8-band MSI experiments.
- `outputs/tables/` - lightweight CSV summaries from completed evaluations.
- `outputs/figures/`, `outputs/presentation/`, and `visual_outputs/` - generated report, presentation, and qualitative result artifacts.
- `data_audit/` and `MCOD_inspection_report/` - dataset audit and inspection summaries.

## Setup

Create and activate a Python environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For model training and inference, install the PyTorch version that matches your CUDA setup from the official PyTorch instructions.

## Expected Local Data

The scripts expect local data paths similar to the working project layout:

```text
data/
MCOD_processed/
SINet-V2/snapshot/
CamoFormer/checkpoint/
ZoomNet/checkpoints/
```

These paths are ignored by Git because datasets, checkpoints, and full prediction folders are too large for a normal GitHub repository.
Tracked output folders contain only report-friendly artifacts and lightweight summaries.

## Typical Workflow

Generate or refresh the MCOD manifest:

```bash
python scripts/generate_mcod_manifest.py
```

Build the processed spectral views:

```bash
python process_dataset.py
```

Run the report and evaluation artifact generation:

```bash
python scripts/complete_tasks_6_to_8.py
```

Evaluate a prediction directory:

```bash
python scripts/evaluate_cod_predictions.py --pred-dir path/to/predictions --split test
```

## Models

SINet-V2 and CamoFormer have separate train/test scripts for standard MCOD fine-tuning and native 8-band MSI input experiments:

```bash
python SINet-V2/train_mcod.py
python SINet-V2/train_mcod_msi8.py
python CamoFormer/train_mcod.py
python CamoFormer/train_mcod_msi8.py
python ZoomNet/main.py --config ZoomNet/configs/zoomnet/mcod_zoomnet.py
python ZoomNet/main.py --config ZoomNet/configs/zoomnet/mcod_zoomnet_msi8.py
```

Check each script before running to confirm dataset roots, checkpoint paths, batch size, and GPU settings for your machine.

## GitHub Upload Notes

Before pushing, create an empty GitHub repository, then connect this local repo to it:

```bash
git remote add origin https://github.com/YOUR_USERNAME/MCOD_FM_robust.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` and the repository name with the actual GitHub repository URL.
