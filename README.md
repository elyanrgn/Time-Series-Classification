# Time-Series Transformers for LSST Classification

## Project Goal
Study the effect of pretraining an IndPatchTST backbone on ETTh1 (regression) and transferring it to LSST time-series classification. The objective is not to maximize accuracy, but to explain and justify each design choice (freezing strategy, hyperparameters, data handling, and evaluation).

## Research Question
Does pretraining on ETTh1 improve LSST classification compared to training from scratch, and under which fine-tuning strategy?

## Key Ideas
- ETTh1 provides long, regular multivariate sequences that can teach temporal patterns.
- LSST is a smaller, noisy, multivariate classification dataset with a different domain.
- Transfer is tested through controlled strategies: head-only, late-encoder tuning, full fine-tune, and scratch.

## Methods
### Datasets
- **ETTh1**: Used for regression pretraining.
- **LSST (UCR/UEA)**: Used for classification.

### Models
- **CNN baseline**: Simple 1D conv model for reference.
- **IndPatchTST**: Channel-independent transformer with patching.
- **IndPatchTSTClassifier**: Pretrained backbone + classification head.

### Transfer Strategies
- **Scratch**: Train full model from random init.
- **Head-only**: Freeze backbone, train classifier head only.
- **Late encoders**: Unfreeze last transformer layers + head.
- **Full fine-tune**: Unfreeze all with smaller LR for backbone.

### Hyperparameters
Tuned with Optuna, **one study per strategy** to avoid cross-strategy bias. Best configs are saved as YAML in `configs/` and reused by evaluation notebooks.

## Repository Structure
- `src/` code for models, training, and data.
- `notebooks/` EDA and training notebooks.
- `configs/` YAMLs for pretrained and fine-tuning settings.
- `artifacts/` generated outputs (models, reports, experiments).
- `data/` local datasets (not tracked).

## Setup
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows
pip install -r requirements.txt
pip install -e .
```

## Data
- **ETTh1**: place `ETTh1.csv` in `data/`.
- **LSST**: auto-downloaded via `tslearn` on first run.

## Reproducibility
### Notebooks
- `01_eda.ipynb`: dataset exploration
- `02_train_cnn.ipynb`: CNN baseline (multi-run stats)
- `03_pretrain_indpatchtst.ipynb`: pretraining with Bayesian search
- `04_train_indpatchtst_classifier.ipynb`: multi-run stats using YAML configs

### Scripts
- Pretraining: `python -m src.models.indpatchtst`
- Transfer experiments: `python -m src.training.adapting_to_classification`

## Outputs
- Models are saved under `artifacts/models/`.
- Reports and aggregated results under `artifacts/reports/`.

## Limitations
- Domain shift between ETTh1 (energy/meteorological signals) and LSST (astronomy).
- LSST is small; variance is high. We report means and standard deviations across multiple seeds.
- Pretraining is regression while target is classification, which may limit transfer.

## What to Expect
The main value is the **analysis of transfer strategies and their rationale**, not the absolute performance. Results should be interpreted in terms of stability and generalization rather than single-run peaks.
