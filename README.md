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
- **MOMENT-1-Large**: State-of-the-art Foundation Model (LTM) used for large-scale transfer learning comparison.

### Transfer Strategies
- **Scratch**: Train full model from random init.
- **Head-only**: Freeze backbone, train classifier head only.
- **Late encoders**: Unfreeze last transformer layers + head.
- **Full fine-tune**: Unfreeze all with smaller LR for backbone.

### Hyperparameters
Tuned with Optuna, **one study per strategy** to avoid cross-strategy bias. Best configs are saved as YAML in `configs/` and reused by evaluation notebooks.

## Technical Analysis (Transformer Transfer)
### IndPatchTST Backbone
The model uses a channel-independent Transformer with patching. Each sample is split into temporal patches (`patch_len`, `stride`), then encoded by a Transformer stack. The backbone is pretrained on ETTh1 regression to learn generic temporal structure before being adapted to LSST classification. The implementation is in `src/models/indpatchtst.py`.

### Classification Head
The classifier wraps the backbone and replaces the regression head with a compact MLP:
`LayerNorm -> Linear(d_model, hidden_dim) -> GELU -> Dropout -> Linear(hidden_dim, n_classes)`.  
See `src/models/indpatchtst_classifier.py`.

### Freezing Strategies (What Is Frozen / Unfrozen)
The backbone is split into groups to control transfer:
- **Embedding group**: patch projection (low-level features).
- **Early encoder group**: first half of Transformer layers.
- **Late encoder group**: second half of Transformer layers.
- **Classifier**: always trainable.
  
Strategies:
- **Scratch**: all parameters trainable (no pretrained weights).
- **Head-only**: backbone fully frozen, train classifier only.
- **Late encoders**: unfreeze only late encoder layers + classifier.
- **Full fine-tune**: unfreeze full backbone + classifier with lower LR on backbone.

These behaviors are implemented in `IndPatchTSTClassifier.freeze_all_backbone`, `unfreeze_late_encoders`, and `unfreeze_all`.

### Comparison With Baseline
The CNN baseline is a lightweight 1D convolutional model trained on the same LSST preprocessing and split logic. This provides a reference point for whether transfer learning adds value beyond a simple architecture.  
Baseline model: `src/models/cnn_baseline.py`  
Baseline training/eval: `src/training/trainer_cnn.py`

### Extension: Large-Scale Foundation Model (MOMENT)
As a benchmark against our ETTh1-to-LSST transfer, we adapted **MOMENT-1-Large**, a Foundation Model pre-trained on 1.1 billion observations. 
- **Strategy**: Full fine-tuning with differential learning rates ($\eta_{backbone}=10^{-5}, \eta_{head}=10^{-4}$).
- **Outcome**: Achieved a test accuracy of **56.24%**, validating that Transformer architectures significantly outperform CNNs for this task.
- **Analysis**: While the model excels at capturing majority patterns (Class 1 F1-score: **0.85**), it remains sensitive to the extreme class imbalance of LSST (macro-F1: **0.321**), highlighting that raw model power cannot fully compensate for domain-specific data scarcity.

### Evaluation Protocol
- Unified LSST preprocessing (padding to window=36, standardization on train only).
- Stratified train/val split with fixed seeds.
- Report accuracy and macro-F1 over **15 runs** to capture variance.
This is orchestrated in `src/training/adapting_to_classification.py`.

## Repository Structure
- `src/` code for models, training, and data.
- `notebooks/` EDA and training notebooks.
- `configs/` YAMLs for pretrained and fine-tuning settings.
- `artifacts/` generated outputs (models, reports, experiments).
- `data/` local datasets (not tracked).

## Code Structure (Where to Find Things)
### Data
- `src/data/dataloader.py`: `build_lsst_dataloaders`, `build_etth1_dataloaders`, `pad_truncate`, `LSST_WINDOW`.

### Models
- `src/models/indpatchtst.py`: backbone definition + ETTh1 pretraining entry point (saves `artifacts/models/best_indpatch_tst_optuna.pth`).
- `src/models/indpatchtst_classifier.py`: classifier wrapper and freezing strategies.
- `src/models/cnn_baseline.py`: CNN baseline model.

### Training / Evaluation
- `src/training/train_indpatchtst_reg.py`: ETTh1 regression training loop.
- `src/training/train_indpatchtst_class.py`: LSST classification training/eval utilities (`train_loop`, `evaluate`).
- `src/training/optuna_search.py`: Optuna objectives for each strategy.
- `src/training/adapting_to_classification.py`: main experiment runner, config I/O, multi-run stats.
- `src/training/indpatchtst_clf_utils.py`: helper functions for augmentation and model assembly.
- `src/training/trainer_cnn.py`: CNN baseline training loop.

### Notebooks (Repro Path)
- `notebooks/01_eda.ipynb`: data exploration.
- `notebooks/02_train_cnn.ipynb`: CNN baseline + stats.
- `notebooks/03_pretrain_indpatchtst.ipynb`: Optuna pretraining (ETTh1).
- `notebooks/04_train_indpatchtst_classifier.ipynb`: transfer strategies + stats.
- `notebooks/05_MOMENT_foundation_model.ipynb`: MOMENT-1-Large (Linear Probing vs Full Fine-Tuning).


## Generated Files (What Appears Where)
- `configs/*.yml`: Optuna best configs for backbone and fine-tuning.
- `artifacts/models/best_indpatch_tst_optuna.pth`: pretrained backbone checkpoint.
- `artifacts/reports/`: aggregated outputs (if generated).
- `lsst_statistics_complete.pth`: saved in the current working directory when running `adapting_to_classification.py` as a script.

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
- `05_MOMENT_foundation_model.ipynb`: Advanced experiment using MOMENT-1-Large (Linear Probing vs Full Fine-Tuning).

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
- Class Imbalance: LSST is highly imbalanced. Models (including MOMENT) tend to favor majority classes with clear periodic patterns (like Class 1), leading to a high gap between Accuracy and Macro-F1.
- Computational Cost: Fine-tuning large-scale models like MOMENT requires significant resources (e.g., ~30 hours).

## What to Expect
The main value is the **analysis of transfer strategies and their rationale**, not the absolute performance. Results should be interpreted in terms of stability and generalization rather than single-run peaks.
