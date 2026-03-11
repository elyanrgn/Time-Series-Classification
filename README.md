# Time-Series Transformers for LSST Classification 🪐⚡

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Étude comparative CNN vs Transformer (IndPatchTST) pour classification multivariée de séries temporelles astronomiques (LSST UCR/UEA).**

## 🎯 Résumé des résultats (15 runs)

| Stratégie | Acc (μ±σ) | F1-macro (μ±σ) | Δ vs CNN |
|-----------|-----------|----------------|----------|
| **CNN Baseline** | **0.540±0.043** | **0.395±0.039** | - |
| TST From Scratch | **0.596±0.014** | 0.374±0.029 | **+0.056** |
| Head Only (ETTh1) | 0.315±0.000 | 0.034±0.000 | -0.225 |
| Late Enc + Head | 0.371±0.002 | 0.120±0.006 | -0.169 |
| Full Fine-tune | 0.537±0.013 | 0.282±0.012 | -0.003 |

**Conclusion :** L'architecture Transformer surpasse le CNN, mais le pré-entraînement ETTh1→LSST **n'améliore pas** les performances (mismatch domaine).

## 🚀 Installation rapide

```bash
# 1. Clone + env
git clone <repo> && cd time-series-transformers
conda create -n ts-lsst python=3.10 -y
conda activate ts-lsst
pip install -r requirements.txt

# 2. Données UCR/UEA (auto-téléchargées)
