"""
adapting_to_classification.py — Fine-tuning IndPatchTST (ETTh1→LSST)
"""

import os
from collections import defaultdict

import numpy as np
import optuna
import torch
import torch.nn as nn
import yaml

from src.data.dataloader import build_lsst_dataloaders
from src.models.indpatchtst_classifier import IndPatchTSTClassifier
from src.training.optuna_search import (
    objective_full_tune,
    objective_head_only,
    objective_late_enc,
    objective_scratch,
)
from src.training.train_indpatchtst_class import evaluate, train_loop
from src.training.indpatchtst_clf_utils import augment_batch, build_clf_model


LSST_WINDOW = 36

CONFIG_DIR = "configs"

# DEFAULTS OPTUNA (au cas où les YAMLs n'existent pas, pour debug ou runs rapides)
DEFAULT_PARAMS_HEAD_ONLY = {
    "hidden_dim": 64,
    "dropout_clf": 0.4,
    "lr_head": 1e-3,
    "weight_decay": 1e-4,
}
DEFAULT_PARAMS_LATE_ENC = {
    "hidden_dim": 64,
    "dropout_clf": 0.4,
    "lr_late": 2e-4,
    "lr_head": 1e-3,
    "weight_decay": 1e-4,
}
DEFAULT_PARAMS_FULL_TUNE = {
    "hidden_dim": 64,
    "dropout_clf": 0.4,
    "lr_backbone": 2e-4,
    "lr_head": 1e-3,
    "weight_decay": 1e-4,
}
DEFAULT_SCRATCH_CONFIG = {
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 4,
    "d_ff": 256,
    "dropout": 0.2,
    "revin": False,
    "patch_len": 10,
    "stride": 3,
}
DEFAULT_SCRATCH_TRAIN_PARAMS = {
    "hidden_dim": 64,
    "dropout_clf": 0.4,
    "lr": 1e-3,
    "weight_decay": 1e-4,
}


def _load_yaml(path, default=None, *, required=False):
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(
                f"Missing required config: {path}. "
                "Run indpatchtst pretraining to generate configs/backbone.yml."
            )
        return default
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or default


def load_run_configs(config_dir=CONFIG_DIR):
    """Charge les configs Optuna depuis YAMLs (fallback sur defaults)."""
    backbone_cfg = _load_yaml(os.path.join(config_dir, "backbone.yml"), required=True)
    scratch_cfg = _load_yaml(
        os.path.join(config_dir, "scratch_backbone.yml"), DEFAULT_SCRATCH_CONFIG
    )
    params_head_only = _load_yaml(
        os.path.join(config_dir, "params_head_only.yml"), DEFAULT_PARAMS_HEAD_ONLY
    )
    params_late_enc = _load_yaml(
        os.path.join(config_dir, "params_late_enc.yml"), DEFAULT_PARAMS_LATE_ENC
    )
    params_full_tune = _load_yaml(
        os.path.join(config_dir, "params_full_tune.yml"), DEFAULT_PARAMS_FULL_TUNE
    )
    scratch_train_params = _load_yaml(
        os.path.join(config_dir, "scratch_train.yml"), DEFAULT_SCRATCH_TRAIN_PARAMS
    )
    return (
        backbone_cfg,
        scratch_cfg,
        params_head_only,
        params_late_enc,
        params_full_tune,
        scratch_train_params,
    )


# AUGMENTATION


# ══════════════════════════════════════════════════════════════════════════════
#  OPTUNA — Une étude par stratégie pré-entraînée
#
#  Pourquoi une étude par stratégie ?
#  lr_backbone n'a AUCUN signal dans objective_clf (backbone gelé) → Optuna
#  choisit une valeur arbitraire dans [1e-6, 1e-4], qui se retrouve ensuite
#  utilisée pour late_enc et full_tune où elle est critique. Chaque stratégie
#  a sa propre dynamique d'entraînement et nécessite ses propres LR.
# ══════════════════════════════════════════════════════════════════════════════


#  EXPÉRIENCE UNIQUE
def run_single_experiment(
    seed,  # ← Nouveau : seed pour reproductibilité
    train_dl,  # ← Directement les DataLoaders
    val_dl,
    test_dl,
    backbone_config,
    scratch_config,
    params_head_only,
    params_late_enc,
    params_full_tune,
    best_scratch_params,
    n_classes,
    n_features,
    LSST_WINDOW,
    device,
    scaler_amp,
):
    """
    Un run complet : 4 stratégies comparées avec leurs hyperparamètres dédiés.

    Chaque stratégie (B, C, D) dispose de ses propres LR issus d'une étude
    Optuna indépendante — sans biais croisé entre les stratégies.
    Chaque stratégie conserve son niveau de gel pendant TOUS ses epochs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    run_results = {}

    strategies = {
        "A_scratch": ("From Scratch (baseline)", None, "scratch", 60),
        "B_head_only": (
            "Head Only",
            "artifacts/models/best_indpatch_tst_optuna.pth",
            "head_only",
            60,
        ),
        "C_late_enc": (
            "Late Encoders + Head",
            "artifacts/models/best_indpatch_tst_optuna.pth",
            "late_enc",
            50,
        ),
        "D_full_tune": (
            "Full Fine-tune (lr différenciés)",
            "artifacts/models/best_indpatch_tst_optuna.pth",
            "full",
            40,
        ),
    }

    for key, (label, pretrained_path, strategy, n_ep) in strategies.items():
        print(f"\n{'─' * 55}")
        print(f"  Stratégie : {label}")
        print(f"{'─' * 55}")

        # ── Config et hyperparamètres dédiés à la stratégie ─────────────────
        if strategy == "scratch":
            cfg = scratch_config
            hd = best_scratch_params["hidden_dim"]
            do = best_scratch_params["dropout_clf"]
        elif strategy == "head_only":
            cfg = backbone_config
            hd = params_head_only["hidden_dim"]
            do = params_head_only["dropout_clf"]
        elif strategy == "late_enc":
            cfg = backbone_config
            hd = params_late_enc["hidden_dim"]
            do = params_late_enc["dropout_clf"]
        else:  # full
            cfg = backbone_config
            hd = params_full_tune["hidden_dim"]
            do = params_full_tune["dropout_clf"]

        model = IndPatchTSTClassifier(
            LSST_WINDOW,
            n_features,
            n_classes,
            cfg,
            pretrained_model_path=pretrained_path,
        ).to(device)
        model.classifier = nn.Sequential(
            nn.LayerNorm(cfg["d_model"]),
            nn.Linear(cfg["d_model"], hd),
            nn.GELU(),
            nn.Dropout(do),
            nn.Linear(hd, n_classes),
        ).to(device)

        # ── Optimiseur avec LR dédiés — gel fixe jusqu'au bout ───────────────
        print(f"\n  ▶ Entraînement ({n_ep} epochs)")
        if strategy == "head_only":
            model.freeze_all_backbone()
            opt1 = torch.optim.AdamW(
                model.classifier.parameters(),
                lr=params_head_only["lr_head"],
                weight_decay=params_head_only["weight_decay"],
            )
        elif strategy == "late_enc":
            model.unfreeze_late_encoders()
            opt1 = torch.optim.AdamW(
                [
                    {
                        "params": model.group_enc_late.parameters(),
                        "lr": params_late_enc["lr_late"],
                    },
                    {
                        "params": model.classifier.parameters(),
                        "lr": params_late_enc["lr_head"],
                    },
                ],
                weight_decay=params_late_enc["weight_decay"],
            )
        elif strategy == "full":
            model.unfreeze_all()
            opt1 = torch.optim.AdamW(
                [
                    {
                        "params": model.backbone.parameters(),
                        "lr": params_full_tune["lr_backbone"],
                    },
                    {
                        "params": model.classifier.parameters(),
                        "lr": params_full_tune["lr_head"],
                    },
                ],
                weight_decay=params_full_tune["weight_decay"],
            )
        else:  # scratch
            model.unfreeze_all()
            opt1 = torch.optim.AdamW(
                model.parameters(),
                lr=best_scratch_params["lr"],
                weight_decay=best_scratch_params["weight_decay"],
            )

        sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=n_ep)
        train_loop(
            model,
            train_dl,
            val_dl,
            opt1,
            criterion,
            n_ep,
            device,
            scheduler=sch1,
            augment=(strategy not in ("full", "scratch")),
            patience=12,
            scaler_amp=scaler_amp,
        )

        acc, f1 = evaluate(model, test_dl, device)
        run_results[key] = {"label": label, "acc": acc, "f1": f1}
        print(f"\n  ✅ [{key}] Acc={acc:.4f} | F1={f1:.4f}")

    return run_results


# ══════════════════════════════════════════════════════════════════════════════
#  STATISTIQUES MULTI-RUNS
# ══════════════════════════════════════════════════════════════════════════════


def run_statistics(n_runs=15, base_seed=0):
    """
    15 runs statistiques avec build_lsst_dataloaders (comme baseline.py).
    Compatible avec le dataloader unifié (normalisation, padding, split stratify).
    """
    all_results = defaultdict(lambda: {"acc": [], "f1": []})

    (
        backbone_cfg,
        scratch_cfg,
        params_head_only,
        params_late_enc,
        params_full_tune,
        best_scratch_train_params,
    ) = load_run_configs()

    for i in range(n_runs):
        seed = base_seed + i
        print(f"\n{'#' * 60}")
        print(f"  RUN {i + 1}/{n_runs} (seed={seed})")
        print(f"{'#' * 60}")

        # ✅ Appel direct à build_lsst_dataloaders (comme dans baseline.py)
        train_dl, val_dl, test_dl, scaler, le, n_classes, n_features = build_lsst_dataloaders(
            seed=seed, batch_size=32
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Les 4 stratégies TST avec leurs hyperparams fixes ─────────────────
        run_results = run_single_experiment(
            seed=seed,
            train_dl=train_dl,
            val_dl=val_dl,
            test_dl=test_dl,
            n_classes=n_classes,
            n_features=n_features,
            LSST_WINDOW=LSST_WINDOW,
            device=device,
            scaler_amp=torch.amp.GradScaler(enabled=device.type == "cuda"),
            backbone_config=backbone_cfg,
            scratch_config=scratch_cfg,
            params_head_only=params_head_only,
            params_late_enc=params_late_enc,
            params_full_tune=params_full_tune,
            best_scratch_params=best_scratch_train_params,
        )

        # ── Stockage des résultats par stratégie ──────────────────────────────
        for key, metrics in run_results.items():
            all_results[key]["acc"].append(metrics["acc"])
            all_results[key]["f1"].append(metrics["f1"])
            all_results[key]["label"] = metrics["label"]

    return all_results


def print_statistics(all_results, baseline=0.40):
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║            RÉSULTATS STATISTIQUES (n=15 runs, seed=0..14)               ║")
    print("╠══════════════════════════════╦══════════════════╦══════════════════╦═════╣")
    print("║ Stratégie                    ║ Acc (μ ± σ)      ║ F1  (μ ± σ)      ║ Δ   ║")
    print("╠══════════════════════════════╬══════════════════╬══════════════════╬═════╣")

    summary = {}
    for key, data in all_results.items():
        accs = np.array(data["acc"])
        f1s = np.array(data["f1"])
        summary[key] = {
            "label": data["label"],
            "acc_mean": accs.mean(),
            "acc_std": accs.std(),
            "acc_min": accs.min(),
            "acc_max": accs.max(),
            "f1_mean": f1s.mean(),
            "f1_std": f1s.std(),
        }
        delta = accs.mean() - baseline
        sign = "+" if delta >= 0 else ""
        print(
            f"║ {data['label']:<28} ║ {accs.mean():.3f} ± {accs.std():.3f}  "
            f"║ {f1s.mean():.3f} ± {f1s.std():.3f}  ║{sign}{delta:+.3f}║"
        )

    print("╠══════════════════════════════╩══════════════════╩══════════════════╩═════╣")
    best_key = max(summary, key=lambda k: summary[k]["acc_mean"])
    s = summary[best_key]
    print(f"║ 🏆 Meilleur : {s['label']:<57}║")
    print(
        f"║    Acc = {s['acc_mean']:.3f} ± {s['acc_std']:.3f}"
        f"  [min={s['acc_min']:.3f}, max={s['acc_max']:.3f}]{'':>14}║"
    )
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    return summary


#  CONFIG BACKBONE
# Le backbone est celui généré par indpatchtst (configs/backbone.yml).

if __name__ == "__main__":
    train_dl0, val_dl0, test_dl0, scaler, le, n_classes, n_features = build_lsst_dataloaders(seed=0)

    os.makedirs("artifacts/models", exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    use_amp = device.type == "cuda"
    scaler_amp = torch.amp.GradScaler() if use_amp else None
    print(f"Mixed Precision AMP : {'activé ⚡' if use_amp else 'désactivé'}")

    # ── Optuna — 1 étude par stratégie pré-entraînée + 1 pour scratch ────────
    def _make_study(seed=42):
        return optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        )

    backbone_cfg = _load_yaml(os.path.join(CONFIG_DIR, "backbone.yml"), required=True)

    opt_args = dict(
        train_dl=train_dl0,
        val_dl=val_dl0,
        device=device,
        window=LSST_WINDOW,
        n_features=n_features,
        n_classes=n_classes,
        backbone_config=backbone_cfg,
        scaler_amp=scaler_amp,
    )

    print("\n── Optuna B — Head Only (25 trials) ──")
    study_b = _make_study()
    study_b.optimize(lambda t: objective_head_only(t, **opt_args), n_trials=25)
    params_head_only = study_b.best_params
    print(f"  ✅ Best acc={study_b.best_value:.4f} | {params_head_only}")

    print("\n── Optuna C — Late Encoders (25 trials) ──")
    study_c = _make_study()
    study_c.optimize(lambda t: objective_late_enc(t, **opt_args), n_trials=25)
    params_late_enc = study_c.best_params
    print(f"  ✅ Best acc={study_c.best_value:.4f} | {params_late_enc}")

    print("\n── Optuna D — Full Fine-tune (25 trials) ──")
    study_d = _make_study()
    study_d.optimize(lambda t: objective_full_tune(t, **opt_args), n_trials=25)
    params_full_tune = study_d.best_params
    print(f"  ✅ Best acc={study_d.best_value:.4f} | {params_full_tune}")

    print("\n── Optuna A — From Scratch backbone+tête (40 trials) ──")
    study_scratch = _make_study()
    study_scratch.optimize(
        lambda t: objective_scratch(
            t,
            train_dl0,
            val_dl0,
            device,
            LSST_WINDOW,
            n_features,
            n_classes,
            scaler_amp,
        ),
        n_trials=40,
    )
    best_scratch_params_all = study_scratch.best_params
    print(f"  ✅ Best acc={study_scratch.best_value:.4f} | {best_scratch_params_all}")

    scratch_config = {
        k: best_scratch_params_all[k]
        for k in [
            "d_model",
            "n_heads",
            "n_layers",
            "d_ff",
            "dropout",
            "patch_len",
            "stride",
        ]
    }
    scratch_config["revin"] = False
    best_scratch_train_params = {
        k: best_scratch_params_all[k] for k in ["hidden_dim", "dropout_clf", "lr", "weight_decay"]
    }

    # Sauvegarde des configs en YAML pour réutilisation dans run_statistics
    with open(os.path.join(CONFIG_DIR, "scratch_backbone.yml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(scratch_config, f, sort_keys=False)
    with open(os.path.join(CONFIG_DIR, "scratch_train.yml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(best_scratch_train_params, f, sort_keys=False)
    with open(os.path.join(CONFIG_DIR, "params_head_only.yml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(params_head_only, f, sort_keys=False)
    with open(os.path.join(CONFIG_DIR, "params_late_enc.yml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(params_late_enc, f, sort_keys=False)
    with open(os.path.join(CONFIG_DIR, "params_full_tune.yml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(params_full_tune, f, sort_keys=False)

    # ── 15 runs statistiques ─────────────────────────────────────────────────
    print("\n── Lancement 15 runs statistiques ──")
    all_results = run_statistics(n_runs=15, base_seed=0)
    summary = print_statistics(all_results, baseline=0.546)

    torch.save(
        {
            "summary": summary,
            "all_results": dict(all_results),
            "backbone_config": backbone_cfg,
            "scratch_config": scratch_config,
            "params_head_only": params_head_only,
            "params_late_enc": params_late_enc,
            "params_full_tune": params_full_tune,
            "scratch_train_params": best_scratch_train_params,
            "scaler_mean": scaler.mean_,
            "scaler_std": scaler.scale_,
            "le_classes": le.classes_,
        },
        "lsst_statistics_complete.pth",
    )
    print("\n💾 Sauvegardé : lsst_statistics_complete.pth")
