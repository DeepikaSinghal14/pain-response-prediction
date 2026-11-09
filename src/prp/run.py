"""Run the full pipeline: simulate -> stratify -> predict -> report.

Usage:  python -m prp.run --config config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from . import plots
from .model import FEATURE_SETS, boosting_model, logistic_model, nested_cv
from .simulate import QST_VARS, data_dictionary, simulate_cohort
from .stratify import stratify


def main(config_path: str = "config.yaml") -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text())
    seed, out = cfg["seed"], Path(cfg["output_dir"])
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(exist_ok=True)

    # 1. Data
    df = simulate_cohort(n=cfg["n_participants"], seed=seed, missing_rate=cfg["missing_rate"])
    df.to_csv("data/simulated_cohort.csv", index=False)
    dd = data_dictionary()
    dd.to_csv("data/data_dictionary.csv", index=False)
    print(f"Simulated {len(df)} participants, responder rate {df['responder'].mean():.1%}")

    # 2. Stratification (outcome not used)
    strat = stratify(df, seed=seed)
    strat["profiles"].to_csv(out / "cluster_profiles.csv")
    plots.cluster_profiles(strat["profiles"], QST_VARS, fig_dir / "cluster_profiles.png")
    print(f"Stratification: k={strat['chosen_k']}, ARI vs ground truth={strat['adjusted_rand_vs_truth']:.2f}")

    # 3. Prediction: modality ablation with regularised logistic regression
    cv = dict(seed=seed, n_splits=cfg["cv"]["outer_folds"], n_repeats=cfg["cv"]["repeats"])
    ablation = {}
    for name, feats in FEATURE_SETS.items():
        ablation[name] = nested_cv(df, feats, logistic_model, importance=(name == "+ Biomarkers (all)"), **cv)
        print(f"  {name:<20} AUC {ablation[name]['auc_mean']:.3f} ± {ablation[name]['auc_sd']:.3f}")
    all_feats = FEATURE_SETS["+ Biomarkers (all)"]
    gbm = nested_cv(df, all_feats, boosting_model, **cv)
    print(f"  {'Gradient boosting':<20} AUC {gbm['auc_mean']:.3f} ± {gbm['auc_sd']:.3f}")

    # 4. Figures
    y = df["responder"].to_numpy()
    plots.ablation(ablation, fig_dir / "modality_ablation.png")
    plots.roc_and_calibration(y, {
        "Clinical only": ablation["Clinical"]["oof_prob"],
        "Multimodal logistic": ablation["+ Biomarkers (all)"]["oof_prob"],
        "Gradient boosting": gbm["oof_prob"],
    }, fig_dir / "roc_calibration.png")
    modality = dict(zip(dd["variable"], dd["modality"]))
    imp = ablation["+ Biomarkers (all)"]["importance"]
    plots.importance(imp, modality, fig_dir / "feature_importance.png")

    # 5. Machine-readable results
    metrics = {
        "n_participants": len(df),
        "responder_rate": round(float(df["responder"].mean()), 3),
        "stratification": {
            "chosen_k": strat["chosen_k"],
            "silhouette_by_k": {int(k): round(v, 3) for k, v in strat["silhouette_by_k"].items()},
            "adjusted_rand_vs_truth": round(strat["adjusted_rand_vs_truth"], 3),
            "responder_rate_chi2_p": strat["responder_rate_chi2_p"],
        },
        "prediction": {
            **{f"logistic | {k}": {"auc_mean": round(v["auc_mean"], 3), "auc_sd": round(v["auc_sd"], 3),
                                   "brier": round(v["brier_mean"], 3)} for k, v in ablation.items()},
            "gradient_boosting | all": {"auc_mean": round(gbm["auc_mean"], 3), "auc_sd": round(gbm["auc_sd"], 3),
                                        "brier": round(gbm["brier_mean"], 3)},
        },
        "top_features": {k: round(float(v), 4) for k, v in imp.head(8).items()},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
