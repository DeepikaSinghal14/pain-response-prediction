"""Figures for the README. One consistent, colour-blind-safe style."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

LABELS = {
    "pcs": "Pain catastrophising (PCS)", "il6_log": "IL-6", "dass_dep": "Depression (DASS-21)",
    "dass_anx": "Anxiety (DASS-21)", "dass_str": "Stress (DASS-21)", "psqi": "Sleep quality (PSQI)",
    "pss10": "Perceived stress (PSS-10)", "paindetect": "painDETECT", "pain_nrs": "Baseline pain (NRS)",
    "crp_log": "CRP", "tnfa_log": "TNF-α", "age": "Age", "sex_female": "Sex (female)", "bmi": "BMI",
    "ppt": "QST: pressure pain threshold", "cpt": "QST: cold pain threshold", "hpt": "QST: heat pain threshold",
    "cdt": "QST: cold detection", "wdt": "QST: warm detection", "mdt": "QST: mechanical detection",
    "mps": "QST: mechanical pain sensitivity", "wur": "QST: wind-up ratio",
}
BLUE, ORANGE, GREY, DARK ="#2F6DB5", "#E08A2E", "#9AA3AD", "#1F2933"
plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GREY, "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def ablation(results: dict, path):
    names = list(results)
    means = [results[n]["auc_mean"] for n in names]
    sds = [results[n]["auc_sd"] for n in names]
    fig, ax = plt.subplots(figsize=(6, 3))
    y = np.arange(len(names))[::-1]
    ax.errorbar(means, y, xerr=sds, fmt="o", color=BLUE, ecolor=GREY, capsize=3, ms=7)
    for yi, m in zip(y, means):
        ax.text(m, yi + 0.22, f"{m:.2f}", ha="center", color=DARK, fontsize=9)
    ax.set_yticks(y, names)
    ax.axvline(0.5, color=GREY, ls=":", lw=1)
    ax.set_xlim(0.45, 0.9)
    ax.set_ylim(-0.6, len(names) - 0.3)
    ax.set_xlabel("ROC AUC (nested CV, mean ± SD over 15 outer folds)")
    ax.set_title("Each added data modality improves response prediction", loc="left", fontsize=11, color=DARK)
    fig.savefig(path)
    plt.close(fig)


def roc_and_calibration(y, probs: dict, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.6))
    for (name, p), c in zip(probs.items(), [GREY, BLUE, ORANGE]):
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, color=c, lw=2, label=name)
        frac, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
        a2.plot(mean_pred, frac, "o-", color=c, lw=1.5, ms=4, label=name)
    for a in (a1, a2):
        a.plot([0, 1], [0, 1], color=GREY, ls=":", lw=1)
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
    a1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (out-of-fold)")
    a2.set(xlabel="Predicted probability", ylabel="Observed responder rate", title="Calibration")
    a1.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def importance(imp: pd.Series, modality: dict, path, top: int = 12):
    imp = imp.head(top)[::-1]
    colours = {"clinical": GREY, "questionnaire": BLUE, "qst": ORANGE, "biomarker": "#4C9A6A"}
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh([LABELS.get(v, v) for v in imp.index], imp.values, color=[colours[modality[v]] for v in imp.index])
    ax.set_xlabel("Drop in ROC AUC when feature is permuted")
    ax.set_title("What drives the prediction (permutation importance)", loc="left", fontsize=11, color=DARK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours.values()]
    ax.legend(handles, ["Clinical", "Questionnaire", "QST", "Biomarker"], frameon=False, fontsize=8, loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def cluster_profiles(profiles: pd.DataFrame, qst_vars, path):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(qst_vars))
    palette = [ORANGE, BLUE, GREY, "#4C9A6A", DARK, "#B5487A"]
    for (name, row), c in zip(profiles.iterrows(), palette):
        label = f"{name} (n={int(row['n'])}, responders {row['responder_rate']:.0%})"
        ax.plot(x, row[qst_vars].to_numpy(dtype=float), "o-", color=c, lw=2, label=label)
    ax.axhline(0, color=GREY, lw=1, ls=":")
    ax.set_xticks(x, [v.upper() for v in qst_vars])
    ax.set_ylabel("Mean QST z-score")
    ax.set_title("Sensory subgroups found without using the outcome", loc="left", fontsize=11, color=DARK)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    fig.savefig(path)
    plt.close(fig)
