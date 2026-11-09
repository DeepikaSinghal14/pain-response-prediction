"""Unsupervised patient stratification on sensory (QST) profiles."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .simulate import QST_VARS


def stratify(df: pd.DataFrame, k_range=range(2, 7), seed: int = 42) -> dict:
    X = StandardScaler().fit_transform(df[QST_VARS])

    sil = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(X)
        sil[k] = float(silhouette_score(X, labels))
    best_k = max(sil, key=sil.get)

    km = KMeans(n_clusters=best_k, n_init=20, random_state=seed).fit(X)
    clusters = km.labels_

    # Order clusters by overall sensitivity so labels are stable and readable
    order = np.argsort(-df[QST_VARS].groupby(clusters).mean().mean(axis=1).to_numpy())
    remap = {old: new for new, old in enumerate(order)}
    clusters = np.array([remap[c] for c in clusters])

    profiles = df[QST_VARS].groupby(clusters).mean().round(2)
    profiles.index = [f"S{c + 1}" for c in profiles.index]
    resp = df.groupby(clusters)["responder"].agg(["size", "mean"])
    resp.index = profiles.index
    profiles["n"] = resp["size"]
    profiles["responder_rate"] = resp["mean"].round(3)

    chi2, p, _, _ = chi2_contingency(pd.crosstab(clusters, df["responder"]))

    out = {
        "silhouette_by_k": sil,
        "chosen_k": int(best_k),
        "adjusted_rand_vs_truth": float(adjusted_rand_score(df["true_phenotype"], clusters)),
        "responder_rate_chi2_p": float(p),
        "profiles": profiles,
        "labels": clusters,
    }
    return out
