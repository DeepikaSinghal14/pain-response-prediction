"""Treatment-response prediction with nested cross-validation.

All preprocessing (imputation, scaling) sits inside the scikit-learn pipeline,
so it is re-fitted within every training fold and never sees test data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .simulate import BIOMARKER_VARS, CLINICAL_VARS, QST_VARS, QUESTIONNAIRE_VARS

# Cumulative modality sets used for the ablation (does adding a data type help?)
FEATURE_SETS = {
    "Clinical": CLINICAL_VARS + ["pain_nrs"],
    "+ Questionnaires": CLINICAL_VARS + QUESTIONNAIRE_VARS,
    "+ QST": CLINICAL_VARS + QUESTIONNAIRE_VARS + QST_VARS,
    "+ Biomarkers (all)": CLINICAL_VARS + QUESTIONNAIRE_VARS + QST_VARS + BIOMARKER_VARS,
}


def logistic_model(seed: int) -> GridSearchCV:
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    return GridSearchCV(
        pipe, {"clf__C": [0.01, 0.1, 1.0, 10.0]}, scoring="roc_auc",
        cv=StratifiedKFold(5, shuffle=True, random_state=seed),
    )


def boosting_model(seed: int) -> GridSearchCV:
    clf = HistGradientBoostingClassifier(random_state=seed)  # handles missing values natively
    return GridSearchCV(
        clf, {"learning_rate": [0.03, 0.1], "max_depth": [2, 3], "max_iter": [100, 200]},
        scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=seed),
    )


def nested_cv(df: pd.DataFrame, features: list[str], make_model, seed: int,
              n_splits: int = 5, n_repeats: int = 3, importance: bool = False) -> dict:
    X, y = df[features], df["responder"].to_numpy()
    outer = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    aucs, briers = [], []
    oof = np.zeros((n_repeats, len(y)))
    imps = []
    for i, (tr, te) in enumerate(outer.split(X, y)):
        model = make_model(seed).fit(X.iloc[tr], y[tr])
        prob = model.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y[te], prob))
        briers.append(brier_score_loss(y[te], prob))
        oof[i // n_splits, te] = prob
        if importance:
            pi = permutation_importance(model, X.iloc[te], y[te], scoring="roc_auc",
                                        n_repeats=10, random_state=seed)
            imps.append(pi.importances_mean)

    out = {
        "auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)),
        "brier_mean": float(np.mean(briers)),
        "fold_aucs": [float(a) for a in aucs],
        "oof_prob": oof.mean(axis=0),  # averaged over repeats, one value per participant
    }
    if importance:
        out["importance"] = (pd.Series(np.mean(imps, axis=0), index=features)
                             .sort_values(ascending=False))
    return out
