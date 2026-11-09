"""Simulate a multimodal chronic-pain cohort with a treatment-response outcome.

The data are synthetic. They mirror the *structure* of a clinical pain study
(demographics, quantitative sensory testing, validated questionnaires, a serum
biomarker panel and a pre/post pain rating), not the values of any real cohort.

Ground truth that the pipeline should be able to recover:
  * three latent sensory phenotypes, defined by QST profiles
  * treatment response (>= 30 % pain reduction, IMMPACT threshold) that depends
    on phenotype, catastrophising, depression, sleep and an inflammatory marker
"""
from __future__ import annotations

import numpy as np
import pandas as pd

QST_VARS = ["cdt", "wdt", "cpt", "hpt", "mdt", "mps", "ppt", "wur"]
QUESTIONNAIRE_VARS = ["pain_nrs", "paindetect", "dass_dep", "dass_anx", "dass_str", "pcs", "psqi", "pss10"]
BIOMARKER_VARS = ["crp_log", "il6_log", "tnfa_log"]
CLINICAL_VARS = ["age", "sex_female", "bmi"]

# QST z-score centroids per latent phenotype (positive = gain of function / more sensitive)
PHENOTYPES = {
    "sensitised":      dict(cdt=0.3, wdt=0.3, cpt=1.2, hpt=1.3, mdt=0.4, mps=1.5, ppt=1.4, wur=1.0),
    "intermediate":    dict(cdt=0.0, wdt=0.0, cpt=0.4, hpt=0.4, mdt=0.0, mps=0.5, ppt=0.5, wur=0.3),
    "low_sensitivity": dict(cdt=-1.1, wdt=-1.0, cpt=-0.3, hpt=-0.4, mdt=-1.2, mps=-0.2, ppt=-0.2, wur=-0.2),
}
PHENOTYPE_P = [0.35, 0.40, 0.25]
PHENOTYPE_EFFECT = {"sensitised": 0.9, "intermediate": 0.0, "low_sensitivity": -0.7}


def _clip(x, lo, hi):
    return np.clip(x, lo, hi)


def simulate_cohort(n: int = 400, seed: int = 42, missing_rate: float = 0.05) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(PHENOTYPES)
    pheno = rng.choice(names, size=n, p=PHENOTYPE_P)

    df = pd.DataFrame({"participant_id": [f"SIM{i:04d}" for i in range(1, n + 1)]})
    df["true_phenotype"] = pheno

    # Clinical / demographic
    df["age"] = _clip(rng.normal(48, 12, n), 18, 80).round(0)
    df["sex_female"] = rng.binomial(1, 0.7, n)
    df["bmi"] = _clip(rng.normal(26.5, 4.5, n), 17, 45).round(1)

    # QST z-scores: phenotype centroid + individual noise
    for v in QST_VARS:
        centre = np.array([PHENOTYPES[p][v] for p in pheno])
        df[v] = (centre + rng.normal(0, 0.7, n)).round(2)

    # Shared latent "distress" factor drives the psychological questionnaires
    distress = rng.normal(0, 1, n)
    sens = df[["mps", "ppt", "hpt"]].mean(axis=1).to_numpy()
    df["pain_nrs"] = _clip(5.5 + 0.6 * sens + 0.4 * distress + rng.normal(0, 1.2, n), 0, 10).round(1)
    df["paindetect"] = _clip(14 + 3 * sens + 1.5 * distress + rng.normal(0, 4, n), 0, 38).round(0)
    df["dass_dep"] = _clip(10 + 6 * distress + rng.normal(0, 4, n), 0, 42).round(0)
    df["dass_anx"] = _clip(8 + 5 * distress + rng.normal(0, 4, n), 0, 42).round(0)
    df["dass_str"] = _clip(14 + 5 * distress + rng.normal(0, 4, n), 0, 42).round(0)
    df["pcs"] = _clip(22 + 7 * distress + 2 * sens + rng.normal(0, 6, n), 0, 52).round(0)
    df["psqi"] = _clip(9 + 2 * distress + rng.normal(0, 2.5, n), 0, 21).round(0)
    df["pss10"] = _clip(19 + 4 * distress + rng.normal(0, 4, n), 0, 40).round(0)

    # Serum biomarkers (log scale); IL-6 mildly higher in sensitised phenotype and with BMI
    df["crp_log"] = (0.2 + 0.04 * (df["bmi"] - 26) + rng.normal(0, 0.6, n)).round(3)
    df["il6_log"] = (0.8 + 0.35 * (pheno == "sensitised") + 0.02 * (df["bmi"] - 26) + rng.normal(0, 0.45, n)).round(3)
    df["tnfa_log"] = (1.0 + rng.normal(0, 0.4, n)).round(3)

    # Treatment response: logistic model on standardised drivers
    z = lambda s: (s - s.mean()) / s.std()
    logit = (
        -0.45
        + np.array([PHENOTYPE_EFFECT[p] for p in pheno])
        - 0.55 * z(df["pcs"])
        - 0.35 * z(df["dass_dep"])
        - 0.30 * z(df["psqi"])
        + 0.35 * z(df["il6_log"])
        + 0.15 * df["sex_female"]
        + rng.normal(0, 0.6, n)  # unexplained heterogeneity
    )
    p_resp = 1 / (1 + np.exp(-logit))
    df["responder"] = rng.binomial(1, p_resp)

    # Continuous end-of-treatment pain, consistent with the responder label
    reduction = np.where(
        df["responder"] == 1,
        rng.uniform(0.30, 0.75, n),
        rng.uniform(-0.15, 0.29, n),
    )
    df["pain_nrs_eot"] = _clip(df["pain_nrs"] * (1 - reduction), 0, 10).round(1)
    df["pain_reduction_pct"] = np.where(df["pain_nrs"] > 0, 100 * (1 - df["pain_nrs_eot"] / df["pain_nrs"]), 0).round(1)

    # Missing at random in questionnaires and biomarkers (as in real cohorts)
    for v in QUESTIONNAIRE_VARS[1:] + BIOMARKER_VARS:
        mask = rng.random(n) < missing_rate
        df.loc[mask, v] = np.nan

    return df


DATA_DICTIONARY = [
    # name, modality, unit/scale, description
    ("participant_id", "id", "-", "Simulated participant identifier"),
    ("true_phenotype", "ground_truth", "category", "Latent sensory phenotype used to simulate data (not a model input)"),
    ("age", "clinical", "years", "Age at baseline"),
    ("sex_female", "clinical", "0/1", "Sex (1 = female)"),
    ("bmi", "clinical", "kg/m2", "Body-mass index"),
    ("cdt", "qst", "z-score", "Cold detection threshold"),
    ("wdt", "qst", "z-score", "Warm detection threshold"),
    ("cpt", "qst", "z-score", "Cold pain threshold"),
    ("hpt", "qst", "z-score", "Heat pain threshold"),
    ("mdt", "qst", "z-score", "Mechanical detection threshold"),
    ("mps", "qst", "z-score", "Mechanical pain sensitivity"),
    ("ppt", "qst", "z-score", "Pressure pain threshold"),
    ("wur", "qst", "z-score", "Wind-up ratio"),
    ("pain_nrs", "questionnaire", "0-10", "Baseline average pain intensity (NRS)"),
    ("paindetect", "questionnaire", "0-38", "painDETECT neuropathic component score"),
    ("dass_dep", "questionnaire", "0-42", "DASS-21 depression subscale"),
    ("dass_anx", "questionnaire", "0-42", "DASS-21 anxiety subscale"),
    ("dass_str", "questionnaire", "0-42", "DASS-21 stress subscale"),
    ("pcs", "questionnaire", "0-52", "Pain Catastrophizing Scale total"),
    ("psqi", "questionnaire", "0-21", "Pittsburgh Sleep Quality Index global score"),
    ("pss10", "questionnaire", "0-40", "Perceived Stress Scale (10 items)"),
    ("crp_log", "biomarker", "log mg/L", "C-reactive protein"),
    ("il6_log", "biomarker", "log pg/mL", "Interleukin-6"),
    ("tnfa_log", "biomarker", "log pg/mL", "Tumour necrosis factor alpha"),
    ("pain_nrs_eot", "outcome", "0-10", "End-of-treatment pain intensity (NRS)"),
    ("pain_reduction_pct", "outcome", "%", "Relative pain reduction from baseline"),
    ("responder", "outcome", "0/1", "Responder: >= 30 % pain reduction (IMMPACT)"),
]


def data_dictionary() -> pd.DataFrame:
    return pd.DataFrame(DATA_DICTIONARY, columns=["variable", "modality", "unit", "description"])
