import numpy as np
import pandas as pd

from prp.model import FEATURE_SETS, logistic_model, nested_cv
from prp.simulate import BIOMARKER_VARS, QUESTIONNAIRE_VARS, data_dictionary, simulate_cohort
from prp.stratify import stratify


def test_simulation_is_reproducible():
    a, b = simulate_cohort(n=100, seed=1), simulate_cohort(n=100, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_simulation_shape_and_ranges():
    df = simulate_cohort(n=200, seed=0)
    assert len(df) == 200
    assert df["participant_id"].is_unique
    assert df["pain_nrs"].between(0, 10).all()
    assert set(df["responder"].unique()) <= {0, 1}
    assert 0.2 < df["responder"].mean() < 0.8


def test_responder_label_matches_pain_reduction():
    df = simulate_cohort(n=300, seed=3)
    ok = df["pain_nrs"] > 0
    assert ((df.loc[ok, "pain_reduction_pct"] >= 29.5) == (df.loc[ok, "responder"] == 1)).mean() > 0.95


def test_missingness_only_where_expected():
    df = simulate_cohort(n=500, seed=2, missing_rate=0.1)
    with_missing = set(df.columns[df.isna().any()])
    assert with_missing <= set(QUESTIONNAIRE_VARS + BIOMARKER_VARS)


def test_data_dictionary_covers_every_column():
    df = simulate_cohort(n=10, seed=0)
    assert set(data_dictionary()["variable"]) == set(df.columns)


def test_ground_truth_never_used_as_feature():
    for feats in FEATURE_SETS.values():
        assert "true_phenotype" not in feats
        assert "responder" not in feats
        assert "pain_nrs_eot" not in feats and "pain_reduction_pct" not in feats


def test_stratification_finds_signal():
    df = simulate_cohort(n=300, seed=4)
    out = stratify(df, k_range=range(2, 4))
    assert out["adjusted_rand_vs_truth"] > 0.2


def test_shuffled_outcome_gives_chance_auc():
    # Guards against leakage: with a random outcome the model must not beat chance
    df = simulate_cohort(n=300, seed=5)
    df["responder"] = np.random.default_rng(0).permutation(df["responder"].to_numpy())
    res = nested_cv(df, FEATURE_SETS["+ Biomarkers (all)"], logistic_model, seed=0, n_splits=5, n_repeats=1)
    assert 0.35 < res["auc_mean"] < 0.65
