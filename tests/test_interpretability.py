import numpy as np
import pytest

from credit_risk.scorecard.interpretability import feature_groups, grouped_permutation


def test_one_hot_grouping_and_determinism(pd_artifacts):
    data = pd_artifacts
    groups = feature_groups(data.bundle.columns)
    assert len(groups["term"]) == 2
    assert len(groups) == 15
    first = grouped_permutation(data.bundle, data.matrix, data.labels, repeats=3, seed=42)
    second = grouped_permutation(data.bundle, data.matrix, data.labels, repeats=3, seed=42)
    assert first == second
    assert first["importances"][0]["feature"] == "loan_amnt"
    assert first["importances"][0]["importance_mean"] > 0.1
    assert first["rows_used"] == len(data.labels)
    assert first["baseline_brier"] >= 0


def test_interpretation_invalid_labels_and_columns(pd_artifacts):
    data = pd_artifacts
    for labels in [np.zeros(len(data.labels)), data.labels[:-1], np.repeat(2, len(data.labels))]:
        with pytest.raises(ValueError):
            grouped_permutation(data.bundle, data.matrix, labels)
    with pytest.raises(ValueError):
        feature_groups(["unknown"])
