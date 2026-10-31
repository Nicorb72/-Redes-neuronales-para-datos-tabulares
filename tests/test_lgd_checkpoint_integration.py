"""Comprueba que el checkpoint entregado corresponde a la selección por validación."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from credit_risk.lgd.neural_network import LGDNeuralNetwork

ROOT = Path(__file__).resolve().parents[1]


def test_saved_lgd_checkpoint_matches_validation_selection():
    document = json.loads((ROOT / 'artifacts/models/lgd_benchmark_results.json').read_text(encoding='utf-8'))
    stats = document['mlp_30_semillas']
    records = stats['per_seed_metrics']
    selection = stats['checkpoint_selection']
    assert stats['n_semillas'] == len(records)
    assert len(records) >= 30
    assert len({row['seed'] for row in records}) == len(records)
    expected = min(records, key=lambda row: row['validation_rmse'])
    assert selection['split'] == 'validation'
    assert selection['criterion'] == 'rmse'
    assert selection['test_used_for_selection'] is False
    assert selection['seed'] == expected['seed']
    assert [row['seed'] for row in records if row['selected']] == [expected['seed']]
    assert document['mejor_mlp_individual']['seed'] == expected['seed']

    for metric in ('mae', 'rmse'):
        values = [row[f'test_{metric}'] for row in records]
        assert stats[metric]['media'] == pytest.approx(np.mean(values))
        assert stats[metric]['std'] == pytest.approx(np.std(values))
        assert stats[metric]['min'] == pytest.approx(min(values))
        assert stats[metric]['max'] == pytest.approx(max(values))

    model = LGDNeuralNetwork.load(ROOT / 'artifacts/models/mlp_lgd.pt', device='cpu')
    assert model.seed == expected['seed']
    for split, prefix in [('validation', 'validation'), ('test', 'test')]:
        X = sparse.load_npz(ROOT / f'data/processed/X_{split}_lgd.npz')
        y = np.load(ROOT / f'data/processed/y_{split}_lgd.npy')
        mae, rmse = model.evaluate(X, y)
        assert mae == pytest.approx(expected[f'{prefix}_mae'], abs=2e-6)
        assert rmse == pytest.approx(expected[f'{prefix}_rmse'], abs=2e-6)
