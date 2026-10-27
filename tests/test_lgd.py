"""Pruebas unitarias para el módulo de Pérdida Dado el Incumplimiento (LGD)."""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from credit_risk.lgd.baseline import BaselineLGD
from credit_risk.lgd.evaluation import calcular_metricas_lgd, resumen_distribucion
from credit_risk.lgd.expected_loss import calcular_perdida_esperada, resumir_perdida_esperada
from credit_risk.lgd.neural_network import LGDNeuralNetwork
from credit_risk.lgd.target import (
    calcular_ead_observada,
    calcular_lgd,
    filtrar_castigados,
)


def test_calcular_ead_observada():
    df = pd.DataFrame({
        "funded_amnt": [10000.0, 5000.0, 20000.0],
        "total_rec_prncp": [2000.0, 5000.0, 0.0],
    })
    ead = calcular_ead_observada(df)
    assert np.allclose(ead, [8000.0, 0.0, 20000.0])


def test_calcular_lgd_formulas_y_recortes():
    # Caso 1: EAD = 10000, recupera 3000 con comisión 500 -> neto 2500 -> LGD = 1 - 2500/10000 = 0.75
    # Caso 2: EAD = 10000, recupera 12000 con comisión 1000 -> neto 11000 -> LGD < 0 -> clip a 0.0
    # Caso 3: EAD = 10000, recupera 0 -> LGD = 1.0
    df = pd.DataFrame({
        "funded_amnt": [10000.0, 10000.0, 10000.0],
        "total_rec_prncp": [0.0, 0.0, 0.0],
        "recoveries": [3000.0, 12000.0, 0.0],
        "collection_recovery_fee": [500.0, 1000.0, 0.0],
    })
    res = calcular_lgd(df)
    assert np.allclose(res["lgd"], [0.75, 0.0, 1.0])
    assert res["lgd_sin_recorte"].iloc[1] < 0.0


def test_filtrar_castigados_excluye_otros_estados():
    df = pd.DataFrame({
        "loan_status": [
            "Charged Off",
            "Fully Paid",
            "Current",
            "Default",
            "Does not meet the credit policy. Status:Charged Off",
            "Late (31-120 days)",
        ],
        "funded_amnt": [10000.0] * 6,
        "total_rec_prncp": [2000.0] * 6,
        "recoveries": [500.0] * 6,
        "collection_recovery_fee": [50.0] * 6,
    })
    filtrados = filtrar_castigados(df)
    # Solo conserva los 2 estados castigados
    assert len(filtrados) == 2
    assert set(filtrados["loan_status"]) == {
        "Charged Off",
        "Does not meet the credit policy. Status:Charged Off",
    }


def test_baseline_lgd_entrenamiento_y_acotamiento():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 10))
    y = np.clip(rng.uniform(0.6, 1.0, size=100), 0.0, 1.0)

    model = BaselineLGD(alpha=1.0)
    assert model.n_parameters == 0
    model.fit(X, y)
    assert model.n_parameters == 11  # 10 pesos + 1 intercepto

    preds = model.predict(X)
    assert len(preds) == 100
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)

    mae, rmse = model.evaluate(X, y)
    assert 0.0 <= mae <= 1.0
    assert 0.0 <= rmse <= 1.0


def test_mlp_regressor_salida_sigmoide_acotada():
    rng = np.random.default_rng(123)
    X = sparse.csr_matrix(rng.normal(size=(80, 15)))
    y = rng.uniform(0.5, 1.0, size=80)

    nn_model = LGDNeuralNetwork(
        hidden_dims=[16, 8],
        dropout=0.1,
        epochs=10,
        batch_size=16,
        seed=123,
    )
    nn_model.fit(X, y)
    assert nn_model.n_parameters > 0

    preds = nn_model.predict(X)
    assert preds.shape == (80,)
    # Salida por construcción Sigmoide debe estar en [0, 1]
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)


def test_calculo_y_resumen_perdida_esperada():
    pd_probs = np.array([0.10, 0.50, 0.90])
    lgd_preds = np.array([0.80, 0.60, 1.00])
    ead_amounts = np.array([10000.0, 20000.0, 5000.0])

    # EL = PD * LGD * EAD
    # 0.10 * 0.80 * 10000 = 800
    # 0.50 * 0.60 * 20000 = 6000
    # 0.90 * 1.00 * 5000  = 4500
    el = calcular_perdida_esperada(pd_probs, lgd_preds, ead_amounts)
    assert np.allclose(el, [800.0, 6000.0, 4500.0])

    res = resumir_perdida_esperada(el, ead_amounts)
    assert res["exposicion_total_usd"] == 35000.0
    assert res["perdida_esperada_total_usd"] == 11300.0
    assert np.isclose(res["tasa_perdida_esperada_pct"], (11300.0 / 35000.0) * 100.0)


def test_metricas_distribucion_y_errores():
    y_true = np.array([0.5, 0.8, 1.0, 0.9, 0.7])
    y_pred = np.array([0.55, 0.75, 0.95, 0.85, 0.72])

    metrics = calcular_metricas_lgd(y_true, y_pred)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "ks_statistic" in metrics
    assert metrics["mae"] > 0

    dist = resumen_distribucion(y_true, "test")
    assert dist["media"] == pytest.approx(float(np.mean(y_true)))
    assert dist["min"] == 0.5
    assert dist["max"] == 1.0
