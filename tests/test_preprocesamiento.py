"""Pruebas de ingeniería de variables, ajuste y guardado de matrices."""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from credit_risk.data.preprocessing import (
    VARIABLES_NUMERICAS,
    VARIABLES_CATEGORICAS,
    crear_antiguedad_crediticia,
    construir_preprocesador,
)
from credit_risk.data.load_data import obtener_raiz_proyecto


def _ejemplo():
    return pd.DataFrame({
        "loan_amnt": [5000, 10000, 15000],
        "annual_inc": [40000, 60000, np.nan],
        "dti": [10, 20, 30],
        "delinq_2yrs": [0, 1, 0],
        "inq_last_6mths": [1, 0, 2],
        "open_acc": [5, 8, 10],
        "pub_rec": [0, 0, 1],
        "revol_bal": [2000, 5000, 8000],
        "revol_util": [20, 40, 60],
        "total_acc": [10, 15, 20],
        "antiguedad_crediticia_meses": [60, 120, 180],
        "term": ["36 months", "60 months", "36 months"],
        "purpose": ["car", "debt_consolidation", "car"],
        "emp_length": ["2 years", None, "5 years"],
        "home_ownership": ["RENT", "OWN", "RENT"],
    })


def test_antiguedad_crediticia():
    df = pd.DataFrame({
        "earliest_cr_line": ["Jan-2010", "Dec-2015", None],
        "issue_d": ["Jan-2015", "Jan-2015", "Feb-2014"],
    })
    res = crear_antiguedad_crediticia(df)["antiguedad_crediticia_meses"]
    assert res.iloc[0] == 60
    assert pd.isna(res.iloc[1])
    assert pd.isna(res.iloc[2])


def test_transformador_solo_ajusta_con_train():
    train = _ejemplo()
    validacion = train.iloc[[0]].copy()
    validacion["annual_inc"] = 90000
    modelo = construir_preprocesador()
    X_train = modelo.fit_transform(train)
    X_val = modelo.transform(validacion)
    assert X_train.shape[0] == 3
    assert X_val.shape == (1, X_train.shape[1])
    assert len(modelo.get_feature_names_out()) == X_train.shape[1]
    mediana = modelo.named_transformers_["numericas"].named_steps["imputacion"].statistics_[1]
    assert mediana == 50000
    assert np.isfinite(X_train.data if sparse.issparse(X_train) else X_train).all()
    assert np.isfinite(X_val.data if sparse.issparse(X_val) else X_val).all()
    assert len(VARIABLES_NUMERICAS) + len(VARIABLES_CATEGORICAS) == 15


def test_matrices_ya_guardadas_si_existen():
    base = obtener_raiz_proyecto() / "data" / "processed"
    ruta = base / "X_train_pd.npz"
    if not ruta.exists():
        pytest.skip("Matrices opcionales no presentes: ejecutar preparación de PD.")
    matriz = sparse.load_npz(ruta)
    y = np.load(base / "y_train_pd.npy")
    assert matriz.shape == (90347, 44)
    assert y.shape == (90347,)
    assert np.isfinite(matriz.data).all()
    assert set(np.unique(y)) == {0, 1}
