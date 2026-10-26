"""Pruebas de las figuras exploratorias usadas por el reporte."""
import pandas as pd
import pytest

from credit_risk.data.eda import (
    graficar_distribucion_etiquetas_pd,
    graficar_particiones_temporales_pd,
)
from credit_risk.data.prepare_data import asignar_particiones


def test_graficar_distribucion_etiquetas_pd(tmp_path):
    datos = pd.DataFrame({
        "default_flag": pd.Series([0, 0, 1, pd.NA], dtype="Int64")
    })

    ruta = graficar_distribucion_etiquetas_pd(
        datos,
        tmp_path / "etiquetas.svg"
    )

    assert ruta.exists()
    assert ruta.stat().st_size > 0


def test_graficar_particiones_temporales_pd(tmp_path):
    datos = pd.DataFrame({
        "issue_year": [2007, 2013, 2014, 2015],
        "default_flag": [0, 1, 0, 1]
    })
    particiones = asignar_particiones(datos)

    ruta = graficar_particiones_temporales_pd(
        particiones,
        tmp_path / "particiones.svg"
    )

    assert ruta.exists()
    assert ruta.stat().st_size > 0


def test_graficar_etiquetas_requiere_objetivo(tmp_path):
    with pytest.raises(ValueError, match="default_flag"):
        graficar_distribucion_etiquetas_pd(
            pd.DataFrame({"loan_status": ["Fully Paid"]}),
            tmp_path / "etiquetas.svg"
        )
