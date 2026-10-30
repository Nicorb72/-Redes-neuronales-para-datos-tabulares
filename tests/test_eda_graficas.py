"""Pruebas de las figuras exploratorias usadas por el reporte."""
import json

import pandas as pd
import pytest

from credit_risk.data.eda import (
    graficar_distribucion_etiquetas_pd,
    graficar_particiones_temporales_pd,
)
from credit_risk.data.generate_eda_figures import generar_desde_resumen
from credit_risk.data.prepare_data import asignar_particiones


def test_generar_figuras_desde_resumen_sin_csv(tmp_path):
    raiz = tmp_path
    ruta_resumen = raiz / "data" / "processed" / "eda_pd_resumen.json"
    ruta_resumen.parent.mkdir(parents=True)
    ruta_resumen.write_text(json.dumps({
        "total_creditos": 10,
        "etiquetas": {"buenos": 4, "malos": 2, "sin_etiqueta": 4},
        "particiones": {
            "train": 2,
            "validation": 1,
            "test": 2,
            "supplementary_2015": 1,
        },
    }), encoding="utf-8")
    figuras = raiz / "reports" / "figures"
    figuras.mkdir(parents=True)

    generar_desde_resumen(raiz, figuras)

    assert (figuras / "distribucion_etiquetas_pd.svg").stat().st_size > 0
    assert (figuras / "particiones_temporales_pd.svg").stat().st_size > 0


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
