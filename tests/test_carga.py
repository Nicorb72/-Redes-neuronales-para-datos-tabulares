"""Pruebas de lectura del dataset original."""
import pytest
from credit_risk.data.load_data import cargar_datos, obtener_raiz_proyecto


def test_raiz_del_proyecto():
    raiz = obtener_raiz_proyecto()
    assert (raiz / "configs" / "default.yaml").exists()
    assert (raiz / "data").is_dir()


def test_carga_columnas_reales():
    ruta = obtener_raiz_proyecto() / "data" / "raw" / "loan" / "loan.csv"
    if not ruta.exists():
        pytest.skip("Falta data/raw/loan/loan.csv (no se versiona en Git).")
    df = cargar_datos(columnas=["id", "loan_status", "loan_amnt"])
    assert len(df) == 887379
    assert set(df.columns) == {"id", "loan_status", "loan_amnt"}
    assert df["id"].is_unique
