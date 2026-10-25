"""Pruebas del etiquetado de PD."""
import pandas as pd
import pytest
from credit_risk.data.preprocessing import crear_variable_objetivo


def test_clasificacion_de_estados():
    estados = [
        "Fully Paid",
        "Does not meet the credit policy. Status:Fully Paid",
        "Charged Off",
        "Default",
        "Late (31-120 days)",
        "Does not meet the credit policy. Status:Charged Off",
        "Current",
        "Issued",
        "In Grace Period",
        "Late (16-30 days)",
    ]
    resultado = crear_variable_objetivo(pd.DataFrame({"loan_status": estados}))
    assert resultado["default_flag"].tolist()[:6] == [0, 0, 1, 1, 1, 1]
    assert resultado["default_flag"].iloc[6:].isna().all()
    assert resultado["loan_status"].tolist() == estados


def test_estado_desconocido_falla():
    with pytest.raises(ValueError, match="desconocidos"):
        crear_variable_objetivo(pd.DataFrame({"loan_status": ["ESTADO_INEXISTENTE"]}))
