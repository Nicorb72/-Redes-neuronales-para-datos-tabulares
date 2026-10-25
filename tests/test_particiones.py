"""Pruebas de fechas y particiones cronológicas."""
import pandas as pd
from credit_risk.data.preprocessing import preparar_fechas
from credit_risk.data.prepare_data import asignar_particiones


def test_fechas_y_anios():
    df = pd.DataFrame({"issue_d": ["Jun-2007", "Dec-2015"]})
    res = preparar_fechas(df)
    assert res["issue_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2007-06-01", "2015-12-01"
    ]
    assert res["issue_year"].tolist() == [2007, 2015]


def test_particiones_sin_traslapes():
    df = pd.DataFrame({
        "issue_year": [2007, 2012, 2013, 2014, 2015, 2015],
        "default_flag": pd.Series([0, 1, 0, 1, 0, pd.NA], dtype="Int64"),
        "id": [1, 2, 3, 4, 5, 6],
    })
    resultado = asignar_particiones(df)
    assert resultado["split"].tolist() == [
        "train", "train", "validation", "test", "supplementary_2015"
    ]
    assert resultado["id"].tolist() == [1, 2, 3, 4, 5]
    assert set(resultado["default_flag"]) == {0, 1}
