"""Construcción de la exposición observada (EAD) y de la LGD realizada.

Definiciones del diccionario LCDataDictionary.xlsx (hoja LoanStats) incluido en
el conjunto de Kaggle:

- ``funded_amnt``: "The total amount committed to that loan at that point in time."
- ``total_rec_prncp``: "Principal received to date".
- ``recoveries``: "post charge off gross recovery".
- ``collection_recovery_fee``: "post charge off collection fee".
- ``out_prncp``: "Remaining outstanding principal for total amount funded".

Fórmula (por crédito castigado ``i``)::

    EAD_i = funded_amnt_i - total_rec_prncp_i
    R_i   = recoveries_i - collection_recovery_fee_i
    LGD_i = clip(1 - R_i / EAD_i, 0, 1)

Solo los créditos castigados (``Charged Off``) tienen un proceso de recuperación
iniciado: su ``out_prncp`` es 0 porque el saldo se reconoció como pérdida y las
``recoveries`` se registran después del castigo. Los estados ``Default`` y
``Late (31-120 days)`` aún conservan saldo vivo y no tienen recuperaciones
observables, por lo que su LGD no está realizada y se excluyen del objetivo.
"""
import numpy as np
import pandas as pd

ESTADOS_CASTIGADOS = (
    "Charged Off",
    "Does not meet the credit policy. Status:Charged Off",
)

COLUMNAS_PERDIDA = [
    "funded_amnt",
    "total_rec_prncp",
    "recoveries",
    "collection_recovery_fee",
]


def calcular_ead_observada(df):
    """Principal pendiente al momento del castigo (EAD observada)."""
    return df["funded_amnt"].astype(float) - df["total_rec_prncp"].astype(float)


def calcular_lgd(df):
    """Devuelve un DataFrame con ``ead_obs``, ``recuperacion_neta``,
    ``lgd_sin_recorte`` y ``lgd`` (recortada a [0, 1]).

    El recorte solo afecta créditos con recuperaciones netas mayores que la
    exposición (LGD negativa), que no son pérdidas sino ganancias contables.
    """
    faltantes = [c for c in COLUMNAS_PERDIDA if c not in df.columns]
    if faltantes:
        raise KeyError(f"Faltan columnas para LGD: {faltantes}")

    datos = df.copy()
    datos["ead_obs"] = calcular_ead_observada(datos)
    datos["recuperacion_neta"] = (
        datos["recoveries"].astype(float)
        - datos["collection_recovery_fee"].astype(float)
    )

    con_exposicion = datos["ead_obs"] > 0
    lgd = pd.Series(np.nan, index=datos.index, dtype=float)
    lgd[con_exposicion] = (
        1.0 - datos.loc[con_exposicion, "recuperacion_neta"]
        / datos.loc[con_exposicion, "ead_obs"]
    )
    datos["lgd_sin_recorte"] = lgd
    datos["lgd"] = lgd.clip(lower=0.0, upper=1.0)
    return datos


def filtrar_castigados(df, estados=ESTADOS_CASTIGADOS):
    """Créditos con pérdida realizada y EAD positiva."""
    estados_limpios = df["loan_status"].astype("string").str.strip()
    datos = df.loc[estados_limpios.isin(list(estados))].copy()
    datos = calcular_lgd(datos)
    return datos.loc[datos["ead_obs"] > 0].copy()
