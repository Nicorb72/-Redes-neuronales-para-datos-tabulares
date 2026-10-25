
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BUENOS = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid"
}

MALOS = {
    "Charged Off",
    "Default",
    "Late (31-120 days)",
    "Does not meet the credit policy. Status:Charged Off"
}

INCIERTOS = {
    "Current",
    "Issued",
    "In Grace Period",
    "Late (16-30 days)"
}


def crear_variable_objetivo(df):
    """
    Construye default_flag a partir de loan_status.

    0: Buen pagador
    1: Mal pagador
    NA: Estado incierto
    """

    datos = df.copy()

    estados = datos["loan_status"].astype("string").str.strip()

    # Identificar categorías no contempladas
    conocidos = BUENOS | MALOS | INCIERTOS

    desconocidos = set(estados.dropna().unique()) - conocidos

    if desconocidos:
        raise ValueError(
            f"Estados crediticios desconocidos: {desconocidos}"
        )

    # Crear variable objetivo
    datos["default_flag"] = pd.Series(
        pd.NA,
        index=datos.index,
        dtype="Int64"
    )

    datos.loc[
        estados.isin(BUENOS), "default_flag"
    ] = 0

    datos.loc[
        estados.isin(MALOS), "default_flag"
    ] = 1

    return datos


def preparar_fechas(df):
    """
    Convierte issue_d en una fecha y extrae
    el año de originación del préstamo.
    """

    datos = df.copy()

    datos["issue_date"] = pd.to_datetime(
        datos["issue_d"].astype("string").str.strip(),
        format="%b-%Y",
        errors="coerce"
    )

    datos["issue_year"] = datos["issue_date"].dt.year

    # Verificar que las fechas sean válidas
    if datos["issue_date"].isna().any():
        raise ValueError(
            "Se encontraron fechas de originación inválidas."
        )

    return datos


def crear_antiguedad_crediticia(df):
    """
    Calcula los meses transcurridos entre el inicio
    del historial crediticio y la originación del préstamo.

    Requiere:
        earliest_cr_line
        issue_d

    Genera:
        antiguedad_crediticia_meses
    """

    datos = df.copy()

    # Convertir fecha inicial del historial
    fecha_historial = pd.to_datetime(
        datos["earliest_cr_line"].astype("string").str.strip(),
        format="%b-%Y",
        errors="coerce"
    )

    # Convertir fecha de originación
    fecha_prestamo = pd.to_datetime(
        datos["issue_d"].astype("string").str.strip(),
        format="%b-%Y",
        errors="coerce"
    )

    # Calcular diferencia en meses
    datos["antiguedad_crediticia_meses"] = (
        (fecha_prestamo.dt.year - fecha_historial.dt.year) * 12
        + fecha_prestamo.dt.month
        - fecha_historial.dt.month
    )

    # Antigüedades negativas son inconsistentes
    datos.loc[
        datos["antiguedad_crediticia_meses"] < 0,
        "antiguedad_crediticia_meses"
    ] = float("nan")

    return datos

# Variables numéricas del modelo PD
VARIABLES_NUMERICAS = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "antiguedad_crediticia_meses"
]

# Variables categóricas del modelo PD
VARIABLES_CATEGORICAS = [
    "term",
    "purpose",
    "emp_length",
    "home_ownership"
]


def construir_preprocesador():
    """
    Construye el transformador de características PD.

    Las estadísticas de imputación y escalamiento
    se aprenderán cuando se ejecute fit() utilizando
    exclusivamente el conjunto de entrenamiento.
    """

    # Transformaciones numéricas
    transformador_numerico = Pipeline([
        (
            "imputacion",
            SimpleImputer(strategy="median")
        ),
        (
            "escalamiento",
            StandardScaler()
        )
    ])

    # Transformaciones categóricas
    transformador_categorico = Pipeline([
        (
            "imputacion",
            SimpleImputer(
                strategy="constant",
                fill_value="Desconocido"
            )
        ),
        (
            "codificacion",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    preprocesador = ColumnTransformer(
        transformers=[
            (
                "numericas",
                transformador_numerico,
                VARIABLES_NUMERICAS
            ),
            (
                "categoricas",
                transformador_categorico,
                VARIABLES_CATEGORICAS
            )
        ],
        sparse_threshold=1.0
    )

    return preprocesador
