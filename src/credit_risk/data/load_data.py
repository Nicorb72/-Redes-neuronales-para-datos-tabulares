
from pathlib import Path
import pandas as pd


def obtener_raiz_proyecto():
    """
    Encuentra la carpeta principal del proyecto.
    """

    for carpeta in Path(__file__).resolve().parents:
        if (
            (carpeta / "configs" / "default.yaml").exists()
            and (carpeta / "data").exists()
        ):
            return carpeta

    raise FileNotFoundError(
        "No se encontró la raíz del proyecto."
    )


def cargar_datos(columnas=None):
    """
    Carga el dataset original de Lending Club.

    Parámetros
    ----------
    columnas : list o None
        Columnas que se desean cargar.
        Si es None, carga todas las variables.

    Retorna
    -------
    pandas.DataFrame
        Datos originales sin modificaciones.
    """

    raiz = obtener_raiz_proyecto()

    ruta = raiz / "data" / "raw" / "loan" / "loan.csv"

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {ruta}"
        )

    df = pd.read_csv(
        ruta,
        usecols=columnas,
        low_memory=False,
        index_col=False
    )

    return df
