
import pandas as pd
from credit_risk.data.load_data import cargar_datos

from credit_risk.data.preprocessing import (
    crear_variable_objetivo,
    preparar_fechas,
    crear_antiguedad_crediticia,
    construir_preprocesador,
    VARIABLES_NUMERICAS,
    VARIABLES_CATEGORICAS
)
import json
import joblib
import numpy as np
from scipy import sparse

def asignar_particiones(df):
    """
    Asigna los créditos etiquetados a conjuntos
    según su año de originación.

    Requiere:
        issue_year
        default_flag

    Retorna:
        DataFrame con columna split.
    """

    datos = df.copy()

    # Conservar únicamente créditos con etiqueta válida
    datos = datos.dropna(
        subset=["default_flag"]
    ).copy()

    datos["default_flag"] = (
        datos["default_flag"].astype(int)
    )

    # Inicializar la partición
    datos["split"] = "sin_asignar"

    # Entrenamiento: 2007-2012
    datos.loc[
        datos["issue_year"].between(2007, 2012),
        "split"
    ] = "train"

    # Validación: 2013
    datos.loc[
        datos["issue_year"] == 2013,
        "split"
    ] = "validation"

    # Prueba: 2014
    datos.loc[
        datos["issue_year"] == 2014,
        "split"
    ] = "test"

    # Análisis complementario: 2015
    datos.loc[
        datos["issue_year"] == 2015,
        "split"
    ] = "supplementary_2015"

    # Verificar asignaciones
    if (datos["split"] == "sin_asignar").any():
        raise ValueError(
            "Existen créditos sin partición temporal."
        )

    return datos



def preparar_matrices_pd():
    """
    Ejecuta la preparación de datos para PD.

    Ajusta el preprocesador solo con entrenamiento
    y transforma los demás conjuntos.

    En esta etapa no guarda archivos.
    """

    print("1. Cargando dataset...")

    columnas = [
        "id", "loan_status", "issue_d",
        "earliest_cr_line"
    ]

    # Añadir las características originales
    columnas += [
        c for c in VARIABLES_NUMERICAS
        if c != "antiguedad_crediticia_meses"
    ]

    columnas += VARIABLES_CATEGORICAS

    # Evitar columnas duplicadas
    columnas = list(dict.fromkeys(columnas))

    df = cargar_datos(columnas=columnas)

    print("2. Construyendo variable objetivo...")
    df = crear_variable_objetivo(df)

    print("3. Preparando fechas...")
    df = preparar_fechas(df)

    print("4. Asignando particiones...")
    df = asignar_particiones(df)

    print("5. Calculando antigüedad crediticia...")
    df = crear_antiguedad_crediticia(df)

    # Asegurar tipos numéricos
    for variable in VARIABLES_NUMERICAS:
        df[variable] = pd.to_numeric(
            df[variable],
            errors="coerce"
        )

    # Asegurar tipos compatibles con imputación
    for variable in VARIABLES_CATEGORICAS:
        df[variable] = df[variable].astype("object")

    predictoras = (
        VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS
    )

    conjuntos = [
        "train",
        "validation",
        "test",
        "supplementary_2015"
    ]

    X = {}
    y = {}

    for nombre in conjuntos:
        parte = df.loc[df["split"] == nombre]

        X[nombre] = parte[predictoras].copy()

        y[nombre] = parte[
            "default_flag"
        ].to_numpy(dtype="int64")

    print("6. Ajustando preprocesador...")

    preprocesador = construir_preprocesador()

    X_transformadas = {}

    # Aprender parámetros únicamente de entrenamiento
    X_transformadas["train"] = (
        preprocesador.fit_transform(X["train"])
    )

    # Aplicar sin reajustar a los otros conjuntos
    for nombre in conjuntos[1:]:
        X_transformadas[nombre] = (
            preprocesador.transform(X[nombre])
        )

    print("\nRESULTADOS DEL PREPROCESAMIENTO")

    for nombre in conjuntos:
        print(
            f"{nombre}: "
            f"{X_transformadas[nombre].shape[0]} registros, "
            f"{X_transformadas[nombre].shape[1]} características"
        )

    return X_transformadas, y, preprocesador

def guardar_matrices_pd(X, y, preprocesador):
    """
    Guarda las matrices, etiquetas y el
    preprocesador de la tarea PD.
    """

    from credit_risk.data.load_data import obtener_raiz_proyecto

    raiz = obtener_raiz_proyecto()

    carpeta_datos = raiz / "data" / "processed"
    carpeta_modelos = raiz / "artifacts" / "models"

    carpeta_datos.mkdir(parents=True, exist_ok=True)
    carpeta_modelos.mkdir(parents=True, exist_ok=True)

    # Guardar matrices y etiquetas
    for nombre in X.keys():

        matriz = sparse.csr_matrix(X[nombre])

        sparse.save_npz(
            carpeta_datos / f"X_{nombre}_pd.npz",
            matriz
        )

        np.save(
            carpeta_datos / f"y_{nombre}_pd.npy",
            y[nombre]
        )

        print(f"Archivos guardados: {nombre}")

    # Guardar preprocesador ajustado
    joblib.dump(
        preprocesador,
        carpeta_modelos / "preprocesador_pd.joblib"
    )

    # Guardar nombres de características
    nombres = list(
        preprocesador.get_feature_names_out()
    )

    with open(
        carpeta_modelos / "columnas_pd.json",
        "w",
        encoding="utf-8"
    ) as archivo:
        json.dump(
            nombres,
            archivo,
            indent=4,
            ensure_ascii=False
        )

    print("\nTodos los archivos fueron guardados.")

def main():
    """
    Ejecuta el procesamiento completo de PD
    y guarda los archivos para entrenamiento.
    """

    print("=" * 55)
    print("PREPARACIÓN DE DATOS PARA PD")
    print("=" * 55)

    X, y, preprocesador = preparar_matrices_pd()

    guardar_matrices_pd(
        X,
        y,
        preprocesador
    )

    print("\nPROCESAMIENTO PD FINALIZADO")


if __name__ == "__main__":
    main()
