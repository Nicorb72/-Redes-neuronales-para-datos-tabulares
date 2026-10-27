"""Módulo de datos y particiones para Pérdida Dado el Incumplimiento (LGD).

Garantiza la regla de comparación justa:
- Mismas particiones temporales que PD (Entrenamiento: 2007-2012, Validación: 2013, Prueba: 2014).
- Mismas 15 características predictoras previas al desembolso (sin fuga de información).
- Mismo esquema de preprocesamiento (imputación + estandarización + OneHotEncoder)
  ajustado exclusivamente en el conjunto de entrenamiento de LGD.
"""
import json
import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from credit_risk.data.load_data import cargar_datos, obtener_raiz_proyecto
from credit_risk.data.preprocessing import (
    crear_antiguedad_crediticia,
    preparar_fechas,
    construir_preprocesador,
    VARIABLES_NUMERICAS,
    VARIABLES_CATEGORICAS,
)
from credit_risk.lgd.target import filtrar_castigados, COLUMNAS_PERDIDA

PREDICTORAS = VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS


def cargar_y_preparar_lgd():
    """Carga y construye el dataset completo de préstamos con LGD observada."""
    columnas = list(dict.fromkeys(
        ["id", "loan_status", "issue_d", "earliest_cr_line"]
        + [c for c in VARIABLES_NUMERICAS if c != "antiguedad_crediticia_meses"]
        + VARIABLES_CATEGORICAS
        + COLUMNAS_PERDIDA
    ))

    df = cargar_datos(columnas=columnas)
    df = preparar_fechas(df)
    df = crear_antiguedad_crediticia(df)
    df_cast = filtrar_castigados(df)

    # Asegurar tipos
    for col in VARIABLES_NUMERICAS:
        df_cast[col] = pd.to_numeric(df_cast[col], errors="coerce")
    for col in VARIABLES_CATEGORICAS:
        df_cast[col] = df_cast[col].astype("object")

    # Particiones temporales idénticas a PD
    df_cast["split"] = "sin_asignar"
    df_cast.loc[df_cast["issue_year"].between(2007, 2012), "split"] = "train"
    df_cast.loc[df_cast["issue_year"] == 2013, "split"] = "validation"
    df_cast.loc[df_cast["issue_year"] == 2014, "split"] = "test"
    df_cast.loc[df_cast["issue_year"] == 2015, "split"] = "supplementary_2015"

    return df_cast


def preparar_matrices_lgd(df_cast=None):
    """Genera matrices dispersas X, vector y (LGD) y vector ead_obs por partición.

    El preprocesador se ajusta únicamente con los registros de entrenamiento.
    """
    if df_cast is None:
        df_cast = cargar_y_preparar_lgd()

    preprocesador = construir_preprocesador()

    conjuntos = ["train", "validation", "test", "supplementary_2015"]
    X_raw = {}
    y = {}
    ead = {}

    for split in conjuntos:
        subset = df_cast[df_cast["split"] == split]
        X_raw[split] = subset[PREDICTORAS].copy()
        y[split] = subset["lgd"].to_numpy(dtype=np.float64)
        ead[split] = subset["ead_obs"].to_numpy(dtype=np.float64)

    # Ajustar preprocesador solo en train
    X_trans = {}
    X_trans["train"] = preprocesador.fit_transform(X_raw["train"])

    for split in conjuntos[1:]:
        X_trans[split] = preprocesador.transform(X_raw[split])

    return X_trans, y, ead, preprocesador


def guardar_matrices_lgd(X_trans, y, ead, preprocesador):
    """Persiste matrices procesadas y metadatos en data/processed y artifacts/models."""
    raiz = obtener_raiz_proyecto()
    carpeta_datos = raiz / "data" / "processed"
    carpeta_modelos = raiz / "artifacts" / "models"
    carpeta_datos.mkdir(parents=True, exist_ok=True)
    carpeta_modelos.mkdir(parents=True, exist_ok=True)

    for split in X_trans.keys():
        sparse.save_npz(carpeta_datos / f"X_{split}_lgd.npz", sparse.csr_matrix(X_trans[split]))
        np.save(carpeta_datos / f"y_{split}_lgd.npy", y[split])
        np.save(carpeta_datos / f"ead_{split}_lgd.npy", ead[split])

    joblib.dump(preprocesador, carpeta_modelos / "preprocesador_lgd.joblib")

    columnas = list(preprocesador.get_feature_names_out())
    with open(carpeta_modelos / "columnas_lgd.json", "w", encoding="utf-8") as f:
        json.dump(columnas, f, indent=4, ensure_ascii=False)


def main():
    print("=" * 60)
    print("PREPARACIÓN DE DATOS PARA LGD (Pérdida Dado el Incumplimiento)")
    print("=" * 60)
    df_cast = cargar_y_preparar_lgd()
    print(f"Total créditos castigados válidos: {len(df_cast):,}")
    for split in ["train", "validation", "test", "supplementary_2015"]:
        n = (df_cast["split"] == split).sum()
        print(f"  Split '{split}': {n:,} registros")

    X_trans, y, ead, preprocesador = preparar_matrices_lgd(df_cast)
    guardar_matrices_lgd(X_trans, y, ead, preprocesador)
    print("\nArtefactos de LGD guardados exitosamente.")


if __name__ == "__main__":
    main()
