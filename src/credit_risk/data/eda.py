
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def resumen_estados_crediticios(df):
    """
    Calcula la frecuencia y el porcentaje
    de cada estado crediticio del dataset.
    """

    if "loan_status" not in df.columns:
        raise ValueError(
            "El dataset debe contener loan_status."
        )

    resumen = (
        df["loan_status"]
        .value_counts(dropna=False)
        .rename_axis("estado_crediticio")
        .reset_index(name="cantidad")
    )

    resumen["porcentaje"] = (
        resumen["cantidad"] / len(df) * 100
    ).round(2)

    return resumen


def resumen_temporal_creditos(df):
    """
    Resume los créditos por año de originación.

    Requiere:
        issue_year
        default_flag

    Incluye todos los créditos, incluso aquellos
    que todavía no tienen etiqueta definitiva.
    """

    columnas = {"issue_year", "default_flag"}

    if not columnas.issubset(df.columns):
        raise ValueError(
            f"Faltan columnas: {columnas - set(df.columns)}"
        )

    datos = df.copy()

    datos["clasificacion"] = "Sin etiqueta"

    datos.loc[
        datos["default_flag"] == 0,
        "clasificacion"
    ] = "Bueno"

    datos.loc[
        datos["default_flag"] == 1,
        "clasificacion"
    ] = "Malo"

    resumen = pd.crosstab(
        datos["issue_year"],
        datos["clasificacion"]
    )

    resumen = resumen.reindex(
        columns=["Bueno", "Malo", "Sin etiqueta"],
        fill_value=0
    )

    resumen["Total"] = resumen.sum(axis=1)

    resumen["Porcentaje sin etiqueta"] = (
        resumen["Sin etiqueta"] / resumen["Total"] * 100
    ).round(2)

    resumen["Tasa de incumplimiento etiquetados"] = (
        resumen["Malo"]
        / (resumen["Bueno"] + resumen["Malo"])
        * 100
    ).round(2)

    return resumen.reset_index()


def graficar_creditos_sin_etiqueta(resumen, ruta_salida):
    """
    Genera una gráfica del porcentaje de
    créditos sin etiqueta por año.
    """

    figura, eje = plt.subplots(figsize=(10, 5))

    eje.plot(
        resumen["issue_year"],
        resumen["Porcentaje sin etiqueta"],
        marker="o",
        linewidth=2,
        color="#7552A3"
    )

    eje.set_title(
        "Créditos sin etiqueta por año de originación"
    )

    eje.set_xlabel("Año de originación")
    eje.set_ylabel("Créditos sin etiqueta (%)")

    eje.set_ylim(0, 100)
    eje.set_xticks(resumen["issue_year"])

    eje.grid(alpha=0.25)

    figura.tight_layout()

    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    figura.savefig(
        ruta_salida,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figura)

    return ruta_salida

def resumen_variables_financieras(df):
    """
    Calcula estadísticas descriptivas de las variables
    financieras según el resultado crediticio.

    Incluye únicamente créditos con etiqueta válida.
    """

    variables = [
        "annual_inc",
        "dti",
        "loan_amnt"
    ]

    datos = df.dropna(
        subset=["default_flag"]
    ).copy()

    datos["clasificacion"] = datos["default_flag"].map({
        0: "Buen pagador",
        1: "Mal pagador"
    })

    resultados = []

    for variable in variables:

        datos[variable] = pd.to_numeric(
            datos[variable],
            errors="coerce"
        )

        resumen = datos.groupby("clasificacion")[
            variable
        ].agg(
            cantidad="count",
            media="mean",
            mediana="median",
            desviacion="std",
            minimo="min",
            maximo="max"
        ).reset_index()

        resumen.insert(0, "variable", variable)

        resultados.append(resumen)

    resultado = pd.concat(
        resultados,
        ignore_index=True
    )

    return resultado.round(2)

def resumen_calidad_datos(df):
    """
    Genera una auditoría de valores faltantes
    y características básicas de cada variable.
    """

    resumen = pd.DataFrame({
        "variable": df.columns,
        "tipo": df.dtypes.astype(str).values,
        "faltantes": df.isna().sum().values,
        "porcentaje_faltantes": (
            df.isna().mean().values * 100
        ).round(2),
        "valores_unicos": df.nunique(dropna=True).values
    })

    return resumen.sort_values(
        "porcentaje_faltantes",
        ascending=False
    ).reset_index(drop=True)

def resumen_valores_extremos(df):
    """
    Analiza la distribución y los valores extremos
    de las variables financieras seleccionadas.
    """

    variables = [
        "annual_inc",
        "dti",
        "loan_amnt",
        "revol_bal",
        "revol_util"
    ]

    resultados = []

    for variable in variables:

        valores = pd.to_numeric(
            df[variable],
            errors="coerce"
        ).dropna()

        p99 = valores.quantile(0.99)

        resultados.append({
            "variable": variable,
            "cantidad": len(valores),
            "minimo": valores.min(),
            "mediana": valores.median(),
            "p95": valores.quantile(0.95),
            "p99": p99,
            "maximo": valores.max(),
            "superiores_p99": int(
                (valores > p99).sum()
            )
        })

    return pd.DataFrame(resultados).round(2)

def auditar_utilizacion_credito(df):
    """
    Analiza los valores extremos de revol_util
    sin modificar ni eliminar registros.
    """

    valores = pd.to_numeric(
        df["revol_util"],
        errors="coerce"
    )

    resumen = {
        "total": len(df),
        "faltantes": int(valores.isna().sum()),
        "mayores_100": int((valores > 100).sum()),
        "mayores_150": int((valores > 150).sum()),
        "mayores_200": int((valores > 200).sum()),
        "negativos": int((valores < 0).sum()),
        "maximo": valores.max()
    }

    return resumen

