
import pandas as pd
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
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


def graficar_distribucion_etiquetas_pd(df, ruta_salida):
    """
    Grafica el conteo de créditos buenos, malos y sin etiqueta para PD.
    """

    if "default_flag" not in df.columns:
        raise ValueError(
            "El DataFrame debe contener default_flag."
        )

    etiquetas = df["default_flag"]
    desconocidas = set(etiquetas.dropna().unique()) - {0, 1}

    if desconocidas:
        raise ValueError(
            f"Etiquetas PD desconocidas: {desconocidas}"
        )

    nombres = [
        "Buen pagador (0)",
        "Mal pagador (1)",
        "Sin etiqueta"
    ]
    cantidades = [
        int((etiquetas == 0).sum()),
        int((etiquetas == 1).sum()),
        int(etiquetas.isna().sum())
    ]
    colores = ["#3B9B83", "#D06A5F", "#7185A3"]

    figura, eje = plt.subplots(figsize=(9, 4.5))
    barras = eje.barh(
        nombres,
        cantidades,
        color=colores
    )

    eje.invert_yaxis()
    eje.set_title("Distribución de etiquetas para PD")
    eje.set_xlabel("Número de créditos")
    eje.set_ylabel("")
    eje.grid(axis="x", alpha=0.25)
    eje.set_axisbelow(True)
    eje.set_xlim(0, max(cantidades, default=0) * 1.18 or 1)

    for barra, cantidad in zip(barras, cantidades):
        eje.annotate(
            f"{cantidad:,}".replace(",", "."),
            (barra.get_width(), barra.get_y() + barra.get_height() / 2),
            xytext=(6, 0),
            textcoords="offset points",
            va="center"
        )

    figura.tight_layout()

    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close(figura)

    return ruta_salida


def graficar_particiones_temporales_pd(particiones, ruta_salida):
    """
    Grafica el tamaño de las particiones creadas para PD.
    """

    if "split" not in particiones.columns:
        raise ValueError(
            "El DataFrame debe contener la columna split."
        )

    orden = [
        "train",
        "validation",
        "test",
        "supplementary_2015"
    ]
    desconocidas = set(particiones["split"].dropna().unique()) - set(orden)

    if desconocidas or particiones["split"].isna().any():
        raise ValueError(
            f"Particiones desconocidas o vacías: {desconocidas}"
        )

    etiquetas = [
        "Entrenamiento (2007–2012)",
        "Validación (2013)",
        "Prueba (2014)",
        "Complementario (2015)"
    ]
    cantidades = [
        int((particiones["split"] == nombre).sum())
        for nombre in orden
    ]

    figura, eje = plt.subplots(figsize=(9, 4.5))
    barras = eje.barh(
        etiquetas,
        cantidades,
        color=["#176B87", "#45A88A", "#D4A44C", "#8D77AD"]
    )

    eje.invert_yaxis()
    eje.set_title("Tamaño de las particiones temporales de PD")
    eje.set_xlabel("Número de créditos etiquetados")
    eje.set_ylabel("")
    eje.grid(axis="x", alpha=0.25)
    eje.set_axisbelow(True)
    eje.set_xlim(0, max(cantidades, default=0) * 1.18 or 1)

    for barra, cantidad in zip(barras, cantidades):
        eje.annotate(
            f"{cantidad:,}".replace(",", "."),
            (barra.get_width(), barra.get_y() + barra.get_height() / 2),
            xytext=(6, 0),
            textcoords="offset points",
            va="center"
        )

    figura.tight_layout()

    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta_salida, dpi=300, bbox_inches="tight")
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
