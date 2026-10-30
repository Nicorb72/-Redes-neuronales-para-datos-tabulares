
"""Genera las figuras exploratorias de PD."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from credit_risk.data.load_data import (
    cargar_datos,
    obtener_raiz_proyecto
)
from credit_risk.data.prepare_data import asignar_particiones
from credit_risk.data.preprocessing import (
    crear_variable_objetivo,
    preparar_fechas
)
from credit_risk.data.eda import (
    graficar_distribucion_etiquetas_pd,
    graficar_particiones_temporales_pd
)


def graficar_resumen(nombres, cantidades, colores, titulo,
                    etiqueta_x, ruta):
    figura, eje = plt.subplots(figsize=(9, 4.5))

    barras = eje.barh(
        nombres,
        cantidades,
        color=colores
    )

    eje.invert_yaxis()
    eje.set_title(titulo)
    eje.set_xlabel(etiqueta_x)
    eje.grid(axis="x", alpha=0.25)
    eje.set_axisbelow(True)
    eje.set_xlim(0, max(cantidades) * 1.18)

    for barra, cantidad in zip(barras, cantidades):
        eje.annotate(
            f"{cantidad:,}".replace(",", "."),
            (barra.get_width(),
             barra.get_y() + barra.get_height() / 2),
            xytext=(6, 0),
            textcoords="offset points",
            va="center"
        )

    figura.tight_layout()
    figura.savefig(ruta, bbox_inches="tight")
    plt.close(figura)


def generar_desde_resumen(raiz, figuras):
    ruta = raiz / "data" / "processed" / "eda_pd_resumen.json"

    with open(ruta, encoding="utf-8") as archivo:
        resumen = json.load(archivo)

    etiquetas = resumen["etiquetas"]
    particiones = resumen["particiones"]

    assert (
        sum(etiquetas.values()) == resumen["total_creditos"]
    )

    assert (
        sum(particiones.values())
        == etiquetas["buenos"] + etiquetas["malos"]
    )

    graficar_resumen(
        ["Buen pagador (0)", "Mal pagador (1)", "Sin etiqueta"],
        [
            etiquetas["buenos"],
            etiquetas["malos"],
            etiquetas["sin_etiqueta"]
        ],
        ["#3B9B83", "#D06A5F", "#7185A3"],
        "Distribución de etiquetas para PD",
        "Número de créditos",
        figuras / "distribucion_etiquetas_pd.svg"
    )

    graficar_resumen(
        [
            "Entrenamiento (2007–2012)",
            "Validación (2013)",
            "Prueba (2014)",
            "Complementario (2015)"
        ],
        [
            particiones["train"],
            particiones["validation"],
            particiones["test"],
            particiones["supplementary_2015"]
        ],
        ["#176B87", "#45A88A", "#D4A44C", "#8D77AD"],
        "Tamaño de las particiones temporales de PD",
        "Número de créditos etiquetados",
        figuras / "particiones_temporales_pd.svg"
    )


def main():
    raiz = obtener_raiz_proyecto()
    figuras = raiz / "reports" / "figures"
    figuras.mkdir(parents=True, exist_ok=True)

    csv = raiz / "data" / "raw" / "loan" / "loan.csv"

    if csv.is_file():
        print("Generando figuras desde el CSV original...")

        datos = cargar_datos(
            columnas=["loan_status", "issue_d"]
        )
        datos = crear_variable_objetivo(datos)
        datos = preparar_fechas(datos)

        particiones = asignar_particiones(datos)

        graficar_distribucion_etiquetas_pd(
            datos,
            figuras / "distribucion_etiquetas_pd.svg"
        )

        graficar_particiones_temporales_pd(
            particiones,
            figuras / "particiones_temporales_pd.svg"
        )
    else:
        print("CSV no encontrado.")
        print("Generando figuras desde el resumen versionado...")
        generar_desde_resumen(raiz, figuras)

    print("Figuras generadas correctamente.")


if __name__ == "__main__":
    main()
