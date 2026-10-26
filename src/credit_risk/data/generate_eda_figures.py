"""Genera las figuras exploratorias usadas en el reporte técnico."""

from credit_risk.data.eda import (
    graficar_distribucion_etiquetas_pd,
    graficar_particiones_temporales_pd,
)
from credit_risk.data.load_data import cargar_datos, obtener_raiz_proyecto
from credit_risk.data.prepare_data import asignar_particiones
from credit_risk.data.preprocessing import (
    crear_variable_objetivo,
    preparar_fechas,
)


def main():
    raiz = obtener_raiz_proyecto()
    datos = cargar_datos(columnas=["loan_status", "issue_d"])
    datos = crear_variable_objetivo(datos)
    datos = preparar_fechas(datos)

    particiones = asignar_particiones(datos)

    figuras = raiz / "reports" / "figures"
    ruta_etiquetas = graficar_distribucion_etiquetas_pd(
        datos,
        figuras / "distribucion_etiquetas_pd.svg"
    )
    ruta_particiones = graficar_particiones_temporales_pd(
        particiones,
        figuras / "particiones_temporales_pd.svg"
    )

    print("Figuras generadas:")
    print(f"- {ruta_etiquetas}")
    print(f"- {ruta_particiones}")


if __name__ == "__main__":
    main()
