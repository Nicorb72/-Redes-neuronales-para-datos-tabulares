"""Métricas, tablas y figuras de evaluación para modelos de PD."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve


def calcular_metricas_pd(y_true, probability) -> dict[str, float]:
    """Calcula discriminación y error probabilístico para la clase 1."""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    probability = np.clip(np.asarray(probability, dtype=float).reshape(-1), 1e-7, 1 - 1e-7)
    fpr, tpr, _ = roc_curve(y_true, probability)
    auc = float(roc_auc_score(y_true, probability))
    return {
        "auc": auc,
        "gini": 2.0 * auc - 1.0,
        "ks": float(np.max(tpr - fpr)),
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
    }


def crear_tabla_calibracion(y_true, probabilidades: dict[str, np.ndarray], n_bins: int = 10) -> pd.DataFrame:
    """Resume frecuencia observada y PD media en intervalos cuantílicos."""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    tablas = []
    for nombre, probability in probabilidades.items():
        probability = np.asarray(probability, dtype=float).reshape(-1)
        bins = pd.qcut(probability, q=int(n_bins), duplicates="drop")
        tabla = (
            pd.DataFrame({"bin": bins, "observed": y_true, "predicted": probability})
            .groupby("bin", observed=True)
            .agg(n=("observed", "size"), observed_default_rate=("observed", "mean"),
                 mean_predicted_pd=("predicted", "mean"))
            .reset_index()
        )
        tabla.insert(0, "model", nombre)
        tablas.append(tabla)
    return pd.concat(tablas, ignore_index=True)


def graficar_calibracion(y_true, probabilidades, n_bins: int, ruta_salida: str | Path) -> Path:
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Calibración perfecta")
    for nombre, probability in probabilidades.items():
        observed, predicted = calibration_curve(
            y_true, probability, n_bins=int(n_bins), strategy="quantile"
        )
        brier = brier_score_loss(y_true, probability)
        ax.plot(predicted, observed, marker="o", linewidth=1.8,
                label=f"{nombre} (Brier={brier:.4f})")
    ax.set(
        xlim=(0, 1), ylim=(0, 1),
        xlabel="PD media predicha",
        ylabel="Tasa observada de incumplimiento",
        title="Calibración en prueba temporal 2014",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=180)
    plt.close(fig)
    return ruta_salida


def graficar_roc(y_true, probabilidades, ruta_salida: str | Path) -> Path:
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Azar")
    for nombre, probability in probabilidades.items():
        fpr, tpr, _ = roc_curve(y_true, probability)
        ax.plot(fpr, tpr, linewidth=1.8,
                label=f"{nombre} (AUC={roc_auc_score(y_true, probability):.4f})")
    ax.set(
        xlim=(0, 1), ylim=(0, 1),
        xlabel="Tasa de falsos positivos",
        ylabel="Tasa de verdaderos positivos",
        title="Curvas ROC en prueba temporal 2014",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=180)
    plt.close(fig)
    return ruta_salida


def graficar_curva_aprendizaje(historial, ruta_salida: str | Path) -> Path:
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(historial)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(frame["epoch"], frame["train_bce"], marker="o", label="Entrenamiento")
    ax.plot(frame["epoch"], frame["validation_bce"], marker="o", label="Validación 2013")
    ax.set(xlabel="Época", ylabel="Binary cross-entropy", title="Curva de aprendizaje del MLP final")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=180)
    plt.close(fig)
    return ruta_salida


def generar_informe_pd(
    ruta_salida: str | Path,
    config: dict,
    resumen: pd.DataFrame,
    seed_final: int,
    semillas: list[int],
    tamanos: dict,
    tasas_incumplimiento: dict,
    metricas_mlp_final: dict,
    metricas_logistica: dict,
    resumen_hiperparametros: dict | None = None,
) -> Path:
    """Escribe el informe reproducible con metodología, métricas y fórmulas."""
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    labels = {"auc": "AUC", "gini": "Gini", "ks": "KS", "brier": "Brier", "log_loss": "Log-loss"}
    metrics_lines = [
        f"| Métrica | Regresión logística | MLP: media ± desviación estándar ({len(semillas)} semillas) |",
        "|---|---:|---:|",
    ]
    for metric, label in labels.items():
        row = resumen.loc[resumen["metric"] == metric].iloc[0]
        metrics_lines.append(
            f"| {label} | {row['logistic_mean']:.5f} | {row['mlp_mean']:.5f} ± {row['mlp_std']:.5f} |"
        )
    split_labels = {
        "train": "Entrenamiento 2007–2012",
        "validation": "Validación 2013",
        "test": "Prueba 2014",
    }
    split_lines = [
        f"| {split_labels[name]} | {tamanos[name]:,} | {tasas_incumplimiento[name]:.2%} |".replace(",", ".")
        for name in split_labels
    ]
    auc_delta = resumen.loc[resumen.metric == "auc", "mlp_mean"].iloc[0] - metricas_logistica["auc"]
    brier_delta = resumen.loc[resumen.metric == "brier", "mlp_mean"].iloc[0] - metricas_logistica["brier"]
    mlp_auc_std = float(resumen.loc[resumen.metric == "auc", "mlp_std"].iloc[0])
    baseline = config["baseline"]["logistic_regression"]
    dropout = float(config["dropout"])
    lambda_l1 = float(config["l1"])
    hidden_width = int(config["width"])
    input_dim = int(config.get("input_dim", 44))
    mlp_parameters = input_dim * hidden_width + hidden_width
    for _ in range(int(config["n_layers"]) - 1):
        mlp_parameters += hidden_width * hidden_width + hidden_width
    mlp_parameters += hidden_width + 1
    mlp_parameter_label = f"{mlp_parameters:,}".replace(",", ".")
    hyperparameter_conclusion = ""
    if resumen_hiperparametros:
        phase1_auc = float(resumen_hiperparametros["phase1_best_auc"])
        phase2_auc = float(resumen_hiperparametros["phase2_best_auc"])
        robustness_mean = float(resumen_hiperparametros["selected_robustness_mean_auc"])
        robustness_std = float(resumen_hiperparametros["selected_robustness_std_auc"])
        hyperparameter_conclusion = (
            f"En la búsqueda de hiperparámetros, 150 ensayos iniciales alcanzaron AUC de validación "
            f"{phase1_auc:.5f} y los 30 ensayos de refinamiento {phase2_auc:.5f}; la ganancia puntual "
            f"fue {phase2_auc - phase1_auc:.5f}. La configuración seleccionada obtuvo AUC medio "
            f"{robustness_mean:.5f} ± {robustness_std:.5f} al repetir cinco semillas. El refinamiento "
            "aportó una mejora pequeña frente a la variación entre semillas, señal de rendimientos "
            "decrecientes en el espacio explorado."
        )
    content = f"""# Evaluación de probabilidad de incumplimiento (PD)

## Diseño experimental

Se reutilizaron las particiones temporales y el preprocesador del proyecto: entrenamiento 2007–2012, validación 2013, prueba 2014 y conjunto complementario 2015. El preprocesador se ajustó solo con entrenamiento. La validación controla parada temprana del MLP; la prueba 2014 se reserva para esta evaluación final. El conjunto de 2015 no se usa por su alta proporción de resultados sin etiqueta.

| Partición | Préstamos etiquetados | Tasa de incumplimiento |
|---|---:|---:|
{chr(10).join(split_lines)}

La tasa de incumplimiento aumenta de entrenamiento a validación y prueba. Esto evidencia un cambio temporal en la población etiquetada; por ello la evaluación de 2014 es más exigente y las probabilidades requieren cautela.

La regresión logística es el modelo de referencia sin regularización (`penalty=None`), con solver `{baseline['solver']}` y sin ponderación de clases; se ajustó sobre las mismas observaciones y características transformadas que el MLP. El MLP usa {config['n_layers']} capas ocultas de {config['width']} neuronas, activación {config['activation']}, dropout {dropout:.4f}, optimizador {config['optimizer']}, tasa de aprendizaje {config['learning_rate']:.6f}, lote {config['batch_size']} y regularización L1 con λ={lambda_l1:.3g}. Se ejecutaron {len(semillas)} semillas ({semillas[0]}–{semillas[-1]}). El checkpoint entregado se ajustó con la semilla fija {seed_final}.

Las corridas comparten los mismos conjuntos temporales e hiperparámetros; solo cambia la semilla de inicialización y el orden de minibatches. La desviación estándar describe variabilidad por aleatoriedad del entrenamiento, no incertidumbre muestral. El modelo entregado usa la semilla configurada {seed_final}; no se seleccionó una semilla usando la prueba.

## Resultados en prueba temporal 2014

{chr(10).join(metrics_lines)}

El AUC medio del MLP supera al de la logística en {auc_delta:+.5f}; el Brier medio cambia en {brier_delta:+.5f} (negativo favorece al MLP). La mejora es moderada y debe interpretarse junto a la variabilidad entre semillas y al cambio de tasa de incumplimiento entre periodos. El MLP de semilla fija {seed_final} obtuvo: AUC={metricas_mlp_final['auc']:.5f}, Gini={metricas_mlp_final['gini']:.5f}, KS={metricas_mlp_final['ks']:.5f}, Brier={metricas_mlp_final['brier']:.5f} y log-loss={metricas_mlp_final['log_loss']:.5f}. La regresión logística obtuvo: AUC={metricas_logistica['auc']:.5f}, Gini={metricas_logistica['gini']:.5f}, KS={metricas_logistica['ks']:.5f}, Brier={metricas_logistica['brier']:.5f} y log-loss={metricas_logistica['log_loss']:.5f}.

## Definiciones matemáticas

Para cada préstamo i, yᵢ ∈ {{0,1}} es el resultado y pᵢ ∈ [0,1] la PD estimada. La regresión logística calcula pᵢ = σ(β₀ + βᵀxᵢ), donde σ(z)=1/(1+e⁻ᶻ), y estima β minimizando la entropía cruzada binaria sin término de regularización. El MLP calcula pᵢ = σ(fθ(xᵢ)); sus pesos se entrenan minimizando la entropía cruzada binaria más la penalización L1: L(θ)=−(1/n)Σᵢ[yᵢlog(pᵢ)+(1−yᵢ)log(1−pᵢ)] + λΣⱼ|wⱼ|. El dropout desactiva aleatoriamente unidades ocultas durante entrenamiento con probabilidad {dropout:.4f}.

- **AUC:** área bajo ROC; mide la probabilidad de ordenar un incumplidor por encima de un buen pagador.
- **Gini:** Gini = 2·AUC − 1.
- **KS:** KS = maxₜ |TPR(t) − FPR(t)|, máxima separación entre las distribuciones acumuladas de las dos clases.
- **Brier:** (1/n)Σᵢ(pᵢ−yᵢ)²; menor valor indica mejor precisión probabilística.
- **Calibración:** compara, por grupos de riesgo, la PD media predicha con la tasa observada. La diagonal representa calibración ideal.

## Lectura y límites

La curva de calibración en prueba muestra que ambos modelos subestiman la tasa observada en varios grupos, especialmente entre los de mayor PD predicha. No se ajustó un calibrador sobre la prueba. Si se calibra el modelo, el calibrador debe ajustarse con datos de desarrollo y su efecto evaluarse posteriormente en una partición independiente.

La comparación debe considerar discriminación y calibración, no solo AUC. Las semillas miden sensibilidad a la aleatoriedad del MLP sobre una partición fija; no reemplazan una validación temporal adicional. La prueba se originó en 2014, pero los estados crediticios del archivo pueden haberse observado después de la originación; por tanto, este experimento es una evaluación temporal de originaciones y no demuestra por sí solo una validación histórica con horizonte de desempeño fijo. Las probabilidades dependen también de la definición de la etiqueta y de la selección de préstamos con estado observable.

## Conclusiones: por qué la mejora del MLP es limitada

El MLP **sí supera ligeramente a la regresión logística en promedio** en esta prueba: la diferencia de AUC es {auc_delta:+.5f}, mientras que la desviación estándar del AUC entre semillas del MLP es {mlp_auc_std:.5f}. Esa desviación mide sensibilidad a la inicialización sobre el mismo conjunto y no es un intervalo de confianza; por tanto, no permite afirmar que la ventaja sea estadísticamente concluyente o que se mantenga en otros periodos. El MLP también obtiene Brier y log-loss medios menores, aunque la curva de calibración muestra subestimación en varios grupos de riesgo.

{hyperparameter_conclusion}

La configuración elegida terminó siendo una red compacta de {config['n_layers']} capas de {config['width']} neuronas y {mlp_parameter_label} parámetros, con dropout {dropout:.3f} y regularización L1. Que la búsqueda no favorezca una arquitectura más grande es compatible con que las 44 variables procesadas disponibles aporten relaciones que la regresión logística ya captura en buena medida, o con que las interacciones no lineales adicionales no generalicen mejor. La optimización no demuestra cuál de estas explicaciones es la causa.

También hay un cambio temporal marcado en la tasa observada: {tasas_incumplimiento['train']:.2%} en entrenamiento, {tasas_incumplimiento['validation']:.2%} en validación y {tasas_incumplimiento['test']:.2%} en prueba. Esto apunta a cambio de población o de madurez/observabilidad de las etiquetas. La curva de aprendizaje muestra que la pérdida de entrenamiento continúa bajando mientras la de validación queda más alta y alcanza su mínimo cerca de la época 11. Aunque se aplicaron dropout y L1, ese patrón sugiere que el límite principal puede estar en la generalización temporal y la información predictiva disponible, no solo en encontrar otra combinación de hiperparámetros.

Por ello, los resultados también pueden estar limitados por la estructura y cobertura de los datos y por cómo se forman los conjuntos de entrenamiento, validación y prueba. La separación temporal es apropiada para probar el desempeño en cohortes posteriores, pero aquí coincide con diferencias grandes en la tasa etiquetada y los estados de los préstamos pueden haberse observado después de originarlos. Además, una parte importante del archivo no tiene etiqueta definitiva. Estos factores pueden cambiar qué préstamos entran al análisis y qué patrones aprende el modelo; con esta única comparación no es posible atribuirles causalmente el rendimiento observado.

Antes de llevar una PD a producción sería necesario un análisis y procesamiento más exhaustivo: revisar madurez y horizonte de las etiquetas, disponibilidad de cada predictor al momento de originación, faltantes y distribuciones por cohorte, y estabilidad temporal de la población. También convendría construir particiones que reproduzcan el uso previsto y estudiar transformaciones o variables derivadas justificadas por el negocio. Ese trabajo podría mejorar la información disponible y la generalización, aunque no garantiza por sí solo un aumento de métricas.

En conjunto, los resultados no indican que la red sea peor: muestran una **ganancia modesta y sensible al periodo**, sin evidencia suficiente para justificar mayor complejidad o declarar superioridad general frente al baseline. La comparación final debe conservar la prueba temporal como evaluación única y reconocer que el análisis utiliza estados crediticios observados en una fecha posterior a la originación.

## Artefactos

- `metricas_por_semilla.csv`: métricas de validación y prueba por semilla del MLP.
- `metricas_resumen.csv` y `resultados.json`: resumen MLP frente a regresión logística.
- `calibracion_deciles_test.csv`: tasas observadas y PD medias por decil.
- `predicciones_test.csv`: probabilidades de ambos modelos en la prueba temporal.
- `curva_aprendizaje_final.png`, `curva_roc_test.png` y `calibracion_test.png`: gráficos de diagnóstico y comparación.
- `artifacts/models/pd_mlp_final.pt`, `pd_logistic.joblib` y `pd_preprocessor.joblib`: modelos y preprocesador.
"""
    ruta_salida.write_text(content, encoding="utf-8")
    return ruta_salida
