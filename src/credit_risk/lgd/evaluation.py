"""Evaluación comparativa y análisis distribucional para Pérdida Dado el Incumplimiento (LGD).

Requisitos del encargo:
1. Reportar MAE y RMSE para cada modelo.
2. Comparar la distribución de las predicciones con la distribución observada.
3. Exportar figuras reproducibles y métricas estadísticas (Kolmogorov-Smirnov, Wasserstein, percentiles).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calcular_metricas_lgd(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula métricas de error y correlación para estimaciones de LGD."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    # Correlaciones
    p_corr, _ = stats.pearsonr(y_true, y_pred) if len(y_true) > 1 else (0.0, 1.0)
    s_corr, _ = stats.spearmanr(y_true, y_pred) if len(y_true) > 1 else (0.0, 1.0)

    # Distancia distribucional
    ks_stat, ks_p = stats.ks_2samp(y_true, y_pred)
    wass_dist = stats.wasserstein_distance(y_true, y_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": float(p_corr),
        "spearman_rho": float(s_corr),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_p),
        "wasserstein_distance": float(wass_dist),
    }


def resumen_distribucion(arr: np.ndarray, nombre: str = "dist") -> dict:
    """Calcula momentos y estadísticas descriptivas de una distribución de LGD."""
    arr = np.asarray(arr, dtype=float)
    return {
        "nombre": nombre,
        "n": int(len(arr)),
        "media": float(np.mean(arr)),
        "desv_est": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "mediana": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "prop_exacto_1": float(np.mean(arr >= 0.999)),
    }


def generar_figura_distribucion_lgd(
    y_true: np.ndarray,
    y_pred_baseline: np.ndarray,
    y_pred_mlp: np.ndarray,
    output_path: str | Path,
    titulo: str = "Distribución de LGD: Observada vs. Modelos en Prueba (2014)",
):
    """Genera y guarda en SVG la gráfica comparativa de densidades de LGD."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    # Subplot 1: Densidad (KDE)
    ax1 = axes[0]
    sns.kdeplot(y_true, ax=ax1, label="LGD Observada", color="#1f77b4", linewidth=2.5, clip=(0, 1))
    sns.kdeplot(y_pred_baseline, ax=ax1, label="Regresión Lineal (Baseline)", color="#ff7f0e", linestyle="--", linewidth=2, clip=(0, 1))
    sns.kdeplot(y_pred_mlp, ax=ax1, label="Red Neuronal (MLP)", color="#2ca02c", linestyle="-.", linewidth=2, clip=(0, 1))
    ax1.set_title("Densidad Estimada (KDE)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("LGD (Pérdida dado el Incumplimiento)", fontsize=11)
    ax1.set_ylabel("Densidad", fontsize=11)
    ax1.set_xlim(-0.05, 1.05)
    ax1.legend(loc="upper left", frameon=True)

    # Subplot 2: Histogramas superpuestos
    ax2 = axes[1]
    bins = np.linspace(0, 1, 21)
    ax2.hist(y_true, bins=bins, alpha=0.4, label="LGD Observada", color="#1f77b4", density=True)
    ax2.hist(y_pred_baseline, bins=bins, alpha=0.4, label="Regresión Lineal", color="#ff7f0e", density=True)
    ax2.hist(y_pred_mlp, bins=bins, alpha=0.4, label="Red Neuronal (MLP)", color="#2ca02c", density=True)
    ax2.set_title("Histograma de Frecuencias Relativas", fontsize=13, fontweight="bold")
    ax2.set_xlabel("LGD", fontsize=11)
    ax2.set_ylabel("Densidad Empírica", fontsize=11)
    ax2.set_xlim(-0.05, 1.05)
    ax2.legend(loc="upper left", frameon=True)

    fig.suptitle(titulo, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Figura de distribución guardada en: {output_path}")
