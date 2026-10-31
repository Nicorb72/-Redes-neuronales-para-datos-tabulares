"""Protocolo de experimentación justa, búsqueda de hiperparámetros y corridas de 30 semillas para LGD.

Cumple estrictamente las reglas de comparación del trabajo:
1. Mismos datos y preprocesamiento idéntico (entrenamiento 2007-2012, validación 2013, prueba 2014).
2. Presupuesto de búsqueda explícito: registra número de arquitecturas evaluadas y tiempo acumulado.
3. Evaluación sobre 30 semillas independientes para la arquitectura final seleccionada.
4. Reporte de complejidad (número de parámetros entrenables) y métricas: media, desviación estándar, min y max.
5. Modo de demostración (< 2 minutos) con un único comando para la sustentación.
"""
import argparse
import json
import re
import time
from pathlib import Path
import numpy as np
from scipy import sparse
import yaml

from credit_risk.data.load_data import obtener_raiz_proyecto
from credit_risk.lgd.baseline import BaselineLGD
from credit_risk.lgd.evaluation import (
    calcular_metricas_lgd,
    resumen_distribucion,
    generar_figura_distribucion_lgd,
)
from credit_risk.lgd.neural_network import LGDNeuralNetwork


def cargar_configuracion_lgd(ruta_config: Path | str | None = None) -> dict:
    """Carga configuración YAML para LGD desde configs/lgd_mlp_best.yaml o ruta personalizada."""
    raiz = obtener_raiz_proyecto()
    if ruta_config is None:
        ruta = raiz / "configs" / "lgd_mlp_best.yaml"
    else:
        ruta = Path(ruta_config)
        if not ruta.is_absolute():
            ruta = raiz / ruta

    if not ruta.exists():
        # Valores por defecto seguros si el archivo no existe
        return {
            "hidden_dims": [64, 32],
            "activation": "relu",
            "output_activation": "sigmoid",
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "dropout": 0.2,
            "loss_function": "mse",
            "batch_size": 128,
            "early_stopping_patience": 8,
            "max_epochs": 60,
            "seed": 42,
            "seeds": 30,
        }

    with open(ruta, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolver_epocas(config: dict, demo: bool, max_epochs: int | None) -> int:
    if max_epochs is not None:
        return max_epochs
    if demo:
        return int(config.get("demo", {}).get("epochs", 15))
    return int(config.get("max_epochs", 60))


def cargar_datos_procesados_lgd():
    """Carga matrices X, etiquetas y ead preprocesadas desde data/processed/.

    Si las matrices no existen (ej. clon limpio), se generan automáticamente
    a partir del dataset crudo garantizando reproducibilidad en un solo comando.
    """
    raiz = obtener_raiz_proyecto()
    d_dir = raiz / "data" / "processed"

    splits = ["train", "validation", "test"]
    archivos_requeridos = []
    for s in splits:
        archivos_requeridos.extend([
            d_dir / f"X_{s}_lgd.npz",
            d_dir / f"y_{s}_lgd.npy",
            d_dir / f"ead_{s}_lgd.npy",
        ])

    archivos_faltantes = [f for f in archivos_requeridos if not f.exists()]
    if archivos_faltantes:
        print("[LGD Pipeline] Matrices procesadas no encontradas en data/processed/.")
        print(f"[LGD Pipeline] Faltan {len(archivos_faltantes)} archivos. Generando automáticamente con credit_risk.lgd.data...")
        from credit_risk.lgd.data import cargar_y_preparar_lgd, preparar_matrices_lgd, guardar_matrices_lgd

        df_cast = cargar_y_preparar_lgd()
        X_trans, y_dict, ead_dict, preproc = preparar_matrices_lgd(df_cast)
        guardar_matrices_lgd(X_trans, y_dict, ead_dict, preproc)
        print("[LGD Pipeline] Matrices de LGD generadas y persistidas exitosamente.")

    X, y, ead = {}, {}, {}
    for s in splits:
        X[s] = sparse.load_npz(d_dir / f"X_{s}_lgd.npz")
        y[s] = np.load(d_dir / f"y_{s}_lgd.npy")
        ead[s] = np.load(d_dir / f"ead_{s}_lgd.npy")
    return X, y, ead


def explorar_hiperparametros(
    X_train,
    y_train,
    X_val,
    y_val,
    demo: bool = False,
    epochs: int | None = None,
    patience: int = 8,
    batch_size: int = 128,
    weight_decay: float = 1e-4,
    activation: str = "relu",
    output_activation: str = "sigmoid",
):
    """Búsqueda sistemática de arquitecturas para MLP con presupuesto explícito."""
    if epochs is None:
        epochs = 15 if demo else 60
    if demo:
        grid = [
            {"hidden_dims": [32], "dropout": 0.1, "lr": 1e-3, "loss": "mse", "epochs": epochs},
            {"hidden_dims": [64, 32], "dropout": 0.2, "lr": 1e-3, "loss": "mse", "epochs": epochs},
        ]
    else:
        grid = [
            {"hidden_dims": [32], "dropout": 0.1, "lr": 1e-3, "loss": "mse", "epochs": epochs},
            {"hidden_dims": [64, 32], "dropout": 0.2, "lr": 1e-3, "loss": "mse", "epochs": epochs},
            {"hidden_dims": [128, 64], "dropout": 0.2, "lr": 5e-4, "loss": "mse", "epochs": epochs},
            {"hidden_dims": [64, 32], "dropout": 0.2, "lr": 1e-3, "loss": "huber", "epochs": epochs},
            {"hidden_dims": [64, 32], "dropout": 0.1, "lr": 1e-3, "loss": "bce", "epochs": epochs},
        ]

    for cfg in grid:
        cfg.update({
            "patience": patience,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "activation": activation,
            "output_activation": output_activation,
        })

    resultados = []
    t0 = time.time()

    for idx, cfg in enumerate(grid, 1):
        t_start = time.time()
        model = LGDNeuralNetwork(
            hidden_dims=cfg["hidden_dims"],
            dropout=cfg["dropout"],
            learning_rate=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            loss_function=cfg["loss"],
            epochs=cfg["epochs"],
            patience=cfg["patience"],
            batch_size=cfg["batch_size"],
            activation=cfg["activation"],
            output_activation=cfg["output_activation"],
            seed=42,
        )
        model.fit(X_train, y_train, X_val, y_val, verbose=False)
        val_mae, val_rmse = model.evaluate(X_val, y_val)
        dur = time.time() - t_start

        record = {
            "id": idx,
            "config": cfg,
            "params": model.n_parameters,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "duration_sec": dur,
        }
        resultados.append(record)

    total_time = time.time() - t0
    # Ordenar por RMSE en validación
    resultados.sort(key=lambda r: r["val_rmse"])
    mejor = resultados[0]

    return {
        "presupuesto_configuraciones": len(grid),
        "tiempo_total_busqueda_sec": total_time,
        "todas_las_configuraciones": resultados,
        "mejor_configuracion": mejor,
    }


def ejecutar_30_semillas_mlp(best_cfg, X_train, y_train, X_val, y_val, X_test, y_test, n_seeds: int = 30):
    """Selecciona por RMSE de validación; evalúa test después de fijar el checkpoint."""
    if n_seeds < 1:
        raise ValueError("Se requiere al menos una semilla LGD.")
    test_maes = []
    test_rmses = []
    fit_times = []
    models = []
    validation_metrics = []

    print(f"\nIniciando {n_seeds} corridas independientes para MLP...")
    for seed in range(1, n_seeds + 1):
        m = LGDNeuralNetwork(
            hidden_dims=best_cfg["hidden_dims"],
            dropout=best_cfg["dropout"],
            learning_rate=best_cfg["lr"],
            weight_decay=best_cfg.get("weight_decay", 1e-4),
            loss_function=best_cfg["loss"],
            epochs=best_cfg["epochs"],
            patience=best_cfg.get("patience", 8),
            batch_size=best_cfg.get("batch_size", 128),
            activation=best_cfg.get("activation", "relu"),
            output_activation=best_cfg.get("output_activation", "sigmoid"),
            seed=seed * 101,
        )
        m.fit(X_train, y_train, X_val, y_val, verbose=False)
        val_mae, val_rmse = m.evaluate(X_val, y_val)
        if not np.isfinite([val_mae, val_rmse]).all():
            raise ValueError(f"Métricas de validación no finitas para la semilla {m.seed}.")
        validation_metrics.append({"mae": val_mae, "rmse": val_rmse})
        fit_times.append(m.fit_time_sec)
        models.append(m)

    # La selección queda fijada antes de consultar test. Empates: primera semilla.
    best_idx = int(np.argmin([record["rmse"] for record in validation_metrics]))
    best_trained_model = models[best_idx]
    selection = {
        "criterion": "rmse",
        "split": "validation",
        "seed": best_trained_model.seed,
        "validation_mae": float(validation_metrics[best_idx]["mae"]),
        "validation_rmse": float(validation_metrics[best_idx]["rmse"]),
        "tie_break": "first_seed_in_evaluation_order",
        "test_used_for_selection": False,
    }
    per_seed = []
    for model, validation in zip(models, validation_metrics):
        mae, rmse = model.evaluate(X_test, y_test)
        test_maes.append(mae)
        test_rmses.append(rmse)
        per_seed.append({
            "seed": model.seed,
            "validation_mae": float(validation["mae"]),
            "validation_rmse": float(validation["rmse"]),
            "test_mae": float(mae),
            "test_rmse": float(rmse),
            "fit_seconds": float(model.fit_time_sec),
            "selected": model is best_trained_model,
        })

    stats_summary = {
        "n_semillas": n_seeds,
        "checkpoint_selection": selection,
        "per_seed_metrics": per_seed,
        "parametros": best_trained_model.n_parameters,
        "mae": {
            "media": float(np.mean(test_maes)),
            "std": float(np.std(test_maes)),
            "min": float(np.min(test_maes)),
            "max": float(np.max(test_maes)),
        },
        "rmse": {
            "media": float(np.mean(test_rmses)),
            "std": float(np.std(test_rmses)),
            "min": float(np.min(test_rmses)),
            "max": float(np.max(test_rmses)),
        },
        "tiempo_ajuste_sec": {
            "media": float(np.mean(fit_times)),
            "std": float(np.std(fit_times)),
            "total": float(np.sum(fit_times)),
        },
        "semillas_evaluadas": [s * 101 for s in range(1, n_seeds + 1)],
    }
    return stats_summary, best_trained_model


def ejecutar_experimento_completo(
    config_path: Path | str | None = None,
    demo: bool = False,
    n_seeds: int | None = None,
    hidden_dims: list[int] | None = None,
    learning_rate: float | None = None,
    max_epochs: int | None = None,
    dropout: float | None = None,
    loss_function: str | None = None,
    patience: int | None = None,
    batch_size: int | None = None,
    weight_decay: float | None = None,
):
    """Flujo end-to-end de experimentación para el Integrante 3.
    
    Carga hiperparámetros desde archivo de configuración YAML (configs/lgd_mlp_best.yaml)
    y permite sobreescribir cualquier parámetro por CLI sin modificar el código fuente.
    """
    config = cargar_configuracion_lgd(config_path)

    # Sobrescrituras por CLI o argumentos directos
    if hidden_dims is not None:
        config["hidden_dims"] = hidden_dims
    if learning_rate is not None:
        config["learning_rate"] = learning_rate
    if max_epochs is not None:
        config["max_epochs"] = max_epochs
    if dropout is not None:
        config["dropout"] = dropout
    if loss_function is not None:
        config["loss_function"] = loss_function
    if patience is not None:
        config["early_stopping_patience"] = patience
    if batch_size is not None:
        config["batch_size"] = batch_size
    if weight_decay is not None:
        config["weight_decay"] = weight_decay

    demo_config = config.get("demo", {})
    seeds_count = n_seeds if n_seeds is not None else (
        int(demo_config.get("seeds", 5)) if demo else int(config.get("seeds", 30))
    )
    epochs_run = _resolver_epocas(config, demo, max_epochs)
    patience_run = int(config.get("early_stopping_patience", 8))
    batch_size_run = int(config.get("batch_size", 128))
    weight_decay_run = float(config.get("weight_decay", 1e-4))

    mejor_cfg = {
        "hidden_dims": config.get("hidden_dims", [64, 32]),
        "dropout": float(config.get("dropout", 0.2)),
        "lr": float(config.get("learning_rate", 0.001)),
        "loss": str(config.get("loss_function", "mse")),
        "epochs": epochs_run,
        "patience": patience_run,
        "batch_size": batch_size_run,
        "weight_decay": weight_decay_run,
        "activation": str(config.get("activation", "relu")),
        "output_activation": str(config.get("output_activation", "sigmoid")),
    }

    print("=" * 65)
    print(f"EXPERIMENTACIÓN LGD (Pérdida Dado el Incumplimiento) {'[MODO DEMO]' if demo else ''}")
    print("=" * 65)
    print(f"Configuración activa: hidden_dims={mejor_cfg['hidden_dims']}, lr={mejor_cfg['lr']}, "
          f"dropout={mejor_cfg['dropout']}, loss={mejor_cfg['loss']}, epochs={mejor_cfg['epochs']}, "
          f"patience={mejor_cfg['patience']}, batch_size={mejor_cfg['batch_size']}, "
          f"weight_decay={mejor_cfg['weight_decay']}, seeds={seeds_count}")

    X, y, ead = cargar_datos_procesados_lgd()
    raiz = obtener_raiz_proyecto()

    # 1. Ajustar Modelo de Referencia (Baseline)
    print("\n1. Ajustando Modelo de Referencia (Regresión Lineal)...")
    t0_base = time.time()
    baseline = BaselineLGD()
    baseline.fit(X["train"], y["train"])
    base_time = time.time() - t0_base
    base_mae, base_rmse = baseline.evaluate(X["test"], y["test"])
    preds_base = baseline.predict(X["test"])

    print(f"   Baseline Params: {baseline.n_parameters} | Tiempo: {base_time:.3f}s")
    print(f"   Baseline Test MAE: {base_mae:.4f} | RMSE: {base_rmse:.4f}")

    # 2. Búsqueda de Hiperparámetros
    print("\n2. Exploración sistemática de arquitecturas de Red Neuronal...")
    busqueda = explorar_hiperparametros(
        X["train"],
        y["train"],
        X["validation"],
        y["validation"],
        demo=demo,
        epochs=epochs_run,
        patience=patience_run,
        batch_size=batch_size_run,
        weight_decay=weight_decay_run,
        activation=str(config.get("activation", "relu")),
        output_activation=str(config.get("output_activation", "sigmoid")),
    )
    # Conservar la configuración personalizada; si no, usar la mejor de la búsqueda.
    if (
        hidden_dims is None
        and loss_function is None
        and learning_rate is None
        and dropout is None
    ):
        mejor_cfg = busqueda["mejor_configuracion"]["config"]
    mejor_cfg.update({
        "patience": patience_run,
        "batch_size": batch_size_run,
        "weight_decay": weight_decay_run,
        "activation": str(config.get("activation", "relu")),
        "output_activation": str(config.get("output_activation", "sigmoid")),
    })
    print(f"   Presupuesto: {busqueda['presupuesto_configuraciones']} arquitecturas evaluadas en {busqueda['tiempo_total_busqueda_sec']:.2f}s")
    print(f"   Arquitectura seleccionada para benchmark: {mejor_cfg['hidden_dims']}, dropout={mejor_cfg['dropout']}, loss={mejor_cfg['loss']}")

    # 3. Corridas Independientes
    stats_30, best_mlp = ejecutar_30_semillas_mlp(
        mejor_cfg,
        X["train"], y["train"],
        X["validation"], y["validation"],
        X["test"], y["test"],
        n_seeds=seeds_count,
    )

    print(f"Checkpoint LGD seleccionado por validación: semilla {best_mlp.seed}; "
          f"RMSE={stats_30['checkpoint_selection']['validation_rmse']:.6f}")
    preds_mlp = best_mlp.predict(X["test"])
    metrics_best_mlp = calcular_metricas_lgd(y["test"], preds_mlp)

    print("\nRESULTADOS FINALES SOBRE PRUEBA (TEST 2014):")
    print("Baseline (Regresión Lineal):")
    print(f"  MAE:  {base_mae:.4f}")
    print(f"  RMSE: {base_rmse:.4f}")
    print(f"  Parámetros: {baseline.n_parameters}")
    print(f"Red Neuronal MLP ({seeds_count} semillas):")
    print(f"  MAE  Media: {stats_30['mae']['media']:.4f} +/- {stats_30['mae']['std']:.4f} [min: {stats_30['mae']['min']:.4f}, max: {stats_30['mae']['max']:.4f}]")
    print(f"  RMSE Media: {stats_30['rmse']['media']:.4f} +/- {stats_30['rmse']['std']:.4f} [min: {stats_30['rmse']['min']:.4f}, max: {stats_30['rmse']['max']:.4f}]")
    print(f"  Parámetros: {best_mlp.n_parameters}")

    # 4. Guardar Modelos y Artefactos
    modelos_dir = raiz / "artifacts" / "models"
    figures_dir = raiz / "reports" / "figures"
    modelos_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Las demos y ejecuciones parciales/personalizadas no deben reemplazar los
    # artefactos oficiales consumidos por el reporte y la integración PD–LGD.
    configuracion_personalizada = config_path is not None or any(value is not None for value in (
        hidden_dims, learning_rate, max_epochs, dropout, loss_function,
        patience, batch_size, weight_decay,
    ))
    benchmark_oficial = not demo and not configuracion_personalizada and seeds_count == 30
    if demo:
        run_mode, suffix = "demo", "_demo"
    elif benchmark_oficial:
        run_mode, suffix = "full_benchmark", ""
    else:
        run_mode = "custom"
        signature = (
            f"h{'x'.join(map(str, mejor_cfg['hidden_dims']))}_lr{mejor_cfg['lr']:g}_"
            f"d{mejor_cfg['dropout']:g}_{mejor_cfg['loss']}_e{mejor_cfg['epochs']}_"
            f"b{mejor_cfg['batch_size']}_p{mejor_cfg['patience']}_"
            f"wd{mejor_cfg['weight_decay']:g}_s{seeds_count}"
        )
        suffix = "_custom_" + re.sub(r"[^A-Za-z0-9_-]+", "-", signature.replace(".", "p"))

    baseline.save(modelos_dir / f"baseline_lgd{suffix}.joblib")
    best_mlp.save(modelos_dir / f"mlp_lgd{suffix}.pt")

    # Guardar gráfica comparativa de distribución
    fig_name = f"distribucion_lgd{suffix}.svg" if suffix else "distribucion_lgd_observada_vs_predicha.svg"
    generar_figura_distribucion_lgd(
        y["test"],
        preds_base,
        preds_mlp,
        figures_dir / fig_name,
    )

    # 5. Guardar resumen consolidado de benchmark
    benchmark_data = {
        "baseline": {
            "tipo": "LinearRegression (Clipped [0, 1])",
            "parametros": baseline.n_parameters,
            "tiempo_ajuste_sec": base_time,
            "test_mae": base_mae,
            "test_rmse": base_rmse,
            "metricas_completas": calcular_metricas_lgd(y["test"], preds_base),
            "resumen_distribucion": resumen_distribucion(preds_base, "Baseline"),
        },
        "busqueda_hiperparametros": busqueda,
        f"mlp_{seeds_count}_semillas": stats_30,
        "run_metadata": {
            "mode": run_mode,
            "n_seeds": seeds_count,
            "selected_loss": mejor_cfg["loss"],
            "selected_hidden_dims": mejor_cfg["hidden_dims"],
            "effective_training_config": {
                "activation": mejor_cfg["activation"],
                "output_activation": mejor_cfg["output_activation"],
                "learning_rate": mejor_cfg["lr"],
                "weight_decay": mejor_cfg["weight_decay"],
                "dropout": mejor_cfg["dropout"],
                "loss_function": mejor_cfg["loss"],
                "batch_size": mejor_cfg["batch_size"],
                "max_epochs": mejor_cfg["epochs"],
                "early_stopping_patience": mejor_cfg["patience"],
                "optimizer": "AdamW",
            },
            "checkpoint_selection": stats_30["checkpoint_selection"],
        },
        "mejor_mlp_individual": {
            "seed": best_mlp.seed,
            "checkpoint_selection": stats_30["checkpoint_selection"],
            "configuracion": mejor_cfg,
            "parametros": best_mlp.n_parameters,
            "metricas_test": metrics_best_mlp,
            "resumen_distribucion": resumen_distribucion(preds_mlp, "MLP"),
        },
        "distribucion_observada": resumen_distribucion(y["test"], "Observada"),
    }

    if benchmark_oficial:
        results_path = raiz / config["artifacts"]["benchmark_results"]
    elif run_mode == "demo":
        results_path = modelos_dir / "lgd_demo_results.json"
    else:
        results_path = modelos_dir / f"lgd{suffix}_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=4, ensure_ascii=False)

    print("\nArtefactos y figuras guardados exitosamente.")
    return benchmark_data


def main():
    parser = argparse.ArgumentParser(description="Experimentos de LGD (Integrante 3)")
    parser.add_argument("--config", type=str, default=None, help="Ruta al archivo YAML de hiperparámetros (default: configs/lgd_mlp_best.yaml)")
    parser.add_argument("--demo", action="store_true", help="Modo demostración rápida (< 2 minutos)")
    parser.add_argument("--seeds", type=int, default=None, help="Número de semillas para el benchmark de la red")
    parser.add_argument("--hidden-dims", type=str, default=None, help="Dimensiones ocultas separadas por coma (ej. '64,32' o '128,64')")
    parser.add_argument("--lr", type=float, default=None, help="Tasa de aprendizaje (learning rate)")
    parser.add_argument("--epochs", type=int, default=None, help="Número máximo de épocas")
    parser.add_argument("--dropout", type=float, default=None, help="Probabilidad de dropout (ej. 0.2)")
    parser.add_argument("--loss", type=str, default=None, choices=["mse", "huber", "bce", "mae"], help="Función de pérdida")
    parser.add_argument("--patience", type=int, default=None, help="Paciencia para early stopping")
    parser.add_argument("--batch-size", type=int, default=None, help="Tamaño de lote (batch size)")
    parser.add_argument("--weight-decay", type=float, default=None, help="Decaimiento de pesos L2/AdamW")
    args = parser.parse_args()

    dims = None
    if args.hidden_dims:
        dims = [int(x.strip()) for x in args.hidden_dims.split(",") if x.strip()]

    ejecutar_experimento_completo(
        config_path=args.config,
        demo=args.demo,
        n_seeds=args.seeds,
        hidden_dims=dims,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        dropout=args.dropout,
        loss_function=args.loss,
        patience=args.patience,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
    )


if __name__ == "__main__":
    main()
