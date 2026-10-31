"""Orquesta baseline, corridas MLP, evaluación y persistencia de artefactos PD.

Ejecutar desde la raíz del proyecto:
    PYTHONPATH=src python -m credit_risk.pd.experiments
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from credit_risk.data.load_data import obtener_raiz_proyecto
from credit_risk.data.prepare_data import preparar_matrices_pd
from credit_risk.pd.baseline import BaselinePD
from credit_risk.pd.evaluation import (
    calcular_metricas_pd,
    crear_tabla_calibracion,
    generar_informe_pd,
    graficar_calibracion,
    graficar_curva_aprendizaje,
    graficar_roc,
)
from credit_risk.pd.neural_network import PDNeuralNetwork, matriz_densa_float32, seleccionar_dispositivo


def cargar_configuracion_pd(raiz: Path | None = None) -> tuple[dict, dict]:
    """Carga los hiperparámetros finales y la configuración general del proyecto."""
    raiz = raiz or obtener_raiz_proyecto()
    config = yaml.safe_load((raiz / "configs" / "pd_mlp_best.yaml").read_text(encoding="utf-8"))
    config_general = yaml.safe_load((raiz / "configs" / "default.yaml").read_text(encoding="utf-8"))
    return config, config_general


def _validar_configuracion(config: dict, config_general: dict) -> None:
    if int(config["positive_class"]) != 1:
        raise ValueError("El pipeline PD requiere que la clase positiva sea default_flag=1.")
    expected_split = {"train": [2007, 2012], "validation": 2013, "test": 2014, "supplementary": 2015}
    if config["split"] != expected_split:
        raise ValueError("Los años configurados deben coincidir con las particiones temporales de prepare_data.py.")
    if int(config_general["evaluation"]["neural_network_seeds"]) < 1:
        raise ValueError("Se requiere al menos una semilla MLP.")


def _evaluar_baseline(config: dict, X: dict, y: dict):
    """Ajusta logística en entrenamiento y evalúa validación/prueba."""
    params = config["baseline"]["logistic_regression"]
    baseline = BaselinePD(
        penalty=params["penalty"],
        solver=params["solver"],
        max_iter=params["max_iter"],
        class_weight=params["class_weight"],
    ).fit(X["train"], y["train"])
    validation_probability = baseline.predict_pd(X["validation"])
    test_probability = baseline.predict_pd(X["test"])
    return baseline, validation_probability, test_probability


def ejecutar_corridas_mlp(
    config: dict,
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    semillas: list[int],
    device=None,
) -> tuple[pd.DataFrame, PDNeuralNetwork, np.ndarray, dict]:
    """Ejecuta N MLP independientes; guarda como final la semilla base configurada."""
    device = device or seleccionar_dispositivo()
    seed_final = int(config["seed"])
    resultados = []
    modelo_final = None
    probabilidad_test_final = None
    metricas_validacion_final = None
    metricas_test_final = None

    for indice, seed in enumerate(semillas, start=1):
        print(f"MLP {indice:02d}/{len(semillas)} — semilla {seed}", flush=True)
        model = PDNeuralNetwork(config, seed=seed, device=device).fit(
            X_train,
            y_train,
            X_validation,
            y_validation,
            max_epochs=int(config["max_epochs"]),
            patience=int(config["early_stopping_patience"]),
        )
        validation_probability = model.predict_proba(X_validation)[:, 1]
        test_probability = model.predict_proba(X_test)[:, 1]
        validation_metrics = calcular_metricas_pd(y_validation, validation_probability)
        test_metrics = calcular_metricas_pd(y_test, test_probability)
        resultados.append({
            "seed": seed,
            "best_epoch": model.best_epoch,
            "epochs_run": model.epochs_run,
            "validation_bce": model.validation_bce,
            "fit_seconds": model.fit_time_sec,
            "n_parameters": model.n_parameters,
            **{f"validation_{name}": value for name, value in validation_metrics.items()},
            **{f"test_{name}": value for name, value in test_metrics.items()},
        })
        if seed == seed_final:
            modelo_final = model
            probabilidad_test_final = test_probability
            metricas_validacion_final = validation_metrics
            metricas_test_final = test_metrics

    if modelo_final is None:
        raise ValueError(f"La lista de semillas debe contener la semilla base {seed_final}.")
    frame = pd.DataFrame(resultados)
    return frame, modelo_final, probabilidad_test_final, {
        "validation": metricas_validacion_final,
        "test": metricas_test_final,
    }


def ejecutar_experimento_completo(demo: bool = False, n_seeds: int | None = None) -> dict:
    """Ejecuta el pipeline PD con la configuración versionada del proyecto.
    
    Parámetros
    ----------
    demo : bool
        Si es True, ejecuta en modo rápido (< 2 minutos) con 1 semilla y 15 épocas.
    n_seeds : int | None
        Número de semillas independientes a evaluar (por defecto 30).
    """
    raiz = obtener_raiz_proyecto()
    config, config_general = cargar_configuracion_pd(raiz)
    _validar_configuracion(config, config_general)
    
    seed_base = int(config["seed"])
    if demo:
        n_seeds = 1
        semillas = [seed_base]
        config["max_epochs"] = 15
        config["early_stopping_patience"] = 4
        print("[Modo DEMO activado para PD: 1 semilla, max_epochs=15, < 2 minutos]")
    else:
        total_seeds = n_seeds or int(config_general["evaluation"]["neural_network_seeds"])
        semillas = list(range(seed_base, seed_base + total_seeds))

    device = seleccionar_dispositivo()
    print(f"Dispositivo: {device}; semillas: {semillas[0]}–{semillas[-1]}", flush=True)

    X, y, preprocessor = preparar_matrices_pd()
    if X["train"].shape[1] != int(config["input_dim"]):
        raise ValueError(
            f"Se esperaban {config['input_dim']} variables procesadas; hay {X['train'].shape[1]}."
        )

    output_dir = raiz / config["artifacts"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline, baseline_validation_probability, baseline_test_probability = _evaluar_baseline(config, X, y)
    logistic_validation_metrics = calcular_metricas_pd(y["validation"], baseline_validation_probability)
    logistic_test_metrics = calcular_metricas_pd(y["test"], baseline_test_probability)
    X_train = matriz_densa_float32(X["train"])
    X_validation = matriz_densa_float32(X["validation"])
    X_test = matriz_densa_float32(X["test"])
    runs, final_model, mlp_test_probability, final_metrics = ejecutar_corridas_mlp(
        config,
        X_train,
        np.asarray(y["train"], dtype=np.float32),
        X_validation,
        np.asarray(y["validation"], dtype=np.float32),
        X_test,
        np.asarray(y["test"], dtype=np.int64),
        semillas,
        device=device,
    )
    runs.to_csv(output_dir / "metricas_por_semilla.csv", index=False)

    metric_names = ["auc", "gini", "ks", "brier", "log_loss"]
    summary = pd.DataFrame([
        {
            "metric": metric,
            "logistic_mean": logistic_test_metrics[metric],
            "logistic_std": 0.0,
            "mlp_mean": float(runs[f"test_{metric}"].mean()),
            "mlp_std": float(runs[f"test_{metric}"].std(ddof=1)),
            "mlp_seed_base": float(final_metrics["test"][metric]),
        }
        for metric in metric_names
    ])
    summary.to_csv(output_dir / "metricas_resumen.csv", index=False)

    seed_base_name = f"MLP (semilla {seed_base})"
    predictions = {
        "Regresión logística": baseline_test_probability,
        seed_base_name: mlp_test_probability,
    }
    calibration = crear_tabla_calibracion(
        y["test"], predictions, n_bins=int(config["evaluation"]["calibration_bins"])
    )
    calibration.to_csv(output_dir / "calibracion_deciles_test.csv", index=False)
    pd.DataFrame({
        "y_true": y["test"],
        "pd_logistic": baseline_test_probability,
        "pd_mlp_seed_base": mlp_test_probability,
    }).to_csv(output_dir / "predicciones_test.csv", index=False)
    graficar_calibracion(
        y["test"], predictions, int(config["evaluation"]["calibration_bins"]),
        output_dir / "calibracion_test.png",
    )
    graficar_roc(y["test"], predictions, output_dir / "curva_roc_test.png")
    graficar_curva_aprendizaje(final_model.train_history, output_dir / "curva_aprendizaje_final.png")
    pd.DataFrame(final_model.train_history).to_csv(output_dir / "curva_aprendizaje_final.csv", index=False)

    if not demo:
        baseline.save(raiz / config["artifacts"]["logistic_model"])
        joblib.dump(preprocessor, raiz / config["artifacts"]["preprocessor"])
        feature_names = list(preprocessor.get_feature_names_out())
        (raiz / config["artifacts"]["feature_names"]).write_text(
            json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        final_model.save(raiz / config["artifacts"]["mlp_model"])

    result_document = {
        "test_period": "2014",
        "train_period": "2007-2012",
        "validation_period": "2013",
        "n_seeds": len(semillas),
        "seeds": semillas,
        "device": str(device),
        "rows": {name: len(y[name]) for name in ("train", "validation", "test")},
        "positive_rate": {name: float(np.mean(y[name])) for name in ("train", "validation", "test")},
        "logistic": {
            "fit_seconds": baseline.fit_time_sec,
            "validation": logistic_validation_metrics,
            "test": logistic_test_metrics,
        },
        "mlp_seed_base": {
            "seed": seed_base,
            "validation": final_metrics["validation"],
            "test": final_metrics["test"],
        },
        "mlp_test_across_seeds": {
            metric: {
                "mean": float(runs[f"test_{metric}"].mean()),
                "std": float(runs[f"test_{metric}"].std(ddof=1)),
            }
            for metric in metric_names
        },
        "interpretation": "MLP standard deviations reflect seed variability on the same temporal holdout, not sampling confidence intervals.",
    }
    json_out = output_dir / ("resultados_demo.json" if demo else "resultados.json")
    json_out.write_text(
        json.dumps(result_document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    hyperparameter_summary_path = raiz / "reports" / "pd_hyperparameter_search" / "resumen_optimizacion.json"
    hyperparameter_summary = (
        json.loads(hyperparameter_summary_path.read_text(encoding="utf-8"))
        if hyperparameter_summary_path.exists()
        else None
    )
    md_out = output_dir / ("informe_pd_demo.md" if demo else "informe_pd.md")
    generar_informe_pd(
        md_out,
        config,
        summary,
        seed_base,
        semillas,
        {name: len(y[name]) for name in ("train", "validation", "test")},
        {name: float(np.mean(y[name])) for name in ("train", "validation", "test")},
        final_metrics["test"],
        logistic_test_metrics,
        hyperparameter_summary,
    )

    print("\nMétricas de prueba 2014 — resumen MLP y referencia logística:")
    print(summary.to_string(index=False))
    print(f"\nArtefactos y explicación guardados en: {output_dir}", flush=True)
    return result_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de experimentos para PD (Probabilidad de Incumplimiento)")
    parser.add_argument("--demo", action="store_true", help="Modo demostración rápida (< 2 minutos) con 1 semilla")
    parser.add_argument("--seeds", type=int, default=None, help="Número de semillas independientes para MLP (por defecto 30)")
    args = parser.parse_args()
    ejecutar_experimento_completo(demo=args.demo, n_seeds=args.seeds)


if __name__ == "__main__":
    main()
