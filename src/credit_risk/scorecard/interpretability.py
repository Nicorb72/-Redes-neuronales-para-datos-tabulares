"""Permutación agrupada de variables originales sobre validación temporal."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from credit_risk.app.contracts import FEATURES, load_bundle, predict_pd, sha256
from credit_risk.app.settings import load_settings


def feature_groups(columns):
    """Mantiene juntas las columnas one-hot de cada variable categórica."""
    groups = {}
    for name in FEATURES:
        indices = [i for i, column in enumerate(columns)
                   if column == f"numericas__{name}" or column.startswith(f"categoricas__{name}_")]
        if not indices:
            raise ValueError(f"No se encontraron columnas transformadas de {name}.")
        groups[name] = indices
    assigned = [i for indices in groups.values() for i in indices]
    if sorted(assigned) != list(range(len(columns))):
        raise ValueError("Las columnas no forman una partición de variables PD.")
    return groups


def grouped_permutation(bundle, matrix, labels, *, repeats=10, seed=42, max_rows=5000):
    """Importancia = Brier permutado - Brier original; sin reajuste del modelo."""
    y = np.asarray(labels)
    if y.ndim != 1 or len(y) != matrix.shape[0] or not np.isin(y, [0, 1]).all():
        raise ValueError("Se requieren etiquetas 0/1 alineadas con las filas de validación.")
    if len(y) < 2 or set(y.tolist()) != {0, 1}:
        raise ValueError("Se requieren ambas clases en validación.")
    if matrix.shape[1] != len(bundle.columns) or repeats < 2 or max_rows < 2:
        raise ValueError("Dimensiones, repeticiones o tamaño de muestra inválidos.")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(y), size=min(len(y), max_rows), replace=False))
    x = matrix[indices].toarray() if sparse.issparse(matrix) else np.asarray(matrix)[indices].copy()
    y = y[indices]
    if not np.isfinite(x).all() or set(y.tolist()) != {0, 1}:
        raise ValueError("La muestra debe ser finita y contener ambas clases; aumente max_rows.")
    baseline = float(np.mean((predict_pd(bundle.model, x, bundle.metadata["input_format"]) - y) ** 2))
    rows = []
    for name, columns in feature_groups(bundle.columns).items():
        differences = []
        for _ in range(repeats):
            changed = x.copy()
            changed[:, columns] = x[rng.permutation(len(y))][:, columns]
            p = predict_pd(bundle.model, changed, bundle.metadata["input_format"])
            differences.append(float(np.mean((p - y) ** 2) - baseline))
        rows.append(dict(feature=name, importance_mean=float(np.mean(differences)),
                         importance_std=float(np.std(differences, ddof=1)),
                         repetitions=differences))
    return dict(method="grouped_permutation_delta_brier", baseline_brier=baseline,
                rows_used=len(y), rows_available=matrix.shape[0], seed=seed, repeats=repeats,
                sample_indices=indices.tolist(),
                importances=sorted(rows, key=lambda row: row["importance_mean"], reverse=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--matrix", default="data/processed/X_validation_pd.npz")
    parser.add_argument("--labels", default="data/processed/y_validation_pd.npy")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--output", default="reports/interpretability/pd_permutation.json")
    args = parser.parse_args()
    config, _, manifest = load_settings(args.config)
    bundle = load_bundle(manifest)
    result = grouped_permutation(bundle, sparse.load_npz(args.matrix),
                                 np.load(args.labels, allow_pickle=False),
                                 repeats=args.repeats, seed=config["seed"] if args.seed is None else args.seed,
                                 max_rows=args.max_rows)
    result.update(model_id=bundle.metadata["model_id"], bundle_sha256=sha256(manifest),
                  matrix_sha256=sha256(args.matrix), labels_sha256=sha256(args.labels),
                  split="validation (debe verificarse su procedencia)")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    table = pd.DataFrame(result["importances"]).drop(columns="repetitions")
    table.to_csv(output.with_suffix(".csv"), index=False)
    # Figura regenerable; no se fabrica ni se versiona un ranking sin modelo real.
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh(table.feature[::-1], table.importance_mean[::-1], xerr=table.importance_std[::-1])
    axis.axvline(0, color="black", linewidth=0.7)
    axis.set_xlabel("Incremento de Brier al permutar (media ± DE entre permutaciones)")
    axis.set_title("Figura I4-1. Importancia por permutación agrupada de PD")
    figure.text(0.01, 0.01, f"Fuente: modelo {bundle.metadata['model_id']}; validación temporal; semilla {result['seed']}.",
                fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)
    print(f"Análisis guardado: {output}")


if __name__ == "__main__":
    main()
