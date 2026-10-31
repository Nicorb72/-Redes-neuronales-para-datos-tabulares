"""Resume evidencia guardada y escenarios PD del reporte; no entrena modelos."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
import torch  # noqa: E402

from credit_risk.app.contracts import load_bundle  # noqa: E402


def artifact_hash(path):
    payload = path.read_bytes()
    if path.suffix.lower() in {".json", ".csv"}:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def collect():
    torch.set_num_threads(1)
    manifest = ROOT / "artifacts/models/pd_bundle.json"
    bundle = load_bundle(manifest)
    logistic = joblib.load(ROOT / "artifacts/models/pd_logistic.joblib")
    matrix_path = ROOT / "data/processed/X_validation_pd.npz"
    matrix = sparse.load_npz(matrix_path).tocsr()
    indices = np.random.default_rng(42).choice(matrix.shape[0], 5000, replace=False)
    sample = matrix[indices].toarray().astype(np.float32)
    names = list(bundle.preprocessor.get_feature_names_out())
    scaler = bundle.preprocessor.named_transformers_["numericas"].named_steps["escalamiento"]
    scenarios = []
    for variable in ("dti", "annual_inc", "loan_amnt"):
        column = names.index(f"numericas__{variable}")
        low, high = np.quantile(matrix[:, column].toarray().ravel(), [0.25, 0.75])
        before, after = sample.copy(), sample.copy()
        before[:, column], after[:, column] = low, high
        p_low = bundle.model.predict_proba(before)[:, 1]
        p_high = bundle.model.predict_proba(after)[:, 1]
        raw_low, raw_high = np.asarray([low, high]) * scaler.scale_[column] + scaler.mean_[column]
        scenarios.append({
            "variable": variable,
            "q25_raw": float(raw_low), "q75_raw": float(raw_high),
            "pd_mean_q25": float(p_low.mean()), "pd_mean_q75": float(p_high.mean()),
            "mean_pd_change_pp": float(np.mean(p_high - p_low) * 100),
            "share_pd_increases": float(np.mean(p_high > p_low)),
            "logistic_coefficient_standardized": float(logistic.coef_[0, column]),
        })
    runs_path = ROOT / "reports/pd_evaluation/metricas_por_semilla.csv"
    runs = pd.read_csv(runs_path)
    if runs.seed.nunique() != 30 or len(runs) != 30:
        raise ValueError("El resumen final requiere las 30 corridas PD distintas.")
    metrics = {}
    for name in ("auc", "gini", "ks", "brier", "log_loss"):
        values = runs[f"test_{name}"]
        metrics[name] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)),
                         "min": float(values.min()), "max": float(values.max())}
    lgd_path = ROOT / "artifacts/models/lgd_benchmark_results.json"
    lgd = json.loads(lgd_path.read_text(encoding="utf-8"))
    return {
        "method": "Escenarios pareados de inferencia, P25 a P75; no causal ni prueba de monotonicidad global.",
        "sample": {"split": "validation 2013", "n": 5000, "seed": 42,
                   "indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()},
        "sources": {p.relative_to(ROOT).as_posix(): artifact_hash(p) for p in (manifest, matrix_path, runs_path, lgd_path, ROOT / "reports/pd_evaluation/resultados.json", ROOT / "reports/pd_hyperparameter_search/resumen_optimizacion.json", ROOT / "artifacts/models/pd_logistic.joblib")},
        "pd_test_30_seeds": metrics,
        "pd_fit_seconds_total": float(runs.fit_seconds.sum()),
        "hypothesis_scenarios": scenarios,
        "lgd_persisted_seed_count": lgd["mlp_30_semillas"]["n_semillas"],
    }


if __name__ == "__main__":
    target = ROOT / "reports/technical-report/evidence.json"
    target.write_text(json.dumps(collect(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(target)
