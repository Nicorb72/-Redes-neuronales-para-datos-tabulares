"""Integra predicciones PD y LGD para estimar pérdida esperada ex ante.

Ejecutar desde la raíz, con ``PYTHONPATH=src`` y ``loan.csv`` disponible.
Las matrices procesadas PD/LGD no guardan una clave común por préstamo; por eso
este flujo vuelve a leer la fuente y transforma una sola cohorte alineada con
los preprocesadores ya ajustados.
"""
import argparse
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from credit_risk.app.contracts import FEATURES, load_bundle, predict_pd
from credit_risk.data.load_data import cargar_datos, obtener_raiz_proyecto
from credit_risk.data.preprocessing import (
    BUENOS,
    VARIABLES_CATEGORICAS,
    VARIABLES_NUMERICAS,
    crear_antiguedad_crediticia,
    preparar_fechas,
)
from credit_risk.lgd.neural_network import LGDNeuralNetwork
from credit_risk.lgd.target import ESTADOS_CASTIGADOS, calcular_lgd
from credit_risk.lgd.expected_loss import calcular_perdida_esperada


def preparar_cohorte_originacion(year: int) -> pd.DataFrame:
    """Carga y transforma préstamos originados en el año solicitado."""
    # antiguedad_crediticia_meses es una variable derivada y no existe en el
    # CSV fuente; se calcula después de cargar earliest_cr_line e issue_d.
    columnas_derivadas = {"antiguedad_crediticia_meses"}
    columnas = list(dict.fromkeys([
        "id", "loan_status", "issue_d", "earliest_cr_line", "funded_amnt",
        "total_rec_prncp", "recoveries", "collection_recovery_fee",
        *(name for name in VARIABLES_NUMERICAS if name not in columnas_derivadas),
        *VARIABLES_CATEGORICAS,
    ]))
    frame = preparar_fechas(cargar_datos(columnas=columnas))
    frame = crear_antiguedad_crediticia(frame)
    frame = frame.loc[frame["issue_year"] == year].copy()
    if frame.empty:
        raise ValueError(f"No hay préstamos de la cohorte de originación {year}.")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("La cohorte debe tener identificadores de préstamo únicos y no nulos.")
    for name in VARIABLES_NUMERICAS:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    for name in VARIABLES_CATEGORICAS:
        frame[name] = frame[name].astype("object")
    frame["funded_amnt"] = pd.to_numeric(frame["funded_amnt"], errors="coerce")
    frame = frame.loc[np.isfinite(frame["funded_amnt"]) & (frame["funded_amnt"] > 0)].copy()
    return frame.reset_index(drop=True)


def estimar_perdida_esperada(year: int = 2014) -> tuple[pd.DataFrame, dict]:
    """Predice PD y LGD sobre las mismas filas y calcula PD × LGD × EAD₀."""
    root = obtener_raiz_proyecto()
    source_path = root / "data" / "raw" / "loan" / "loan.csv"
    cohort = preparar_cohorte_originacion(year)
    raw_features = cohort.loc[:, FEATURES]

    pd_bundle = load_bundle(root / "artifacts" / "models" / "pd_bundle.json")
    pd_matrix = pd_bundle.preprocessor.transform(raw_features)
    pd_probability = predict_pd(pd_bundle.model, pd_matrix, pd_bundle.metadata["input_format"])

    lgd_preprocessor = joblib.load(root / "artifacts" / "models" / "preprocesador_lgd.joblib")
    if list(lgd_preprocessor.feature_names_in_) != FEATURES:
        raise ValueError("El preprocesador LGD no comparte el contrato de variables PD.")
    lgd_matrix = lgd_preprocessor.transform(raw_features)
    lgd_model_path = root / "artifacts" / "models" / "mlp_lgd.pt"
    lgd_model = LGDNeuralNetwork.load(lgd_model_path, device="cpu")
    if lgd_matrix.shape[1] != lgd_model.input_dim:
        raise ValueError("La dimensión de entrada no coincide con el modelo LGD guardado.")
    lgd_prediction = lgd_model.predict(lgd_matrix)

    ead_at_origination = cohort["funded_amnt"].to_numpy(dtype=float)
    expected_loss = calcular_perdida_esperada(pd_probability, lgd_prediction, ead_at_origination)
    status = cohort["loan_status"].astype("string").str.strip()
    settled_good = status.isin(BUENOS).to_numpy()
    charged_off = status.isin(ESTADOS_CASTIGADOS).to_numpy()
    resolved = settled_good | charged_off
    realized_loss = np.full(len(cohort), np.nan, dtype=float)
    realized_loss[settled_good] = 0.0
    if charged_off.any():
        realized = calcular_lgd(cohort.loc[charged_off])
        realized_loss[charged_off] = (
            realized["ead_obs"].to_numpy(dtype=float)
            * realized["lgd"].to_numpy(dtype=float)
        )
    with source_path.open("rb") as source_file:
        source_sha256 = hashlib.file_digest(source_file, "sha256").hexdigest()
    with lgd_model_path.open("rb") as model_file:
        lgd_model_sha256 = hashlib.file_digest(model_file, "sha256").hexdigest()

    predictions = pd.DataFrame({
        "loan_id": cohort["id"].astype(str),
        "issue_year": cohort["issue_year"].astype(int),
        "loan_status_snapshot": status,
        "pd_predicha": pd_probability,
        "lgd_predicha_condicional_a_default": lgd_prediction,
        "ead_originacion_usd": ead_at_origination,
        "perdida_esperada_pd_x_lgd_x_ead_usd": expected_loss,
        "perdida_realizada_usd_si_resuelta": realized_loss,
    })
    resolved_expected = expected_loss[resolved]
    resolved_realized = realized_loss[resolved]
    summary = {
        "cohorte_originacion": year,
        "fuente_csv": str(source_path.relative_to(root)),
        "fuente_csv_sha256": source_sha256,
        "n_prestamos_puntuados": int(len(cohort)),
        "n_prestamos_con_resultado_resuelto": int(resolved.sum()),
        "cobertura_resultados_resueltos": float(resolved.mean()),
        "pd_media_predicha": float(np.mean(pd_probability)),
        "lgd_media_predicha": float(np.mean(lgd_prediction)),
        "ead_originacion_total_usd": float(np.sum(ead_at_origination)),
        "perdida_esperada_total_cohorte_usd": float(np.sum(expected_loss)),
        "perdida_esperada_cohorte_resuelta_usd": float(np.sum(resolved_expected)),
        "perdida_realizada_cohorte_resuelta_usd": float(np.nansum(resolved_realized)),
        "ead_ex_ante": "funded_amnt al momento de originación",
        "ead_realizada": "principal pendiente al castigo para Charged Off; cero para Fully Paid",
        "nota_comparacion": (
            "La pérdida esperada usa EAD de originación; la pérdida observada usa EAD al castigo. "
            "La diferencia de momento de exposición debe considerarse al interpretar la comparación."
        ),
        "nota_horizonte_pd": (
            "El objetivo PD se deriva de estados observados en el archivo y no de un horizonte fijo. "
            "La estimación conjunta es exploratoria; no constituye una provisión regulatoria."
        ),
        "modelo_pd": pd_bundle.metadata["model_id"],
        "modelo_lgd": str(lgd_model_path.relative_to(root)),
        "modelo_lgd_seed": int(lgd_model.seed),
        "modelo_lgd_checkpoint_sha256": lgd_model_sha256,
    }
    return predictions, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2014, help="Año de originación de la cohorte (default: 2014).")
    parser.add_argument("--output-dir", default="reports/lgd_evaluation")
    args = parser.parse_args()
    root = obtener_raiz_proyecto()
    predictions, summary = estimar_perdida_esperada(args.year)
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / f"perdida_esperada_pd_lgd_{args.year}.csv", index=False)
    (output_dir / f"perdida_esperada_pd_lgd_{args.year}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
