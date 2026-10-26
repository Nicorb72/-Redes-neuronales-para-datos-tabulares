"""Empaqueta rutas y hashes de artefactos existentes; nunca entrena un modelo."""
import argparse
import json
import os
from pathlib import Path

from credit_risk.app.contracts import FEATURES, load_bundle, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--training-data", required=True, help="Versión/procedencia de train")
    parser.add_argument("--reference-population", required=True)
    parser.add_argument("--preprocessor", default="artifacts/models/preprocesador_pd.joblib")
    parser.add_argument("--columns", default="artifacts/models/columnas_pd.json")
    parser.add_argument("--reference-matrix", default="data/processed/X_train_pd.npz")
    parser.add_argument("--input-format", choices=["sparse", "dense"], default="sparse")
    parser.add_argument("--output", default="artifacts/models/pd_bundle.json")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    metadata = dict(schema_version=1, positive_class=1, raw_features=FEATURES,
                    model_id=args.model_id, training_data=args.training_data,
                    reference_population=args.reference_population, reference_split="train",
                    input_format=args.input_format, artifacts={})
    for key in ("model", "preprocessor", "columns", "reference_matrix"):
        source = Path(getattr(args, key)).resolve()
        metadata["artifacts"][key] = {
            "path": Path(os.path.relpath(source, output.parent)).as_posix(),
            "sha256": sha256(source),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix(".pending.json")
    try:
        pending.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        load_bundle(pending)
        pending.replace(output)
    finally:
        pending.unlink(missing_ok=True)
    print(f"Contrato PD validado: {output}")


if __name__ == "__main__":
    main()
