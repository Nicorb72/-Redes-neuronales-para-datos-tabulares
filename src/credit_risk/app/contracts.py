"""Contrato de inferencia: artefactos locales confiables, sin fit ni descargas."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from credit_risk.data.preprocessing import VARIABLES_CATEGORICAS, VARIABLES_NUMERICAS
from credit_risk.scorecard.scaling import probabilities

FEATURES = VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS
COUNTS = {"delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "total_acc",
          "antiguedad_crediticia_meses"}


class ArtifactError(ValueError):
    """Artefactos ausentes, incompatibles o dañados."""


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bundle_fingerprint(manifest_path):
    """Huella de contenido para invalidar caché incluso si cambian archivos in situ."""
    path = Path(manifest_path).resolve()
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        digests = [sha256(path)]
        for key in ("model", "preprocessor", "columns", "reference_matrix"):
            artifact = path.parent / metadata["artifacts"][key]["path"]
            digests.append(sha256(artifact))
        return tuple(digests)
    except Exception as exc:
        raise ArtifactError(f"No se pudo verificar el paquete PD: {exc}") from exc


def predict_pd(model, matrix, input_format="sparse"):
    """Resuelve explícitamente la clase 1 incluso si classes_ está invertida."""
    classes = np.asarray(getattr(model, "classes_", []))
    if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
        raise ArtifactError("El modelo debe declarar classes_ = [0, 1] (en cualquier orden).")
    if input_format not in {"sparse", "dense"}:
        raise ArtifactError("input_format debe ser sparse o dense.")
    if input_format == "dense" and sparse.issparse(matrix):
        matrix = matrix.toarray()
    output = np.asarray(model.predict_proba(matrix), dtype=float)
    if output.shape != (matrix.shape[0], 2):
        raise ArtifactError("predict_proba debe retornar una matriz n x 2.")
    probabilities(output)
    if not np.allclose(output.sum(axis=1), 1, atol=1e-7, rtol=0):
        raise ArtifactError("Las probabilidades por fila no suman 1.")
    return output[:, int(np.flatnonzero(classes == 1)[0])]


@dataclass
class PDBundle:
    model: object
    preprocessor: object
    reference_pd: np.ndarray
    metadata: dict
    columns: list

    @property
    def categories(self):
        encoder = self.preprocessor.named_transformers_["categoricas"].named_steps["codificacion"]
        return {key: [value for value in values if isinstance(value, str)]
                for key, values in zip(VARIABLES_CATEGORICAS, encoder.categories_)}

    def validate(self, frame):
        if set(frame.columns) != set(FEATURES) or frame.columns.duplicated().any() or frame.empty:
            raise ValueError("Se requieren exactamente las 15 variables PD y al menos una fila.")
        result = frame[FEATURES].copy()
        for name in VARIABLES_NUMERICAS:
            values = pd.to_numeric(result[name], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"{name}: ingrese un valor finito no negativo.")
            if name in COUNTS and (values != np.floor(values)).any():
                raise ValueError(f"{name}: se requiere un número entero.")
            result[name] = values
        for name, choices in self.categories.items():
            if not result[name].isin(choices).all():
                raise ValueError(f"{name}: categoría no observada durante el entrenamiento.")
        return result

    def predict(self, frame):
        matrix = self.preprocessor.transform(self.validate(frame))
        if matrix.shape[1] != len(self.columns):
            raise ArtifactError("Dimensión transformada incompatible.")
        return predict_pd(self.model, matrix, self.metadata["input_format"])


def load_bundle(manifest_path):
    """No cargar joblib de terceros: su deserialización ejecuta código Python."""
    path = Path(manifest_path).resolve()
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata["schema_version"] != 1 or metadata["positive_class"] != 1:
            raise ArtifactError("Versión de contrato o clase positiva incompatible.")
        for key in ("model_id", "training_data", "reference_population", "reference_split"):
            if not isinstance(metadata[key], str) or not metadata[key].strip():
                raise ArtifactError(f"Falta procedencia: {key}.")
        if metadata["reference_split"] != "train":
            raise ArtifactError("La referencia del percentil debe ser train, sin reutilizar test.")
        if metadata["raw_features"] != FEATURES:
            raise ArtifactError("Las variables de entrada difieren del contrato de datos PD.")
        paths = {}
        for key in ("model", "preprocessor", "columns", "reference_matrix"):
            entry = metadata["artifacts"][key]
            artifact = (path.parent / entry["path"]).resolve()
            if sha256(artifact) != entry["sha256"]:
                raise ArtifactError(f"Hash incompatible: {key}.")
            paths[key] = artifact
        preprocessor = joblib.load(paths["preprocessor"])
        columns = json.loads(paths["columns"].read_text(encoding="utf-8"))
        if columns != list(preprocessor.get_feature_names_out()):
            raise ArtifactError("columnas_pd.json no coincide con el preprocesador ajustado.")
        if list(preprocessor.feature_names_in_) != FEATURES:
            raise ArtifactError("El preprocesador no usa las variables PD esperadas en orden.")
        model = joblib.load(paths["model"])
        if getattr(model, "n_features_in_", None) != len(columns):
            raise ArtifactError("El modelo no declara la dimensión transformada esperada.")
        reference = sparse.load_npz(paths["reference_matrix"])
        if reference.shape[0] == 0 or reference.shape[1] != len(columns):
            raise ArtifactError("Matriz de referencia vacía o incompatible.")
        if not np.isfinite(reference.data).all():
            raise ArtifactError("La matriz de referencia contiene valores no finitos.")
        # Lotes acotados también permiten modelos que requieren entrada densa.
        pd_reference = np.concatenate([
            predict_pd(model, reference[i:i + 2048], metadata["input_format"])
            for i in range(0, reference.shape[0], 2048)
        ])
        bundle = PDBundle(model, preprocessor, pd_reference, metadata, columns)
        if any(not values for values in bundle.categories.values()):
            raise ArtifactError("No hay categorías válidas para el formulario.")
        return bundle
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError(f"No se pudo cargar el paquete PD: {exc}") from exc
