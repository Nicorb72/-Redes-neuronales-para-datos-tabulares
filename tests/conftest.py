"""Artefactos sintéticos exclusivos de pruebas; nunca se usan en la app por defecto."""
import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from credit_risk.app.contracts import FEATURES, load_bundle, sha256
from credit_risk.data.preprocessing import construir_preprocesador


@pytest.fixture
def pd_artifacts(tmp_path):
    rng = np.random.default_rng(17)
    rows = 160
    raw = pd.DataFrame({
        "loan_amnt": rng.integers(1000, 20000, rows),
        "annual_inc": rng.integers(20000, 100000, rows),
        "dti": rng.uniform(1, 35, rows),
        "delinq_2yrs": rng.integers(0, 4, rows),
        "inq_last_6mths": rng.integers(0, 6, rows),
        "open_acc": rng.integers(1, 20, rows),
        "pub_rec": rng.integers(0, 3, rows),
        "revol_bal": rng.integers(0, 20000, rows),
        "revol_util": rng.uniform(1, 100, rows),
        "total_acc": rng.integers(20, 60, rows),
        "antiguedad_crediticia_meses": rng.integers(12, 300, rows),
        "term": rng.choice([" 36 months", " 60 months"], rows),
        "purpose": rng.choice(["car", "debt_consolidation"], rows),
        "emp_length": rng.choice(["2 years", "5 years"], rows),
        "home_ownership": rng.choice(["OWN", "RENT"], rows),
    })[FEATURES]
    preprocessor = construir_preprocesador()
    matrix = sparse.csr_matrix(preprocessor.fit_transform(raw))
    labels = (raw.loan_amnt > 11000).astype(int).to_numpy()
    model = LogisticRegression(C=10, random_state=17).fit(matrix, labels)
    columns = list(preprocessor.get_feature_names_out())
    joblib.dump(model, tmp_path / "model.joblib")
    joblib.dump(preprocessor, tmp_path / "preprocessor.joblib")
    (tmp_path / "columns.json").write_text(json.dumps(columns), encoding="utf-8")
    sparse.save_npz(tmp_path / "reference.npz", matrix)
    metadata = dict(schema_version=1, positive_class=1, model_id="SYNTHETIC-TEST-ONLY",
                    raw_features=FEATURES, training_data="synthetic test fixture; seed 17",
                    reference_population="synthetic test rows", reference_split="train",
                    input_format="sparse", artifacts={})
    for key, file in dict(model="model.joblib", preprocessor="preprocessor.joblib",
                          columns="columns.json", reference_matrix="reference.npz").items():
        metadata["artifacts"][key] = dict(path=file, sha256=sha256(tmp_path / file))
    manifest = tmp_path / "pd_bundle.json"
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    return SimpleNamespace(bundle=load_bundle(manifest), manifest=manifest, metadata=metadata,
                           matrix=matrix, raw=raw, labels=labels, tmp_path=tmp_path)
