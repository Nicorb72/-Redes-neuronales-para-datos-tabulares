import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from streamlit.testing.v1 import AppTest

from credit_risk.app.contracts import ArtifactError, bundle_fingerprint, load_bundle, predict_pd
from credit_risk.app.service import assess, simulate
from credit_risk.scorecard import Scorecard

ROOT = Path(__file__).resolve().parents[1]


def test_saved_model_and_scenario(pd_artifacts):
    data = pd_artifacts
    row = data.raw.iloc[[0]]
    original = row.copy(deep=True)
    before, after = simulate(data.bundle, Scorecard(), row, {"loan_amnt": 1000})
    expected = data.bundle.model.predict_proba(data.bundle.preprocessor.transform(row))[0, 1]
    assert before.pd == pytest.approx(expected)
    assert before.score == pytest.approx(Scorecard().score(expected))
    assert 0 <= before.percentile <= 100
    assert after.pd < before.pd
    assert after.score > before.score
    assert row.equals(original)
    same, unchanged = simulate(data.bundle, Scorecard(), row, {"loan_amnt": row.loan_amnt.iloc[0]})
    assert same == unchanged
    with pytest.raises(ValueError):
        assess(data.bundle, Scorecard(), data.raw.iloc[:2])


@pytest.mark.parametrize("change", [{"annual_inc": np.nan}, {"loan_amnt": -1},
                                    {"term": "unknown"}, {"pub_rec": 0.2}])
def test_bad_user_inputs(pd_artifacts, change):
    with pytest.raises(ValueError):
        simulate(pd_artifacts.bundle, Scorecard(), pd_artifacts.raw.iloc[[0]], change)


@pytest.mark.parametrize("mutation", ["hash", "columns", "dimension", "positive_class", "split", "missing"])
def test_bad_artifact_contract(pd_artifacts, mutation):
    data = pd_artifacts
    metadata = data.metadata
    if mutation == "hash":
        metadata["artifacts"]["model"]["sha256"] = "wrong"
    elif mutation == "columns":
        metadata["raw_features"] = []
    elif mutation == "dimension":
        import joblib
        from credit_risk.app.contracts import sha256
        model = data.bundle.model
        model.n_features_in_ = 999
        joblib.dump(model, data.tmp_path / "model.joblib")
        metadata["artifacts"]["model"]["sha256"] = sha256(data.tmp_path / "model.joblib")
    elif mutation == "positive_class":
        metadata["positive_class"] = 0
    elif mutation == "split":
        metadata["reference_split"] = "test"
    else:
        metadata["artifacts"]["model"]["path"] = "missing.joblib"
    data.manifest.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ArtifactError):
        load_bundle(data.manifest)


def test_positive_class_order_and_invalid_output():
    class Reversed:
        classes_ = np.array([1, 0])

        def predict_proba(self, matrix):
            return np.tile([0.8, 0.2], (len(matrix), 1))

    np.testing.assert_allclose(predict_pd(Reversed(), np.zeros((2, 3))), [0.8, 0.8])
    class Invalid(Reversed):
        def predict_proba(self, matrix):
            return np.array([[0.8, 0.8]])

    with pytest.raises(ArtifactError):
        predict_pd(Invalid(), np.zeros((1, 3)))


def missing_config(tmp_path):
    config = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    config["app"]["pd_bundle"] = str(tmp_path / "does-not-exist.json")
    path = tmp_path / "missing.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_app_missing_artifacts(tmp_path):
    path = missing_config(tmp_path)
    app = AppTest.from_string(f"from credit_risk.app.web import render\nrender({str(path)!r})")
    app.run(timeout=20)
    assert not app.exception
    assert "Integración pendiente" in app.warning[0].value
    assert len(app.metric) == 0
    assert len(app.number_input) == 0


def test_app_prediction_and_simulation(pd_artifacts):
    config = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    config["app"]["pd_bundle"] = str(pd_artifacts.manifest)
    path = pd_artifacts.tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    app = AppTest.from_string(f"from credit_risk.app.web import render\nrender({str(path)!r})")
    app.run(timeout=20)
    assert not app.exception
    assert len(app.number_input) == 11
    app.button[0].click().run(timeout=20)
    assert not app.exception
    assert len(app.metric) == 3
    # loan_amnt es la variable inicial del selector de escenario.
    app.number_input[-1].set_value(1000.0)
    app.button[1].click().run(timeout=20)
    assert not app.exception
    assert len(app.metric) == 6
    assert any("Cambio de PD" in item.value for item in app.markdown)


def test_single_command_check_reports_missing_model(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(ROOT / "run_app.py"), "--check", "--config", str(missing_config(tmp_path))],
        cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 2
    assert "Integración pendiente" in completed.stdout


def test_export_check_interpretation_commands(pd_artifacts):
    """Prueba completa con artefactos de prueba, sin sustituir los modelos reales."""
    from scipy import sparse

    data = pd_artifacts
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    exported = data.tmp_path / "exported.json"
    command = [sys.executable, "-m", "credit_risk.scorecard.export",
               "--model", str(data.tmp_path / "model.joblib"),
               "--preprocessor", str(data.tmp_path / "preprocessor.joblib"),
               "--columns", str(data.tmp_path / "columns.json"),
               "--reference-matrix", str(data.tmp_path / "reference.npz"),
               "--model-id", "SYNTHETIC-TEST-ONLY", "--training-data", "synthetic",
               "--reference-population", "synthetic", "--output", str(exported)]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    config = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    config["app"]["pd_bundle"] = str(exported)
    config_path = data.tmp_path / "valid.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = subprocess.run([sys.executable, str(ROOT / "run_app.py"), "--check",
                             "--config", str(config_path)], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    sparse.save_npz(data.tmp_path / "validation.npz", data.matrix)
    np.save(data.tmp_path / "labels.npy", data.labels)
    output = data.tmp_path / "importance.json"
    result = subprocess.run([
        sys.executable, "-m", "credit_risk.scorecard.interpretability", "--config", str(config_path),
        "--matrix", str(data.tmp_path / "validation.npz"), "--labels", str(data.tmp_path / "labels.npy"),
        "--repeats", "2", "--max-rows", "160", "--output", str(output),
    ], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert output.with_suffix(".svg").exists()
    assert output.with_suffix(".csv").exists()
    assert json.loads(output.read_text(encoding="utf-8"))["model_id"] == "SYNTHETIC-TEST-ONLY"
    # Un archivo de modelo alterado invalida la caché, aun sin cambiar el manifiesto.
    fingerprint = bundle_fingerprint(exported)
    with (data.tmp_path / "model.joblib").open("ab") as stream:
        stream.write(b"changed")
    assert fingerprint != bundle_fingerprint(exported)
