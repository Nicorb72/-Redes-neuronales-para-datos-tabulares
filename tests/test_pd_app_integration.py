"""Regresión con el checkpoint publicado y los datos reales de validación."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.metrics import brier_score_loss, roc_auc_score
from streamlit.testing.v1 import AppTest

from credit_risk.app.contracts import ArtifactError, load_bundle, sha256
from credit_risk.app.service import assess, simulate
from credit_risk.app.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]


def test_final_model_reproduces_published_validation():
    _, _, manifest = load_settings()
    bundle = load_bundle(manifest)
    x = sparse.load_npz(ROOT / "data/processed/X_validation_pd.npz")
    y = np.load(ROOT / "data/processed/y_validation_pd.npy", allow_pickle=False)
    p = bundle.model.predict_proba(x)[:, 1]
    report = json.loads((ROOT / "reports/pd_evaluation/resultados.json").read_text(encoding="utf-8"))
    expected = report["mlp_seed_base"]["validation"]
    assert bundle.metadata["model_format"] == "pd_mlp"
    assert bundle.model.seed == 202612
    assert bundle.model.n_features_in_ == 44
    assert len(bundle.reference_pd) == report["rows"]["train"]
    assert np.isfinite(p).all()
    assert roc_auc_score(y, p) == pytest.approx(expected["auc"], abs=2e-6)
    assert brier_score_loss(y, p) == pytest.approx(expected["brier"], abs=2e-6)


def test_final_model_form_and_scenario():
    from credit_risk.app.contracts import COUNTS, FEATURES

    _, card, manifest = load_settings()
    bundle = load_bundle(manifest)
    stats = bundle.preprocessor.named_transformers_["numericas"].named_steps["imputacion"].statistics_
    row = {name: round(float(value)) if name in COUNTS else float(value)
           for name, value in zip(FEATURES[:11], stats)}
    row.update({name: choices[0] for name, choices in bundle.categories.items()})
    frame = pd.DataFrame([row])[FEATURES]
    result = assess(bundle, card, frame)
    assert 0 <= result.pd <= 1
    assert np.isfinite(result.score)
    scores = card.score(bundle.reference_pd)
    assert result.percentile == pytest.approx(100 * np.mean(scores <= result.score))
    before, after = simulate(bundle, card, frame, {"annual_inc": row["annual_inc"] + 10000})
    assert before == result
    assert after.pd != before.pd
    assert np.isfinite(after.score)


@pytest.mark.parametrize("mutation", ["class", "dimension", "format"])
def test_bad_torch_checkpoints_are_rejected(tmp_path, mutation):
    import torch

    _, _, manifest = load_settings()
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    for entry in metadata["artifacts"].values():
        entry["path"] = str((manifest.parent / entry["path"]).resolve())
    checkpoint = torch.load(metadata["artifacts"]["model"]["path"], weights_only=True)
    if mutation == "class":
        checkpoint["positive_class"] = 0
    elif mutation == "dimension":
        checkpoint["input_dim"] = 999
    else:
        metadata["model_format"] = "unknown"
    target = tmp_path / "invalid.pt"
    torch.save(checkpoint, target)
    metadata["artifacts"]["model"] = dict(path=str(target), sha256=sha256(target))
    config = tmp_path / "invalid.json"
    config.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ArtifactError):
        load_bundle(config)


def test_web_uses_real_model():
    app = AppTest.from_string("from credit_risk.app.web import render\nrender()")
    app.run(timeout=30)
    assert not app.exception
    assert not app.warning
    links = app.get("link_button")
    assert [link.proto.label for link in links] == [
        "Leer el blog del proyecto",
        "Ver el video promocional",
    ]
    assert links[0].proto.url == (
        "https://nicorb72.github.io/-Redes-neuronales-para-datos-tabulares/reports/technical-report/"
    )
    assert links[1].proto.url == (
        "https://www.youtube.com/watch?v=TijgcPZNboU&feature=youtu.be"
    )
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert len(app.metric) == 3
    app.number_input[-1].set_value(1000.0)
    app.button[1].click().run(timeout=30)
    assert not app.exception
    assert len(app.metric) == 6


def test_cloud_entrypoint_uses_saved_model():
    app = AppTest.from_file(str(ROOT / "deploy/streamlit_app.py"))
    app.run(timeout=30)
    assert not app.exception
    assert not app.warning
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert len(app.metric) == 3
