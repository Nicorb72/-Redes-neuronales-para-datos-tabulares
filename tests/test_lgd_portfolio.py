import hashlib

import numpy as np
import pandas as pd

from credit_risk.app.contracts import FEATURES
from credit_risk.data.preprocessing import VARIABLES_CATEGORICAS, VARIABLES_NUMERICAS
from credit_risk.lgd import portfolio


def test_estimar_perdida_esperada_alinea_modelos_y_agregados(monkeypatch, tmp_path):
    ids = ["loan-c", "loan-a", "loan-b"]
    funded = np.array([3000.0, 1000.0, 2000.0])
    cohort = pd.DataFrame({
        "id": ids,
        "issue_year": [2014, 2014, 2014],
        "loan_status": ["Charged Off", "Fully Paid", "Current"],
        "funded_amnt": funded,
        "total_rec_prncp": [500.0, 1000.0, 0.0],
        "recoveries": [400.0, 0.0, 0.0],
        "collection_recovery_fee": [100.0, 0.0, 0.0],
    })
    for name in VARIABLES_NUMERICAS:
        if name not in cohort:
            cohort[name] = 100.0
    cohort["loan_amnt"] = funded
    for name in VARIABLES_CATEGORICAS:
        cohort[name] = "fixture"

    class Preprocessor:
        feature_names_in_ = np.asarray(FEATURES)

        def transform(self, frame):
            assert list(frame.columns) == FEATURES
            return frame["loan_amnt"].to_numpy(dtype=float).reshape(-1, 1)

    class PDModel:
        pass

    bundle = type("Bundle", (), {
        "preprocessor": Preprocessor(),
        "metadata": {"input_format": "dense", "model_id": "test-pd"},
        "model": PDModel(),
    })()

    class LGDModel:
        input_dim = 1
        seed = 1212

        def predict(self, matrix):
            return 0.4 + matrix[:, 0] / 10000.0

    source = tmp_path / "data" / "raw" / "loan" / "loan.csv"
    source.parent.mkdir(parents=True)
    source.write_text("test source", encoding="utf-8")
    model_path = tmp_path / "artifacts" / "models" / "mlp_lgd.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"synthetic model")

    monkeypatch.setattr(portfolio, "obtener_raiz_proyecto", lambda: tmp_path)
    monkeypatch.setattr(portfolio, "preparar_cohorte_originacion", lambda year: cohort.copy())
    monkeypatch.setattr(portfolio, "load_bundle", lambda path: bundle)
    monkeypatch.setattr(portfolio.joblib, "load", lambda path: Preprocessor())
    monkeypatch.setattr(
        portfolio,
        "predict_pd",
        lambda model, matrix, input_format: matrix[:, 0] / 10000.0,
    )
    monkeypatch.setattr(
        portfolio.LGDNeuralNetwork,
        "load",
        lambda path, device: LGDModel(),
    )

    predictions, summary = portfolio.estimar_perdida_esperada(2014)

    assert predictions["loan_id"].tolist() == ids
    np.testing.assert_allclose(predictions["pd_predicha"], [0.3, 0.1, 0.2])
    np.testing.assert_allclose(predictions["lgd_predicha_condicional_a_default"], [0.7, 0.5, 0.6])
    np.testing.assert_allclose(predictions["ead_originacion_usd"], funded)
    np.testing.assert_allclose(predictions["perdida_esperada_pd_x_lgd_x_ead_usd"], [630.0, 50.0, 240.0])
    assert predictions["perdida_realizada_usd_si_resuelta"].iloc[0] == 2200.0
    assert predictions["perdida_realizada_usd_si_resuelta"].iloc[1] == 0.0
    assert np.isnan(predictions["perdida_realizada_usd_si_resuelta"].iloc[2])
    assert summary["n_prestamos_puntuados"] == 3
    assert summary["n_prestamos_con_resultado_resuelto"] == 2
    assert summary["ead_originacion_total_usd"] == 6000.0
    assert summary["perdida_esperada_total_cohorte_usd"] == 920.0
    assert summary["perdida_esperada_cohorte_resuelta_usd"] == 680.0
    assert summary["perdida_realizada_cohorte_resuelta_usd"] == 2200.0
    assert summary["modelo_lgd_seed"] == 1212
    assert summary["modelo_lgd_checkpoint_sha256"] == hashlib.sha256(b"synthetic model").hexdigest()
