from pathlib import Path
from urllib.parse import urlparse

import yaml

from credit_risk.scorecard import Scorecard

ROOT = Path(__file__).resolve().parents[3]


def load_settings(path=None):
    path = Path(path or ROOT / "configs/default.yaml").resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    card = Scorecard(**config["scorecard"])
    app = config.get("app", {})
    # Rutas relativas a la raíz del repositorio, independientemente del cwd.
    bundle = Path(app.get("pd_bundle", "artifacts/models/pd_bundle.json"))
    if not bundle.is_absolute():
        bundle = ROOT / bundle
    for name in ("report_url", "video_url"):
        value = app.get(name)
        if value and (urlparse(value).scheme not in {"https", "http"} or not urlparse(value).netloc):
            raise ValueError(f"{name} debe ser una URL http(s) válida.")
    return config, card, bundle
