"""Arranque local: python run_app.py [--config ...] [--check]."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--check", action="store_true", help="Valida artefactos sin abrir la web")
    args = parser.parse_args()
    if args.check:
        from credit_risk.app.contracts import load_bundle
        from credit_risk.app.settings import load_settings
        try:
            _, _, manifest = load_settings(args.config)
            bundle = load_bundle(manifest)
            print(f"PD listo: {bundle.metadata['model_id']}; referencia: {len(bundle.reference_pd)} filas")
            return 0
        except (ValueError, OSError) as exc:
            print(f"Integración pendiente: {exc}")
            return 2
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.call([
        sys.executable, "-m", "streamlit", "run",
        str(ROOT / "src/credit_risk/app/web.py"),
        "--server.address", "127.0.0.1", "--server.port", str(args.port),
        "--browser.gatherUsageStats", "false", "--", "--config", str(Path(args.config).resolve()),
    ], env=env, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
