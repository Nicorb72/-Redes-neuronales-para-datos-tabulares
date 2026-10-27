"""Entrada para Streamlit Community Cloud desde la raíz del repositorio."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.app.web import render  # noqa: E402

render()
