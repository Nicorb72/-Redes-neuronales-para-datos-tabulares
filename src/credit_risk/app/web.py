"""Interfaz Streamlit para el paquete PD versionado."""
import argparse

import pandas as pd
import streamlit as st

from credit_risk.app.contracts import ArtifactError, FEATURES, COUNTS, bundle_fingerprint, load_bundle
from credit_risk.app.service import assess, simulate
from credit_risk.app.settings import load_settings
from credit_risk.data.preprocessing import VARIABLES_NUMERICAS

LABELS = {
    "loan_amnt": "Monto del préstamo (USD)", "annual_inc": "Ingreso anual (USD)",
    "dti": "Relación deuda / ingreso (%, dti)",
    "delinq_2yrs": "Atrasos en los últimos 2 años (delinq_2yrs)",
    "inq_last_6mths": "Consultas de crédito en 6 meses (inq_last_6mths)",
    "open_acc": "Cuentas abiertas (open_acc)", "pub_rec": "Registros públicos (pub_rec)",
    "revol_bal": "Saldo rotativo (USD, revol_bal)",
    "revol_util": "Utilización rotativa (%, revol_util)",
    "total_acc": "Total de cuentas (total_acc)",
    "antiguedad_crediticia_meses": "Antigüedad crediticia al solicitar (meses)",
    "term": "Plazo del préstamo", "purpose": "Propósito del préstamo",
    "emp_length": "Antigüedad laboral", "home_ownership": "Situación de vivienda",
}


@st.cache_resource(show_spinner="Cargando modelo y población de referencia…")
def cached_bundle(path, fingerprint):
    # fingerprint forma parte de la clave de caché; se verifica en cada rerun.
    return load_bundle(path)


def input_value(name, bundle, key, initial=None):
    if name in VARIABLES_NUMERICAS:
        if initial is None:
            imputer = bundle.preprocessor.named_transformers_["numericas"].named_steps["imputacion"]
            initial = imputer.statistics_[VARIABLES_NUMERICAS.index(name)]
            if name in COUNTS:
                initial = round(float(initial))
        return st.number_input(LABELS[name], min_value=0.0, value=max(0.0, float(initial)),
                               step=1.0 if name in COUNTS else 0.1, key=key)
    choices = bundle.categories[name]
    index = choices.index(initial) if initial in choices else 0
    return st.selectbox(LABELS[name], choices, index=index, format_func=str.strip, key=key)


def show_result(result):
    left, middle, right = st.columns(3)
    left.metric("Probabilidad de incumplimiento", f"{result.pd:.2%}")
    middle.metric("Puntaje", f"{result.score:.1f}")
    right.metric("Percentil del puntaje", f"{result.percentile:.1f}")


def render(config_path=None):
    st.set_page_config(page_title="Tu perfil crediticio", page_icon="📊", layout="wide")
    st.title("Explora tu perfil crediticio")
    st.write("Consulta la estimación del modelo y explora cómo responde a cambios en tus datos.")
    st.caption("Proyecto académico con datos históricos de Lending Club. No constituye una aprobación de crédito.")
    try:
        config, card, manifest = load_settings(config_path)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        st.error(f"Configuración inválida: {exc}")
        return
    app = config.get("app", {})
    with st.sidebar:
        st.header("Acerca del proyecto")
        if app.get("report_url"):
            st.link_button("Leer el reporte técnico", app["report_url"])
        else:
            st.caption("Reporte técnico: enlace pendiente.")
        if app.get("video_url"):
            st.link_button("Ver el video", app["video_url"])
        else:
            st.caption("Video promocional: enlace pendiente.")
        st.write(f"Puntaje base: {card.base_score:g}")
        st.write(f"Odds buenas:malas: {card.base_good_to_bad_odds:g}:1")
        st.write(f"Puntos al duplicar las odds: {card.points_to_double_good_to_bad_odds:g}")
    try:
        # Se releen los hashes en cada ejecución para detectar cambios de artefactos.
        bundle = cached_bundle(str(manifest), bundle_fingerprint(manifest))
    except ArtifactError:
        st.warning("Integración pendiente: todavía no hay un paquete PD válido disponible.")
        st.info("El equipo debe entregar el modelo entrenado, su preprocesador y la población "
                "de referencia. Las estimaciones se habilitarán al validar esos archivos.")
        return
    st.caption(f"Modelo: {bundle.metadata['model_id']}")
    with st.expander("Cómo leer los resultados"):
        st.write("Una PD mayor indica mayor riesgo estimado. Un puntaje mayor indica menor riesgo. "
                 "El percentil es el porcentaje de la referencia con puntaje menor o igual al tuyo; "
                 "no es la probabilidad de aprobación ni una comparación con toda la población actual.")
        st.write(f"Referencia: {bundle.metadata['reference_population']} ({len(bundle.reference_pd)} registros).")
        st.write("Los valores iniciales son medianas (redondeadas para conteos) y categorías de entrenamiento: revisa todos los campos. "
                 "Las categorías conservan la codificación original del conjunto. "
                 "Los resultados pueden ser poco fiables para perfiles alejados de esos datos.")
    with st.form("perfil"):
        st.subheader("1. Tus características al solicitar el préstamo")
        columns = st.columns(2)
        values = {}
        for i, name in enumerate(FEATURES):
            with columns[i % 2]:
                values[name] = input_value(name, bundle, f"base_{name}")
        submitted = st.form_submit_button("Calcular mi perfil")
    if submitted:
        frame = pd.DataFrame([values])
        try:
            assess(bundle, card, frame)
            st.session_state["profile"] = values
        except (ValueError, TypeError) as exc:
            st.error(str(exc))
            st.session_state.pop("profile", None)
    if "profile" not in st.session_state:
        return
    frame = pd.DataFrame([st.session_state["profile"]])
    try:
        result = assess(bundle, card, frame)
        st.subheader("2. Tu resultado")
        show_result(result)
        st.subheader("3. ¿Qué pasaría si cambias una característica?")
        st.caption("Se compara con tu última consulta calculada. Cambia una variable a la vez "
                   "para observar la dirección de la respuesta del modelo. No predice un efecto causal.")
        feature = st.selectbox("Característica a explorar", FEATURES, format_func=LABELS.get)
        value = input_value(feature, bundle, f"scenario_{feature}", frame.iloc[0][feature])
        if st.button("Comparar escenario"):
            before, after = simulate(bundle, card, frame, {feature: value})
            show_result(after)
            st.write(f"Cambio de PD: {(after.pd - before.pd) * 100:+.2f} puntos porcentuales. "
                     f"Cambio de puntaje: {after.score - before.score:+.1f} puntos. "
                     f"Cambio de percentil: {after.percentile - before.percentile:+.1f}.")
    except (ValueError, TypeError) as exc:
        st.error(f"No se pudo calcular el perfil: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    render(args.config)
