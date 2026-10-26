"""Resultados y escenarios calculados siempre con el mismo paquete y escala."""
from dataclasses import dataclass

from credit_risk.scorecard import percentile


@dataclass(frozen=True)
class Assessment:
    pd: float
    score: float
    percentile: float


def assess(bundle, card, frame):
    if len(frame) != 1:
        raise ValueError("La consulta debe contener una sola persona.")
    pd = float(bundle.predict(frame)[0])
    score = float(card.score(pd))
    reference = card.score(bundle.reference_pd)
    return Assessment(pd, score, float(percentile(score, reference)))


def simulate(bundle, card, frame, changes):
    if not changes or not set(changes).issubset(frame.columns):
        raise ValueError("Seleccione variables PD válidas para el escenario.")
    updated = frame.copy(deep=True)
    for key, value in changes.items():
        updated[key] = value
    return assess(bundle, card, frame), assess(bundle, card, updated)
