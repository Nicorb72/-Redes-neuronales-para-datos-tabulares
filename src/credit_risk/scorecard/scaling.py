"""Escala logarítmica con odds buenas:malas; PD es P(default_flag=1)."""
from dataclasses import dataclass

import numpy as np


def probabilities(values):
    p = np.asarray(values, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("PD debe ser finita y estar entre 0 y 1.")
    return p


@dataclass(frozen=True)
class Scorecard:
    base_score: float = 600
    base_good_to_bad_odds: float = 50
    points_to_double_good_to_bad_odds: float = 20
    epsilon: float = 1e-9

    def __post_init__(self):
        values = (self.base_score, self.base_good_to_bad_odds,
                  self.points_to_double_good_to_bad_odds, self.epsilon)
        if not np.isfinite(values).all():
            raise ValueError("Los parámetros de scorecard deben ser finitos.")
        if self.base_good_to_bad_odds <= 0 or self.points_to_double_good_to_bad_odds <= 0:
            raise ValueError("Odds y PDO deben ser positivos.")
        if not 0 < self.epsilon < 0.5:
            raise ValueError("epsilon debe estar entre 0 y 0.5.")

    @property
    def factor(self):
        return self.points_to_double_good_to_bad_odds / np.log(2)

    @property
    def offset(self):
        return self.base_score - self.factor * np.log(self.base_good_to_bad_odds)

    def score(self, pd):
        p = np.clip(probabilities(pd), self.epsilon, 1 - self.epsilon)
        return self.offset + self.factor * (np.log1p(-p) - np.log(p))

    def pd(self, score):
        s = np.asarray(score, dtype=float)
        if not np.isfinite(s).all():
            raise ValueError("El puntaje debe ser finito.")
        # exp(-logaddexp) evita overflow en la inversión logística.
        return np.exp(-np.logaddexp(0, (s - self.offset) / self.factor))


def percentile(score, reference_scores):
    """ECDF inclusiva: porcentaje de referencia con puntaje <= al consultado."""
    reference = np.asarray(reference_scores, dtype=float)
    if reference.ndim != 1 or reference.size == 0 or not np.isfinite(reference).all():
        raise ValueError("La población de referencia debe ser un vector finito no vacío.")
    s = np.asarray(score, dtype=float)
    if not np.isfinite(s).all():
        raise ValueError("El puntaje debe ser finito.")
    return 100 * np.searchsorted(np.sort(reference), s, side="right") / reference.size
