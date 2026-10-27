"""Modelo de referencia de regresión logística para PD."""

from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


class BaselinePD:
    """Regresión logística con probabilidades para la clase de incumplimiento."""

    def __init__(self, C: float = 1.0, solver: str = "lbfgs", max_iter: int = 2000,
                 class_weight=None):
        self.model = LogisticRegression(
            C=float(C),
            solver=solver,
            max_iter=int(max_iter),
            class_weight=class_weight,
        )
        self.fit_time_sec = 0.0

    @property
    def n_parameters(self) -> int:
        if not hasattr(self.model, "coef_"):
            return 0
        return int(self.model.coef_.size + self.model.intercept_.size)

    def fit(self, X, y):
        """Ajusta el baseline; el llamador define qué partición se utiliza."""
        started = time.perf_counter()
        self.model.fit(X, y)
        self.fit_time_sec = time.perf_counter() - started
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Devuelve probabilidades en columnas ordenadas como clase 0 y clase 1."""
        if not hasattr(self.model, "classes_"):
            raise RuntimeError("El modelo logístico debe entrenarse antes de predecir.")
        classes = np.asarray(self.model.classes_)
        if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
            raise ValueError("El modelo debe contener las clases 0 y 1.")
        probabilities = np.asarray(self.model.predict_proba(X), dtype=float)
        return probabilities[:, [int(np.flatnonzero(classes == label)[0]) for label in (0, 1)]]

    def predict_pd(self, X) -> np.ndarray:
        """Devuelve únicamente P(default=1)."""
        return self.predict_proba(X)[:, 1]

    def evaluate(self, X, y) -> dict[str, float]:
        """Evalúa las probabilidades con las métricas oficiales de PD."""
        from credit_risk.pd.evaluation import calcular_metricas_pd

        return calcular_metricas_pd(y, self.predict_pd(X))

    def save(self, filepath: str | Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # Se conserva el formato compatible con el artefacto sklearn anterior.
        joblib.dump(self.model, filepath)

    @classmethod
    def load(cls, filepath: str | Path):
        estimator = joblib.load(filepath)
        if not isinstance(estimator, LogisticRegression):
            raise TypeError("El artefacto no contiene una LogisticRegression.")
        instance = cls(
            C=estimator.C,
            solver=estimator.solver,
            max_iter=estimator.max_iter,
            class_weight=estimator.class_weight,
        )
        instance.model = estimator
        return instance
