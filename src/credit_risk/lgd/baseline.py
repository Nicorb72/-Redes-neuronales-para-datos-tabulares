"""Modelo de referencia (baseline) para Pérdida Dado el Incumplimiento (LGD).

Utiliza una regresión lineal (mínimos cuadrados ordinarios o regularización Ridge)
como modelo de referencia de baja complejidad. Dado que la LGD es por definición
una proporción de pérdida en el intervalo [0, 1], las predicciones del modelo lineal
se proyectan/recortan al rango válido [0, 1].
"""
from dataclasses import dataclass
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class BaselineLGDMetrics:
    mae: float
    rmse: float
    train_time_sec: float
    n_parameters: int


class BaselineLGD:
    """Modelo de referencia de regresión lineal para LGD con proyección a [0, 1]."""

    def __init__(self, alpha=0.0):
        self.alpha = float(alpha)
        if self.alpha > 0:
            self.model = Ridge(alpha=self.alpha, random_state=42)
        else:
            self.model = LinearRegression()
        self.is_fitted = False

    @property
    def n_parameters(self):
        """Número de parámetros entrenables (pesos + sesgo)."""
        if not self.is_fitted:
            return 0
        return int(len(self.model.coef_) + 1)

    def fit(self, X, y):
        """Ajusta el modelo lineal con datos de entrenamiento."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """Predice la LGD y recorta estrictamente al intervalo [0, 1]."""
        if not self.is_fitted:
            raise RuntimeError("El modelo debe ser ajustado antes de predecir.")
        y_pred = self.model.predict(X)
        return np.clip(y_pred, 0.0, 1.0)

    def evaluate(self, X, y):
        """Calcula MAE y RMSE sobre un conjunto de evaluación."""
        preds = self.predict(X)
        mae = float(mean_absolute_error(y, preds))
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        return mae, rmse

    def save(self, filepath):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        return joblib.load(filepath)
