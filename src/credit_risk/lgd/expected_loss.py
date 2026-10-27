"""Cálculo y análisis de la Pérdida Esperada (Expected Loss - EL).

Fórmula regulatoria y financiera estándar (Basilea II / III, IFRS 9)::

    E[L] = PD * LGD * EAD

Donde:
1. PD (Probability of Default): Probabilidad de que el prestatario incurra en impago
   (obtenida del modelo de clasificación binaria).
2. LGD (Loss Given Default): Proporción económica de la deuda que se pierde tras el proceso
   de cobranza y recuperación (obtenida del modelo de regresión).
3. EAD (Exposure at Default): Magnitud de la obligación pendiente en el momento del incumplimiento.

Justificación de la definición de EAD:
- Momento de Originación (Underwriting / Otorgamiento):
  Al evaluar una solicitud inicial, el momento exacto de un hipotético impago futuro se desconoce.
  Para créditos amortizables a cuota fija (como los de Lending Club), la exposición máxima comprometida
  por la institución financiera es el monto total financiado: ``EAD_orig = funded_amnt`` (o ``loan_amnt``).
  Esta definición representa la exposición conservadora de riesgo ex-ante.

- Momento de Impago Observado (Ex-post):
  Si el crédito ya ha amortizado parte del principal antes del castigo, la exposición real no recuperada
  es el saldo de principal insoluto::
      EAD_obs = funded_amnt - total_rec_prncp

Este módulo proporciona el cálculo de pérdida esperada a nivel individual y agregado para la cartera.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class ExpectedLossSummary:
    n_loans: int
    total_exposure: float
    total_expected_loss: float
    mean_expected_loss_pct: float
    total_actual_loss: float = None
    mae_loss: float = None


def calcular_perdida_esperada(
    pd_probs: np.ndarray,
    lgd_preds: np.ndarray,
    ead_amounts: np.ndarray,
) -> np.ndarray:
    """Calcula la pérdida esperada individual: E[L] = PD * LGD * EAD.

    Garantiza que PD y LGD estén en [0, 1] y EAD >= 0.
    """
    pd_clean = np.clip(np.asarray(pd_probs, dtype=float), 0.0, 1.0)
    lgd_clean = np.clip(np.asarray(lgd_preds, dtype=float), 0.0, 1.0)
    ead_clean = np.maximum(np.asarray(ead_amounts, dtype=float), 0.0)

    return pd_clean * lgd_clean * ead_clean


def resumir_perdida_esperada(
    el_array: np.ndarray,
    ead_array: np.ndarray,
    actual_loss: np.ndarray = None,
) -> dict:
    """Genera métricas consolidadas de riesgo de crédito para una cartera."""
    total_exp = float(np.sum(ead_array))
    total_el = float(np.sum(el_array))
    el_pct = (total_el / total_exp * 100.0) if total_exp > 0 else 0.0

    res = {
        "n_creditos": int(len(el_array)),
        "exposicion_total_usd": total_exp,
        "perdida_esperada_total_usd": total_el,
        "tasa_perdida_esperada_pct": el_pct,
        "perdida_esperada_media_usd": float(np.mean(el_array)),
        "perdida_esperada_mediana_usd": float(np.median(el_array)),
    }

    if actual_loss is not None:
        act_clean = np.asarray(actual_loss, dtype=float)
        total_act = float(np.sum(act_clean))
        res["perdida_real_total_usd"] = total_act
        res["tasa_perdida_real_pct"] = (total_act / total_exp * 100.0) if total_exp > 0 else 0.0
        res["mae_perdida_usd"] = float(np.mean(np.abs(el_array - act_clean)))

    return res
