"""Auditoría de exclusión de variables posteriores al desembolso."""
from credit_risk.data.preprocessing import VARIABLES_NUMERICAS, VARIABLES_CATEGORICAS


PROHIBIDAS = {
    "loan_status", "default_flag", "recoveries", "collection_recovery_fee",
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "out_prncp", "out_prncp_inv", "last_pymnt_d",
    "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
}


def test_predictoras_sin_fuga_evidente():
    predictoras = VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS
    assert len(predictoras) == 15
    assert len(set(predictoras)) == 15
    assert not (set(predictoras) & PROHIBIDAS)
