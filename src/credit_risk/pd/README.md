# Modelos PD

El código se organiza siguiendo el patrón de LGD:

- `baseline.py`: regresión logística de referencia y serialización.
- `neural_network.py`: arquitectura MLP, entrenamiento con parada temprana,
  predicción y guardado/carga del checkpoint.
- `evaluation.py`: métricas PD, calibración y gráficos.
- `experiments.py`: pipeline completo; carga la configuración, prepara los
  splits temporales, entrena ambos modelos, ejecuta las semillas y genera los
  resultados.
- `train_and_evaluate.py`: alias compatible con el punto de entrada anterior.
  La implementación principal está en `experiments.py`.

La preparación de datos se reutiliza desde
`credit_risk.data.prepare_data.preparar_matrices_pd`; no se duplica la lógica
de etiquetas, particiones ni preprocesamiento.

Desde la raíz del repositorio, ejecutar:

```bash
PYTHONPATH=src python -m credit_risk.pd.experiments
```

Los hiperparámetros están en `configs/pd_mlp_best.yaml`; el número de semillas
se lee de `configs/default.yaml`. Los resultados se guardan en
`reports/pd_evaluation/` y los modelos junto al preprocesador en
`artifacts/models/`.
