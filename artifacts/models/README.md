# Modelos entrenados

Guarden aquí los modelos finales que necesita la aplicación y la generación de
figuras. Añadan junto a cada modelo su configuración y metadatos de
preprocesamiento para que pueda cargarse sin reentrenar.

## Modelos PD

El pipeline reproducible se ejecuta desde la raíz del repositorio con:

```bash
PYTHONPATH=src python -m credit_risk.pd.experiments
```

El proceso genera `pd_mlp_final.pt` (estado del MLP para la semilla base de la
configuración), `pd_logistic.joblib` (clasificador logístico de referencia),
`pd_preprocessor.joblib` (preprocesador ajustado solo con entrenamiento) y
`pd_feature_names.json` (orden de las variables procesadas). La configuración
operativa está en `configs/pd_mlp_best.yaml`; el número de semillas se toma de
`configs/default.yaml`.

Las métricas, predicciones de prueba y gráficos se guardan en
`reports/pd_evaluation/`. La prueba temporal 2014 se utiliza únicamente para
evaluación y no para elegir hiperparámetros ni semillas.
