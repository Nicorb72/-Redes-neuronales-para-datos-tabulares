# Datos procesados

Este directorio contiene la versión exacta de las matrices que consumen los
experimentos PD. Las matrices CSR (`X_*.npz`) y etiquetas (`y_*.npy`) se
generan una vez desde `data/raw/` con:

```bash
PYTHONPATH=src python -m credit_risk.data.prepare_data
```

El pipeline de entrenamiento carga estos archivos directamente y no necesita
el CSV original. Se versionan train (2007–2012), validation (2013) y test
(2014); el conjunto de 2015 es complementario. El preprocesador ajustado en
train y los nombres de las 44 columnas se guardan en
`artifacts/models/pd_preprocessor.joblib` y `pd_feature_names.json`.
El pipeline comprueba que las dimensiones y columnas coincidan antes de
entrenar.

Los artefactos procesados no están excluidos de Git: si el archivo supera los
límites del repositorio, acuerden una alternativa reproducible y documenten
cómo obtenerlo sin depender de APIs durante la ejecución.
