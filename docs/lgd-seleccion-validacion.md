# Selección del checkpoint LGD

El checkpoint final se elige por el menor **RMSE de validación 2013** entre
las semillas evaluadas. En caso de empate exacto se conserva la primera
semilla del orden de ejecución (`101, 202, …, 3030`).

El proceso entrena las corridas, calcula sus métricas de validación y fija
el checkpoint. Solo después evalúa las corridas en prueba 2014. Las métricas
de prueba describen rendimiento y variabilidad; no deciden qué modelo guardar.

`artifacts/models/lgd_benchmark_results.json` conserva:

- `checkpoint_selection`: criterio, partición, semilla y valor de validación.
- `per_seed_metrics`: validación, prueba, tiempo y selección de cada corrida.
- Estadísticas de prueba de todas las semillas; el checkpoint individual no
  tiene por qué ser el de menor error de prueba.

Reproducir desde la raíz con las dependencias y matrices del proyecto:

```console
python -m credit_risk.lgd.experiments --seeds 30
python -m pytest tests/test_lgd.py tests/test_lgd_checkpoint_integration.py -q
```

Se necesita `PYTHONPATH=src`, como explica el README. Para una prueba rápida,
`--demo` entrena cinco semillas y guarda `*_demo.*`, sin sobrescribir el
benchmark completo ni el modelo final.

Las pruebas de regresión invierten a propósito el ranking de prueba y verifican
que no cambia la selección; cubren empates, métricas no finitas y cero semillas.
La prueba de integración carga el checkpoint y reproduce su MAE/RMSE en validación
y prueba, además de verificar la consistencia de las treinta corridas guardadas.

Esta corrección evita seleccionar la semilla por prueba. No convierte el conjunto
de prueba ya utilizado históricamente en un holdout nuevo; para una evaluación
externa adicional debe reservarse otro periodo sin usarlo para selección.
