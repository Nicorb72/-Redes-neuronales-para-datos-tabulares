# Riesgo de crédito con redes neuronales para datos tabulares

**Reporte técnico en formato de blog**

Este trabajo aborda dos componentes del riesgo de crédito de consumo:
estimar la probabilidad de incumplimiento (PD) y, para los préstamos que
incumplen, la pérdida dado incumplimiento (LGD). El equipo prepara una
comparación entre modelos de referencia y redes neuronales, una scorecard y
una aplicación para presentar resultados.

## Integrantes

1. **Nicole Guaranguay** — Datos y análisis.
2. **Nicolas Ruiz** — Clasificación: probabilidad de incumplimiento (PD).
3. **Sara Acevedo** — Regresión: pérdida dado incumplimiento (LGD).
4. **Sebastián Cataño** — Scorecard, interpretación y aplicación web.
5. **Ivan Dario Gomez Cabrera** — Integración, reproducibilidad y reporte.

## 1. Preparación de datos para PD

### 1.1 Conjunto de datos y objetivo

El módulo de datos utiliza el archivo `loan.csv` del conjunto
[Credit Risk Analysis de Kaggle](https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset).
El README del módulo reporta **887.379 préstamos**, originados entre junio de
2007 y diciembre de 2015. El archivo original no está incluido en GitHub por
su tamaño. La fecha de descarga debe registrarse cuando se consolide la
versión del conjunto usada por el equipo.

La variable `default_flag` se deriva de `loan_status`: los pagos confirmados
se codifican como 0, los incumplimientos como 1 y los estados sin resultado
definitivo quedan sin etiqueta. Estos últimos no se convierten en buenos
pagadores ni se utilizan para entrenar la clasificación de PD.

**Tabla 1. Codificación de `loan_status` para el objetivo PD.**<br>
**Fuente:** módulo de datos, `src/credit_risk/data/preprocessing.py`.

| Etiqueta | Estados de `loan_status` |
| --- | --- |
| 0 — Buen pagador | `Fully Paid`; `Does not meet the credit policy. Status:Fully Paid` |
| 1 — Mal pagador | `Charged Off`; `Default`; `Late (31-120 days)`; `Does not meet the credit policy. Status:Charged Off` |
| Sin etiqueta | `Current`; `Issued`; `In Grace Period`; `Late (16-30 days)` |

Según los conteos documentados por el módulo, hay **209.711** préstamos
etiquetados como buenos, **58.819** como malos y **618.849** sin etiqueta.
La **Figura 1** presenta esos conteos: la clase sin etiqueta es sustancial y
debe permanecer separada de las clases de entrenamiento.

<figure>
  <img src="../figures/distribucion_etiquetas_pd.svg" alt="Gráfico de barras con 209.711 préstamos buenos, 58.819 malos y 618.849 sin etiqueta.">
  <figcaption><strong>Figura 1.</strong> Conteo documentado de etiquetas para PD. <strong>Fuente:</strong> conteos registrados en <code>src/credit_risk/data/README.md</code>; gráfico elaborado para este reporte. El archivo original no estaba disponible en este clon para recalcularlos.</figcaption>
</figure>

### 1.2 Particiones cronológicas

`asignar_particiones` conserva los préstamos con etiqueta válida y los
separa según el año de originación: entrenamiento entre 2007 y 2012,
validación en 2013, prueba en 2014 y un conjunto complementario para 2015.
La **Figura 2** compara el tamaño reportado de cada partición.

**Tabla 2. Particiones temporales reportadas para PD.**<br>
**Fuente:** módulo de datos, `src/credit_risk/data/README.md`.

| Partición | Año(s) de originación | Registros |
| --- | --- | ---: |
| Entrenamiento | 2007–2012 | 90.347 |
| Validación | 2013 | 73.194 |
| Prueba | 2014 | 74.154 |
| Complementario | 2015 | 30.835 |

<figure>
  <img src="../figures/particiones_temporales_pd.svg" alt="Gráfico de barras con los tamaños reportados de entrenamiento, validación, prueba y conjunto complementario de 2015.">
  <figcaption><strong>Figura 2.</strong> Tamaño de las particiones temporales reportadas para PD. <strong>Fuente:</strong> conteos registrados en <code>src/credit_risk/data/README.md</code>; gráfico elaborado para este reporte. El archivo original no estaba disponible en este clon para recalcularlos.</figcaption>
</figure>

La partición por originación no constituye por sí sola una validación
histórica estricta: el estado final de algunos préstamos pudo conocerse
después de su fecha de originación. Además, el README del módulo reporta que
el 67,82 % del conjunto está en estado `Current` y que el 92,68 % de los
créditos originados en 2015 no tiene etiqueta. Estas observaciones limitan la
interpretación de las tasas de incumplimiento por año y pueden generar sesgo
de selección entre los préstamos etiquetados.

### 1.3 Variables y transformaciones

La implementación prepara **15 variables predictoras de entrada**:

- **Numéricas (11):** `loan_amnt`, `annual_inc`, `dti`, `delinq_2yrs`,
  `inq_last_6mths`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`,
  `total_acc` y `antiguedad_crediticia_meses`.
- **Categóricas (4):** `term`, `purpose`, `emp_length` y `home_ownership`.

La antigüedad crediticia se calcula como la diferencia en meses entre
`earliest_cr_line` y `issue_d`; valores negativos se consideran faltantes. El
preprocesamiento imputa numéricas con la mediana y las estandariza; imputa
categóricas con `Desconocido` y las codifica con one-hot encoding. El
preprocesador se ajusta **solo con entrenamiento** y luego transforma
validación, prueba y el conjunto complementario. El README del módulo reporta
44 características después de la codificación.

La revisión documentada no identificó IDs de préstamo duplicados y señala
44.825 valores faltantes en `emp_length` (5,05 %). Los valores financieros
extremos se conservan inicialmente, por lo que su efecto debe evaluarse en
los modelos. La auditoría de fuga excluye predictores evidentes de resultados
posteriores al desembolso, pero el propio módulo advierte que no verifica por
completo la disponibilidad temporal de cada variable.

### 1.4 Implementación y reproducción

El flujo está dividido en `load_data.py` (carga del CSV),
`preprocessing.py` (objetivo, fechas, variables y transformador),
`prepare_data.py` (particiones y matrices) y `eda.py` (resúmenes y funciones
de análisis exploratorio). Desde la raíz del repositorio, con las
dependencias instaladas y `loan.csv` en `data/raw/loan/`, la preparación se
ejecuta en PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m credit_risk.data.prepare_data
```

El flujo guarda matrices dispersas `X_*.npz`, etiquetas `y_*.npy`, el
preprocesador `preprocesador_pd.joblib` y los nombres de variables
transformadas `columnas_pd.json`. Estos artefactos locales están excluidos de
Git en la configuración actual; el equipo debe decidir cuáles datos
procesados y modelos finales necesarios para reproducir o servir el proyecto
deben versionarse.

**Estado de las figuras.** El módulo `eda.py` contiene funciones para
graficar los conteos de etiquetas y las particiones temporales. Se regeneran
con `python -m credit_risk.data.generate_eda_figures` desde la raíz, tras
configurar `PYTHONPATH=src` y disponer del CSV original. En este clon no está
el CSV, por lo que las Figuras 1 y 2 muestran los conteos documentados en el
README del módulo, sin recálculo independiente. Deben regenerarse y
contrastarse con el archivo fuente antes de presentar los conteos como
verificados.

## 2. Trabajo pendiente

Esta versión documenta la preparación de PD. Todavía faltan la exploración
recalculada desde el archivo fuente, los modelos y métricas de PD, la
construcción y evaluación de LGD, la scorecard, la aplicación, la discusión
comparativa, la bibliografía APA 7 y la cacería de alucinaciones. No se
presentan aquí resultados de modelos que aún no están integrados.
