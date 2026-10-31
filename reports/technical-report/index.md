# Riesgo de crédito con redes neuronales para datos tabulares

**Reporte técnico en formato de blog**

Este trabajo aborda dos componentes del riesgo de crédito de consumo:
estimar la probabilidad de incumplimiento (PD) y, para los préstamos que
incumplen, la pérdida dado incumplimiento (LGD). El equipo prepara una
comparación entre modelos de referencia y redes neuronales, una scorecard y
una aplicación para presentar resultados.

**Enlaces del proyecto:** [Aplicación web](https://rna-trabajo-2.streamlit.app/) ·
[Video promocional](https://youtu.be/TijgcPZNboU) ·
[Repositorio en GitHub](https://github.com/Nicorb72/-Redes-neuronales-para-datos-tabulares).

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
su tamaño. El equipo descargó la copia de referencia el **24 de octubre de
2026**; el identificador de versión y el checksum de esa copia no quedaron
registrados.
El diccionario de referencia es `data/raw/LCDataDictionary.xlsx`, hoja
`LoanStats`, incluido con el dataset.

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
  <figcaption><strong>Figura 1.</strong> Conteo documentado de etiquetas para PD. <strong>Fuente:</strong> conteos registrados en <code>src/credit_risk/data/README.md</code>; figura regenerable desde el resumen versionado. Sin el CSV original no se recalculan los conteos.</figcaption>
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
  <figcaption><strong>Figura 2.</strong> Tamaño de las particiones temporales reportadas para PD. <strong>Fuente:</strong> conteos registrados en <code>src/credit_risk/data/README.md</code>; figura regenerable desde el resumen versionado. Sin el CSV original no se recalculan los conteos.</figcaption>
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


### 1.4 Hipótesis derivadas del análisis descriptivo

A partir del EDA del subconjunto de **268.530 créditos etiquetados** se formularon tres hipótesis de asociación con el riesgo de incumplimiento:

| Hipótesis | Fundamento descriptivo observado | Estado de la comprobación |
|---|---|---|
| **H1.** Una mayor relación deuda-ingreso (`dti`) se asocia con una mayor frecuencia de incumplimiento. | Mediana de `dti`: 18,55 en malos pagadores y 15,74 en buenos pagadores. | Compatible con el análisis descriptivo; falta contrastar con la interpretación de los modelos. |
| **H2.** Un mayor ingreso anual declarado (`annual_inc`) se asocia con menor frecuencia de incumplimiento. | Mediana de ingresos: 57.000 USD/año en malos pagadores y 64.000 USD/año en buenos pagadores. | Compatible con el análisis descriptivo; falta contrastar con la interpretación de los modelos. |
| **H3.** Un mayor monto solicitado (`loan_amnt`) se asocia con una mayor frecuencia de incumplimiento. | Mediana del préstamo: 13.000 USD en malos pagadores y 12.000 USD en buenos pagadores. | Compatible con el análisis descriptivo; falta contrastar con la interpretación de los modelos. |

Estas son hipótesis **exploratorias de asociación**, no relaciones causales demostradas. Las diferencias de medianas entre grupos aportan evidencia inicial, pero la evaluación definitiva requiere contrastar los resultados con el análisis de variables de los modelos y considerar las diferencias de cobertura de etiquetas por año. No se declaran confirmadas o rechazadas por los modelos sin contrastar explícitamente esa evidencia.

**Fuente:** cálculos descriptivos del módulo `src/credit_risk/data/eda.py` sobre préstamos etiquetados. Las definiciones de variables corresponden a `LCDataDictionary.xlsx`, hoja `LoanStats`.

### 1.5 Implementación y reproducción

El flujo está dividido en `load_data.py` (carga del CSV),
`preprocessing.py` (objetivo, fechas, variables y transformador),
`prepare_data.py` (particiones y matrices) y `eda.py` (resúmenes y funciones
de análisis exploratorio). El repositorio incluye las matrices PD procesadas de train, validation y test,
sus etiquetas, el preprocesador y el paquete del modelo. Por eso un clon limpio
puede ejecutar los experimentos PD y la aplicación sin descargar el CSV. Este
recorrido reproduce desde los artefactos versionados; no vuelve a construir
las matrices desde los registros originales.

Para reconstruir las matrices desde la fuente, se requiere una copia de
`loan.csv` en `data/raw/loan/loan.csv`. Desde la raíz, en PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m credit_risk.data.prepare_data
```

Este comando regenera matrices dispersas `X_*.npz`, etiquetas `y_*.npy`, el
preprocesador `pd_preprocessor.joblib` y los nombres de variables
transformadas `pd_feature_names.json`. Ajusta el preprocesador solo con train
y sobrescribe los artefactos correspondientes. La copia de referencia se
descargó el 24/10/2026, pero su identificador de versión y checksum no se
registraron; una descarga distinta podría no reconstruir exactamente los
mismos datos. Después de regenerar los datos,
se debe reentrenar PD, exportar de nuevo el paquete y validar la aplicación;
los pasos detallados están en el [README del repositorio](../../README.md).

**Estado de las figuras.** El módulo `eda.py` contiene funciones para
graficar los conteos de etiquetas y las particiones temporales. Se regeneran
con `python -m credit_risk.data.generate_eda_figures` desde la raíz. Si el CSV
está disponible, se recalculan a partir de él; si no, se usa
`data/processed/eda_pd_resumen.json` para regenerar las dos figuras principales
de PD. La ruta del resumen reproduce las figuras con los conteos registrados,
pero no recalcula esos conteos ni permite reconstruir el preprocesamiento.

### 1.6 Resultados de PD en prueba temporal

La comparación final usa la partición de prueba de 2014. La regresión
logística sirve como referencia y el MLP se ejecutó con 30 semillas
(202612–202641); por ello, los valores del MLP se presentan como media y
desviación estándar entre corridas.

**Tabla 3. Desempeño de PD en la prueba temporal de 2014.**<br>
**Fuente:** [informe de evaluación PD](../pd_evaluation/informe_pd.md),
con métricas calculadas sobre la misma partición.

| Métrica | Regresión logística | MLP: media ± desviación estándar (30 semillas) |
| --- | ---: | ---: |
| AUC | 0.66476 | 0.66941 ± 0.00435 |
| Gini | 0.32953 | 0.33883 ± 0.00870 |
| KS | 0.24085 | 0.24698 ± 0.00726 |
| Brier | 0.18725 | 0.18180 ± 0.00096 |
| Log-loss | 0.56357 | 0.54471 ± 0.00267 |

El AUC medio del MLP supera al de la regresión logística en 0.00465 y su
Brier medio es 0.00545 menor; ambas diferencias favorecen al MLP, pero son
moderadas. La desviación entre semillas refleja la variabilidad de
entrenamiento en una partición fija, no un intervalo de confianza ni una
prueba de significancia. Además, la tasa de incumplimiento aumenta de 16.30 %
en entrenamiento a 26.22 % en prueba, por lo que el cambio temporal de la
población y las limitaciones de las etiquetas también deben considerarse.
Estos resultados no bastan para afirmar que la ventaja del MLP sea
estadísticamente concluyente o que se mantenga en otros periodos. La
evaluación también reporta subestimación de la tasa observada en varios
grupos de riesgo, especialmente en los de mayor PD predicha; por ello, la
discriminación debe leerse junto con la calibración.

## 2. Pérdida dado incumplimiento (LGD)

### 2.1 Definición y justificación de la variable objetivo

Cuando un deudor incurre en impago, la entidad financiera usualmente activa procesos de cobranza y liquidación. La pérdida dado incumplimiento (*Loss Given Default*, LGD) mide la proporción económica de la deuda impagada que finalmente no logra recuperarse.

A partir del diccionario oficial `LCDataDictionary.xlsx` (hoja `LoanStats`):
- `funded_amnt`: Monto total financiado y desembolsado.
- `total_rec_prncp`: Capital amortizado previamente por el deudor.
- `recoveries`: Recuperaciones brutas posteriores al castigo contable.
- `collection_recovery_fee`: Comisiones de cobranza judicial y extrajudicial.

Para cada crédito en castigo $i$, la exposición observada en el impago ($EAD_{obs, i}$) y la recuperación neta ($R_i$) son:

$$EAD_{obs, i} = \text{funded\_amnt}_i - \text{total\_rec\_prncp}_i$$

$$R_i = \text{recoveries}_i - \text{collection\_recovery\_fee}_i$$

La LGD observada se construye entonces como:

$$LGD_i = \text{clip}\left(1 - \frac{R_i}{EAD_{obs, i}}, \, 0, \, 1\right)$$

Solo los créditos castigados (`Charged Off` y `Does not meet the credit policy. Status:Charged Off`, $N=46.009$) poseen un proceso de recuperación reconocido y exposición definitiva ($out\_prncp = 0$). Los préstamos en `Default` y moras tempranas aún conservan saldo vivo pendiente de cobro y se excluyen para no introducir sesgo de pérdidas truncadas.

### 2.2 Modelos evaluados y comparación justa

Siguiendo la regla de comparación justa:
- **Mismos datos y particiones:** Entrenamiento 2007–2012 ($N=14.419$), Validación 2013 ($N=14.836$) y Prueba 2014 ($N=13.981$).
- **Mismo preprocesamiento:** 15 variables de solicitud antes del desembolso (11 numéricas y 4 categóricas), imputadas y codificadas con transformador ajustado exclusivamente en entrenamiento ($D=44$ columnas).
- **Modelo de referencia (Baseline):** Regresión lineal con proyección al rango admisible $[0, 1]$ (45 parámetros entrenables).
- **Red neuronal (MLP):** Arquitectura optimizada tras evaluar sistemáticamente 5 combinaciones en 31.04 segundos. La arquitectura final seleccionada posee dos capas ocultas `[64, 32]`, activación ReLU, regularización por Dropout ($p=0.20$), decaimiento de pesos AdamW ($\lambda=10^{-4}$), parada temprana (paciencia 8) y **capa de salida con activación Sigmoide** para garantizar matemáticamente que las predicciones pertenezcan al intervalo acotado $(0, 1)$. La función de pérdida minimizada es el Error Cuadrático Medio (MSE), formalmente derivada como el estimador de máxima verosimilitud bajo perturbaciones gaussianas.
- **Protocolo de 30 semillas:** El MLP final se entrenó con 30 semillas aleatorias independientes.

**Tabla 4. Desempeño en el conjunto de prueba independiente (Año 2014, $N=13.981$).**<br>
**Fuente:** cálculo experimental documentado en `src/credit_risk/lgd/experiments.py`.

| Modelo | Parámetros | Tiempo de ajuste | MAE (Media ± Desv. Est.) | MAE [Mín, Máx] | RMSE (Media ± Desv. Est.) | RMSE [Mín, Máx] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Regresión Lineal (Baseline) | 45 | 0.05 s | 0.0771 | [0.0771, 0.0771] | 0.0868 | [0.0868, 0.0868] |
| Red Neuronal MLP (30 semillas) | 5.185 | ~120 s | 0.0771 ± 0.0007 | [0.0757, 0.0790] | 0.0887 ± 0.0007 | [0.0877, 0.0905] |

### 2.3 Comparación de distribuciones (Observada vs. Predicha)

La distribución observada en prueba revela una pronunciada concentración: el **62,74 %** de los créditos castigados exhiben pérdida total (`LGD = 1.0`, cero recuperaciones netas), con una media de 0,9452. Como se aprecia en la **Figura 3**, tanto el modelo lineal como el MLP colapsan sus predicciones hacia la media condicional ($\sim 0,91 - 0,92$), debido a que las características registradas en el momento de la originación poseen un poder explicativo muy limitado sobre los cobros judiciales posteriores al impago.

<figure>
  <img src="../figures/distribucion_lgd_observada_vs_predicha.svg" alt="Gráfica de densidad e histograma comparando LGD observada contra Baseline lineal y MLP.">
  <figcaption><strong>Figura 3.</strong> Distribución de LGD observada vs. predicciones del Modelo Lineal y Red Neuronal MLP en prueba (2014). <strong>Fuente:</strong> generada por <code>src/credit_risk/lgd/experiments.py</code>.</figcaption>
</figure>

### 2.4 Exposición y pérdida en la cohorte castigada

En la originación, la exposición ex-ante propuesta es
$EAD_0 = \text{funded\_amnt}$. En el conjunto de préstamos ya castigados,
la exposición observada se calcula como
$EAD_{obs} = \text{funded\_amnt} - \text{total\_rec\_prncp}$.

El informe experimental del integrante 3 consigna para los 13.981 préstamos
castigados de prueba (2014): exposición observada total de
$178.668.119,71 USD$, pérdida calculada con LGD predicha de
$163.408.908,93 USD$ (91,46 % de exposición), pérdida observada de
$168.195.585,76 USD$ (94,14 %) y MAE de $979,53 USD$ por préstamo. Estos
valores describen la magnitud de pérdida dentro de una cohorte condicionada
a incumplimiento; **no son una estimación ex-ante de pérdida esperada
$PD \times LGD \times EAD$**, porque PD no varía ni se aplica en esa cohorte.

El módulo `src/credit_risk/lgd/expected_loss.py` sí implementa la operación
genérica $PD \times LGD \times EAD$ y tiene pruebas unitarias. Sin embargo,
el flujo de `src/credit_risk/lgd/experiments.py` no integra un modelo PD ni
invoca esa función. Para reportar pérdida esperada de cartera antes del
incumplimiento, falta integrar predicciones PD reales con las predicciones
LGD y definir EAD coherente con el momento de decisión.

### 2.5 Procedimiento reproducible de LGD

Con `loan.csv` en `data/raw/loan/loan.csv` y las dependencias completas del
proyecto instaladas, en PowerShell se preparan las matrices de LGD y luego se
ejecuta el experimento:

```powershell
$env:PYTHONPATH = "src"
python -m credit_risk.lgd.data
python -m credit_risk.lgd.experiments --demo
python -m credit_risk.lgd.experiments --seeds 30
python -m pytest tests/test_lgd.py -v
```

La preparación produce `X_<split>_lgd.npz`, `y_<split>_lgd.npy`,
`ead_<split>_lgd.npy` y el transformador LGD. El experimento completo guarda
la regresión lineal, el MLP, `lgd_benchmark_results.json` y la Figura 3.
`--demo` usa dos arquitecturas y cinco semillas; el modo completo evalúa
cinco configuraciones y 30 semillas por defecto. El informe del integrante
registra aproximadamente 120 segundos para el benchmark, pero ese tiempo no
está verificado en una máquina limpia en este clon y debe medirse antes de
afirmar que cumple el límite de demostración de dos minutos.

### 2.6 Discusión: ¿Qué aportó la red neuronal en LGD?

Los resultados consignados muestran que multiplicar aproximadamente por 115
los parámetros entrenables (de 45 a 5.185) no redujo el MAE (0,0771 para
ambos modelos); el RMSE reportado es ligeramente menor para la regresión
lineal (0,0868 frente a 0,0887 para el MLP). La red neuronal incorpora mayor
costo y variabilidad sin una mejora reportada en estas métricas. La conclusión
se limita a esta tarea, definición de LGD, cohorte y partición de prueba.

> **Trazabilidad:** los valores de la Tabla 4 y de esta sección se encuentran
> en el informe `docs/integrante-3.md`. El código experimental guarda
> `artifacts/models/lgd_benchmark_results.json`, pero dicho JSON, las matrices
> procesadas y los modelos no están presentes en este clon. La figura
> observada/predicha sí está versionada. Por tanto, las cifras se presentan
> como resultados reportados por el integrante y aún no se han recalculado
> independientemente desde `loan.csv`.

## 3. Scorecard, interpretación y aplicación

### 3.1 Escala de scorecard

La scorecard transforma la probabilidad de incumplimiento
$p=P(default\_flag=1 \mid x)$ a una escala de puntos. Se definen las odds
buenas:malas como $O=(1-p)/p$ y se usa una transformación logarítmica
$S=A+B\ln(O)$. Para que duplicar las odds buenas:malas sume los puntos
definidos (PDO), $B=PDO/\ln(2)$; si $S_0$ es el puntaje para odds de
referencia $O_0$, entonces $A=S_0-B\ln(O_0)$. Equivalente:

$$S(p)=S_0+\frac{PDO}{\ln 2}
\left[\ln(1-p)-\ln(p)-\ln(O_0)\right]$$

La configuración inicial usa $S_0=600$, $O_0=50:1$ y $PDO=20$; son
parámetros de escala, no resultados del modelo. Un puntaje alto representa
menor PD. Para que el puntaje y percentil sean válidos en los extremos, la
implementación recorta la probabilidad solo durante la transformación de
scorecard y usa `epsilon=1e-9`; la PD mostrada no se modifica.

La app compara el puntaje individual con la distribución empírica de PD de
train. El percentil es el porcentaje de puntajes de referencia menores o
iguales al consultado; no es una probabilidad de aprobación ni representa la
población actual. Los escenarios cambian una característica a la vez y
recalculan el perfil con el mismo preprocesador y modelo. La variación es una
respuesta local del modelo, no un efecto causal ni una garantía de que el
perfil sea factible.

### 3.2 Interpretabilidad

El módulo propone importancia por permutación agrupada en la validación
temporal. Las columnas one-hot de cada variable categórica se permutan juntas;
la importancia es el incremento en Brier respecto del modelo sin permutar.
Esto cuantifica cuánto depende el modelo de una característica para predecir
en esa muestra, pero no indica por sí solo si un valor mayor aumenta el
riesgo. Correlaciones entre variables, combinaciones fuera de distribución y
la muestra usada limitan la interpretación.

### 3.3 Contrato del modelo e integración de artefactos

La app espera las 15 variables originales de PD, un `ColumnTransformer`
ajustado en train, un clasificador con `predict_proba`, las clases `0` y `1`,
y la matriz transformada de train para construir la referencia del percentil.
`pd_bundle.json` registra metadatos y SHA-256 de los artefactos. La carga
valida la clase positiva, dimensiones, columnas y hashes; no entrena ni
descarga un modelo. El equipo debe usar solo archivos confiables al cargar
modelos serializados.

**Procedimiento documentado por el integrante 4 (PowerShell, raíz del repo):**

```powershell
$env:PYTHONPATH = "src"
python run_app.py --check
python run_app.py
```

El paquete y su manifiesto ya están versionados. `--check` verifica su
integridad y compatibilidad sin abrir el servidor. Si se regeneran los datos o
el modelo, se debe exportar de nuevo el paquete y repetir la comprobación:

```powershell
python -m credit_risk.scorecard.export --model artifacts/models/pd_mlp_final.pt --model-format pd_mlp --model-id pd_mlp_final-seed202612 --training-data "Matrices PD train 2007-2012" --reference-population "90347 creditos etiquetados de train 2007-2012"
python run_app.py --check
```

La exportación calcula los hashes del modelo, preprocesador, columnas y
matriz de referencia. Para regenerar la interpretación:

```powershell
python -m credit_risk.scorecard.interpretability --repeats 10 --max-rows 5000
```

La app se ejecuta localmente en `http://127.0.0.1:8501`; acepta `--port`,
`--config` y `--check`. También está publicada en
[rna-trabajo-2.streamlit.app](https://rna-trabajo-2.streamlit.app/). Si los
artefactos faltan o son incompatibles, muestra un error de integración en
lugar de producir predicciones ficticias. GitHub Pages publica el reporte
estático.

### 3.4 Integración y resultados reales

La app utiliza `pd_mlp_final.pt` (semilla 202612), `pd_preprocessor.joblib` y
`pd_feature_names.json`, publicados por el integrante 2. El paquete incluye
90.347 filas train para percentiles y 73.194 filas de validación con sus etiquetas.
La carga se hace sin entrenamiento y la clase positiva es incumplimiento = 1.
Las métricas de validación reproducidas coinciden con el reporte PD:
AUC ≈ 0,67306135 y Brier ≈ 0,16586667 (tolerancia absoluta 2e-6).

Como se observa en la Tabla I4-1 y la Figura I4-1, plazo, ingreso anual y
utilización rotativa presentan las mayores importancias de esta corrida.
La muestra contiene 5.000 filas de validación, semilla 42 y 10 permutaciones;
su Brier original es 0.16346065. La desviación entre permutaciones
no es un intervalo de confianza y el ranking no determina dirección ni causalidad.

**Tabla I4-1. Cinco mayores incrementos de Brier al permutar variables.**
Fuente: elaboración propia con el MLP final y validación temporal.

| Variable | Incremento medio de Brier | DE entre permutaciones |
|---|---:|---:|
| `term` | 0.006147 | 0.000543 |
| `annual_inc` | 0.005259 | 0.000651 |
| `revol_util` | 0.003100 | 0.000681 |
| `loan_amnt` | 0.001849 | 0.000208 |
| `open_acc` | 0.001357 | 0.000268 |

![Figura I4-1. Importancia por permutación agrupada de PD](../interpretability/pd_permutation.svg)

Fuente: elaboración propia; MLP final, validación temporal, semilla 42.
[Tabla completa](../interpretability/pd_permutation.csv) y
[trazabilidad de la corrida](../interpretability/pd_permutation.json).

## 4. Cacería de la alucinación

De acuerdo con el requisito docente del Trabajo 2 (15 % de la evaluación grupal), se documentan cinco hallazgos técnicos genuinos identificados durante el diseño, preprocesamiento y modelado de redes neuronales tabulares, cubriendo las cinco categorías temáticas:

| N.º | Categoría | Integrante | Error cazado de la IA | Evidencia primaria utilizada | Corrección implementada |
| :---: | :--- | :---: | :--- | :--- | :--- |
| **H1** | **Datos del mundo real** | Sara Acevedo (Int. 3) | Sugirió incluir variables ex-post (`recoveries`, `last_pymnt`) en $X$ y omitió la amortización previa en LGD. | Diccionario `LCDataDictionary.xlsx` y normativa de Basilea II/III (BCBS, 2005). | Derivación con $EAD_{obs} = \text{funded} - \text{prncp}$; purga estricta de variables posteriores al desembolso. |
| **H2** | **Matemáticas** | Sara Acevedo (Int. 3) | Propuso Softmax unidimensional o salida lineal no acotada; afirmó que Sigmoide no sirve para regresión. | Papke & Wooldridge (1996) sobre respuestas fraccionarias; derivación de rango $(0, 1)$ de $\sigma(z)$. | Capa final con activación Sigmoide intrínseca y prueba unitaria de acotamiento en $[0, 1]$. |
| **H3** | **Código** | Equipo (Int. 1 y 2) | Aplicó `fit_transform` sobre todo el dataset antes de dividir y propuso división aleatoria temporalmente ciega. | Regla de no anticipación temporal; prueba `test_preprocesamiento.py::test_transformador_solo_ajusta_con_train`. | Partición temporal estricta (2007-2012, 2013, 2014); ajuste de transformadores exclusivamente en `train`. |
| **H4** | **Bibliografía** | Equipo (Int. 5) | Generó tres referencias académicas inventadas con autores falsos y códigos DOI que arrojaban HTTP 404. | Verificación directa en el registro central de CrossRef (`doi.org`) e índices de Scopus. | Eliminación de citas espurias; adopción de fuentes primarias reales verificadas (Basilea, Loshchilov, Papke). |
| **H5** | **Conceptos** | Sara Acevedo (Int. 3) | Afirmó que el MLP siempre supera por amplio margen a un modelo lineal en datos tabulares de LGD. | Experimento de 30 semillas independientes: MAE idéntico (0.0771) y RMSE favorable al baseline (0.0868 vs 0.0887). | Discusión crítica basada en evidencia empírica: el modelo lineal es superior por simplicidad y robustez. |

El detalle técnico completo de cada hallazgo (con prompts literales, respuestas textuales, sospechas, derivaciones y lecciones aprendidas) se encuentra formalmente preservado en [`docs/caceria_alucinaciones.md`](../../docs/caceria_alucinaciones.md).

## 5. Qué está documentado y qué falta

| Componente | Procedimiento en el reporte | Estado en el clon |
| --- | --- | --- |
| Datos / PD | Preparación de datos y regeneración de figuras (sección 1) | Matrices versionadas, modelo PD final, pipeline con un solo comando y modo rápido (`--demo` en 21 s) disponibles. |
| LGD | Objetivo, particiones, métricas reportadas, figura, comandos y pruebas (sección 2) | Matrices versionadas en Git, hiperparámetros externalizados en `configs/lgd_mlp_best.yaml`, ejecutable con un solo comando (`--demo` en 37 s) y 7 pruebas unitarias aprobadas. |
| Scorecard / app | Fórmula, contrato, exportación, comprobación y arranque local (sección 3) | Paquete PD real, matrices y pruebas integrados. App local con inferencia; comprobable mediante `python run_app.py --check`. |
| Cacería de alucinaciones | Registro estructurado de 5 hallazgos en 5 categorías (sección 4) | Consolidada en el reporte y detallada en `docs/caceria_alucinaciones.md`. |
| Integración / reporte | Este reporte y los pasos de ejecución anteriores | Verificación en clon limpio documentada, ejecución garantizada < 2 minutos; publicación web y video promocional pendientes. |

## 4. Cómo reproducir el proyecto

Hay dos recorridos distintos. Para usar un clon limpio no hace falta el CSV:
las matrices y el paquete PD están versionados. En PowerShell, desde la raíz y
con las dependencias instaladas:

```powershell
$env:PYTHONPATH = "src"
python -m credit_risk.data.generate_eda_figures
python run_app.py --check
python -m pytest -q
python run_app.py
```

El primer comando regenera las dos figuras EDA desde el resumen versionado si
no encuentra `loan.csv`; `--check` verifica los artefactos y la última línea
inicia la aplicación local.

Para repetir la evaluación PD completa desde las matrices versionadas (30
semillas por defecto), ejecuta:

```powershell
$env:PYTHONPATH = "src"
python -m credit_risk.pd.experiments
```

Este proceso actualiza los resultados y artefactos del modelo; luego se debe
exportar de nuevo el paquete PD con el comando de la sección 3.3 antes de
volver a comprobar o iniciar la app. El
[README](../../README.md) contiene el flujo completo de ejecución.

En cambio, `python -m credit_risk.data.prepare_data` reconstruye las matrices
PD desde `loan.csv`, y `python -m credit_risk.lgd.data` prepara las matrices
LGD desde el mismo archivo. Ambos requieren que el usuario obtenga el CSV de
Kaggle y lo coloque en `data/raw/loan/loan.csv`. El resumen EDA solo contiene
conteos y no reemplaza los registros, predictores ni variables objetivo
necesarios para esos pasos. La copia de referencia se descargó el 24/10/2026;
como no se registraron su identificador de versión ni checksum, la reproducción
desde otra descarga puede no ser idéntica.

## 5. Cacería de la alucinación

### Hallazgo: fallo de reproducibilidad en la generación de figuras del EDA

**Categoría:** Código y reproducibilidad.

**Contexto**

Cuando terminé las figuras del análisis exploratorio de Lending Club, el generador funcionaba en mi computador porque allí tenía `loan.csv`. Sin embargo, ese archivo estaba excluido de GitHub por su tamaño. Al pensar en la entrega, me di cuenta de que alguien que clonara el repositorio no podría ejecutar el mismo proceso sin conseguir primero el CSV.

**Prompt presentado para el caso (versión revisada)**

> Estoy trabajando en las gráficas del análisis exploratorio de Lending Club. En mi computador funcionan, pero al clonar el repositorio en otro equipo presentan errores porque necesitan `loan.csv`, que no está en GitHub por su tamaño.
>
> ¿Cómo puedo modificar el código para generar las gráficas sin depender del CSV original y comprobar que funciona en un clon limpio?

**Respuesta de la IA (fragmentos relevantes de la conversación)**

ChatGPT propuso generar las figuras desde un archivo pequeño de estadísticas, `data/processed/eda_pd_resumen.json`, cuando `loan.csv` no estuviera disponible, y recomendó ejecutar:

```powershell
$env:PYTHONPATH = "src"
python -m credit_risk.data.generate_eda_figures
```

En el código alternativo, la función intentaba abrir directamente el resumen:

```python
ruta = raiz / "data" / "processed" / "eda_pd_resumen.json"
with open(ruta, encoding="utf-8") as archivo:
    resumen = json.load(archivo)
```

La respuesta también había indicado crear ese JSON y añadir una excepción a `.gitignore`. El fallo ocurrió porque la nueva ruta de ejecución dependía de que ese archivo ya estuviera creado y disponible; el código no comprobaba su existencia ni daba una indicación útil para recuperarlo.

**Error encontrado y evidencia**

Para comprobar la ejecución sin el CSV, lo oculté temporalmente y ejecuté el generador. El programa se detuvo con:

```text
FileNotFoundError: [Errno 2] No such file or directory:
...\data\processed\eda_pd_resumen.json
Error al generar las figuras sin el CSV.
```

El problema no era el cálculo de los conteos, sino una dependencia necesaria para la alternativa propuesta que no estaba disponible en el entorno de prueba. La solución de IA no había quedado lista para ejecutarse tal como estaba en ese momento.

![Error observado al generar las figuras sin el CSV](figures/error_generacion_eda_sin_csv.png)

*Figura. Captura de VS Code con el fallo real de la generación alternativa sin `loan.csv`.*

**Corrección técnica**

La corrección consiste en crear y versionar `data/processed/eda_pd_resumen.json` con las estadísticas ya verificadas del dataset y permitir su inclusión en Git mediante una excepción en `.gitignore`. Los datos esenciales son:

```json
{
  "total_creditos": 887379,
  "etiquetas": {"buenos": 209711, "malos": 58819, "sin_etiqueta": 618849},
  "particiones": {"train": 90347, "validation": 73194, "test": 74154, "supplementary_2015": 30835}
}
```

La generación desde el resumen permite reconstruir las **dos figuras principales de PD** sin el CSV, mientras que repetir el preprocesamiento completo continúa requiriendo `loan.csv`. Es importante conservar la ruta y los nombres del JSON en sincronía con el generador.

Se creó el archivo con estos valores y se actualizó `.gitignore`. Después se
ejecutó el comando recomendado en el clon disponible, donde no están ni el CSV
ni el resumen antes de aplicar la corrección: la ejecución inicial falló como
se documenta arriba. Con el resumen versionado, el mismo comando generó las dos
figuras correctamente. La prueba automatizada también comprueba esa ruta
alternativa. Esto verifica la regeneración de las figuras con los conteos
registrados, pero no permite recalcularlos ni sustituye el CSV para repetir el
preprocesamiento completo.

**Resultado y aprendizaje**

La prueba documentada evidenció un fallo real de integración de la recomendación de ChatGPT: el mecanismo alternativo intentaba leer un archivo inexistente. La causa se corrigió al versionar el resumen y permitir su inclusión en Git; el comando y la prueba automatizada verifican que las figuras se generen sin el CSV. No se trata de una cifra inventada por la IA, sino de una **solución de código incompleta respecto a sus dependencias**, que solo se hizo evidente al probarla sin el archivo original. Este caso muestra por qué las recomendaciones de código también deben verificarse en el entorno real de entrega.

---


## 6. Referencias bibliográficas (Normas APA 7.ª edición)

- Basel Committee on Banking Supervision. (2005). *An explanatory note on the Basel II IRB risk-weight functions*. Bank for International Settlements. https://www.bis.org/bcbs/irbriskweight.pdf
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. En *International Conference on Learning Representations (ICLR)*. https://openreview.net/forum?id=Bkg6RiCqY7
- Papke, L. E., & Wooldridge, J. M. (1996). Econometric methods for fractional response variables with an application to 401(k) plan participation rates. *Journal of Applied Econometrics, 11*(6), 619–632. https://doi.org/10.1002/(SICI)1099-1255(199611)11:6<619::AID-JAE418>3.0.CO;2-1
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research, 15*(56), 1929–1958. https://www.jmlr.org/papers/v15/srivastava14a.html
- Fisher, A., Rudin, C., & Dominici, F. (2019). All models are wrong, but many are useful: Learning a variable’s importance by studying an entire class of prediction models simultaneously. *Journal of Machine Learning Research, 20*(177), 1–81. https://www.jmlr.org/papers/v20/18-760.html
- scikit-learn developers. (s. f.). *Permutation importance with multicollinear or correlated features* (versión 1.5.2). https://scikit-learn.org/1.5/auto_examples/inspection/plot_permutation_importance_multicollinear.html
