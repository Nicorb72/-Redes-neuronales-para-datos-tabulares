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

## 2. Pérdida dado incumplimiento (LGD) y Pérdida Esperada

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

**Tabla 3. Desempeño en el conjunto de prueba independiente (Año 2014, $N=13.981$).**<br>
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

### 2.4 Pérdida Esperada ($E[L] = PD \times LGD \times EAD$)

Al originar un crédito, el momento futuro de un posible impago es desconocido; por tanto, la exposición conservadora ex-ante es $EAD_0 = \text{funded\_amnt}$. Al evaluar retrospectivamente la cartera morosa, la exposición residual al incumplimiento es $EAD_{obs} = \text{funded\_amnt} - \text{total\_rec\_prncp}$.

Evaluando los 13.981 créditos castigados de 2014:
- **Exposición total comprometida:** $178.668.119,71 USD
- **Pérdida esperada estimada por el modelo:** $163.408.908,93 USD (tasa esperada: 91,46 %)
- **Pérdida real observada:** $168.195.585,76 USD (tasa real: 94,14 %)
- **Error absoluto medio:** $979,53 USD por préstamo.

### 2.5 Discusión: ¿Qué aportó la red neuronal en LGD?

La evidencia empírica es contundente: multiplicar por 115 la cantidad de parámetros entrenables (de 45 a 5.185) **no produjo ninguna mejora** en el error absoluto (MAE idéntico de 0,0771) y resultó en un RMSE ligeramente peor (0,0887 vs. 0,0868). La red neuronal añade costo computacional y variabilidad entre semillas sin aportar ventajas en capacidad predictiva para esta tarea tabular, confirmando que una mayor complejidad arquitectural no compensa la debilidad intrínseca de la señal predictiva ex-ante sobre recuperaciones ex-post.

## 3. Trabajo pendiente

Faltan los modelos y métricas finales de clasificación PD (Integrante 2), la integración final de la aplicación web publicada (Integrante 4) y la consolidación de la cacería de alucinaciones (Integrante 5).

## 4. Referencias bibliográficas (Normas APA 7.ª edición)

- Basel Committee on Banking Supervision. (2005). *An explanatory note on the Basel II IRB risk-weight functions*. Bank for International Settlements. https://www.bis.org/bcbs/irbriskweight.pdf
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. En *International Conference on Learning Representations (ICLR)*. https://openreview.net/forum?id=Bkg6RiCqY7
- Papke, L. E., & Wooldridge, J. M. (1996). Econometric methods for fractional response variables with an application to 401(k) plan participation rates. *Journal of Applied Econometrics, 11*(6), 619–632. https://doi.org/10.1002/(SICI)1099-1255(199611)11:6<619::AID-JAE418>3.0.CO;2-1
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research, 15*(56), 1929–1958. https://www.jmlr.org/papers/v15/srivastava14a.html
