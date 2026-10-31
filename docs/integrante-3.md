# Integrante 3: Pérdida Dado el Incumplimiento (LGD) y Pérdida Esperada (EL)

**Responsable:** Integrante 3  
**Rama de trabajo:** `INTEGRANTE-3`  
**Fecha:** 26 de octubre de 2026  

---

## 1. Alcance y responsabilidades

Según la distribución oficial del trabajo y la consigna docente:
1. **Construir y justificar la variable objetivo LGD:** derivación matemática a partir del diccionario oficial de Lending Club, identificación de campos y discusión de limitaciones del mundo real.
2. **Modelo de referencia (Baseline):** implementar y ajustar una regresión lineal con proyección al rango admisible $[0, 1]$.
3. **Red neuronal (MLP de regresión):** diseñar la arquitectura en PyTorch, justificar la función de activación de salida (Sigmoide) y la función de pérdida (MSE / Huber / BCE) a partir de la naturaleza acotada de la LGD y fundamentos probabilísticos (Máxima Verosimilitud).
4. **Evaluación comparativa justa:** medir MAE, RMSE, correlaciones y comparar formalmente la distribución de predicciones vs. la observada (test de Kolmogorov-Smirnov, distancia de Wasserstein).
5. **Protocolo de 30 semillas independientes:** ejecutar la configuración final del MLP con 30 semillas distintas, reportando media, desviación estándar, mejor y peor corrida, junto con presupuesto explícito de búsqueda y conteo de parámetros entrenables.
6. **Pérdida esperada ($E[L] = PD \times LGD \times EAD$):** definir y contrastar la exposición al incumplimiento ($EAD$) en el otorgamiento vs. al impago, cuantificando el impacto económico en la cartera de prueba.
7. **Modo demostración en vivo:** permitir la ejecución del experimento en menos de 2 minutos con un único comando.

---

## 2. Definición y justificación de la variable objetivo (LGD)

### 2.1. Definición matemática
La Pérdida Dado el Incumplimiento (*Loss Given Default*, LGD) mide la fracción económica de la deuda impagada que la entidad crediticia no logra recuperar tras agotar las gestiones de cobranza o liquidación judicial.

A partir del diccionario oficial `LCDataDictionary.xlsx` (hoja `LoanStats`) provisto en el dataset de Lending Club en Kaggle:
- `funded_amnt`: Monto total financiado y desembolsado por el prestamista al cliente.
- `total_rec_prncp`: Capital (principal) amortizado y pagado por el prestatario hasta la fecha.
- `recoveries`: Recuperaciones brutas obtenidas con posterioridad al castigo (*post charge-off gross recovery*).
- `collection_recovery_fee`: Comisiones y gastos pagados a empresas de cobranza judicial y extrajudicial (*post charge-off collection fee*).

Para cada crédito castigado $i$, la exposición en el momento del incumplimiento ($EAD_{obs, i}$) y la recuperación neta ($R_i$) se expresan formalmente como:

$$EAD_{obs, i} = \text{funded\_amnt}_i - \text{total\_rec\_prncp}_i$$

$$R_i = \text{recoveries}_i - \text{collection\_recovery\_fee}_i$$

Por consiguiente, la tasa de recuperación realizada es $RR_i = \frac{R_i}{EAD_{obs, i}}$, y la variable objetivo LGD resulta:

$$LGD_i = \text{clip}\left(1 - \frac{\text{recoveries}_i - \text{collection\_recovery\_fee}_i}{\text{funded\_amnt}_i - \text{total\_rec\_prncp}_i}, \, 0, \, 1\right)$$

### 2.2. Justificación de los créditos incluidos y excluidos
En los datos de Lending Club, solo los créditos con estado `Charged Off` y `Does not meet the credit policy. Status:Charged Off` poseen un proceso de recuperación completado o reconocido contablemente como fallido; su saldo vivo (`out_prncp`) es exactamente 0. Por el contrario:
- Los créditos `Default` (1.219 créditos) y `Late (31-120 days)` (11.591 créditos) aún se encuentran en mora activa, conservan saldos vivos pendientes (`out_prncp > 0`) y no han sido asignados a agencias de cobro final, por lo que su pérdida definitiva no se ha materializado. Incluirlos con recuperaciones parciales generaría un sesgo de subestimación artificial de la LGD.
- En consecuencia, el universo de modelado de LGD consta exactamente de **46.009 créditos castigados** con exposición residual positiva.

### 2.3. Limitaciones en el mundo real
1. **Ausencia de factor de descuento temporal (Recovery Discounting):** Los estándares de Basilea II/III (Comité de Supervisión Bancaria de Basilea, 2005) exigen descontar los flujos recuperados a la tasa efectiva original del préstamo para reflejar el costo de oportunidad del dinero durante el periodo de cobranza (que puede durar entre 12 y 36 meses). Los datos públicos de Lending Club no desglosan las fechas exactas de cada abono de recuperación, por lo que la fórmula calcula la pérdida nominal contable.
2. **Truncamiento en cohortes recientes:** Los créditos originados en años recientes (2014-2015) han tenido una ventana de maduración más corta para que las agencias cobren, provocando que la proporción de créditos con cero recuperación (`LGD = 1.0`) pase del 25% en 2010 al 62.7% en 2014.
3. **Valores atípicos contables:** En 38 casos, las recuperaciones brutas superaron el capital adeudado debido a recargos o penalizaciones moratorias no desglosadas; el recorte a 0 garantiza la coherencia física de la pérdida.

---

## 3. Fundamentación matemática de los modelos

### 3.1. Justificación de la función de activación de salida (Sigmoide)
La variable dependiente satisface $y_i \in [0, 1]$. El uso de una capa de salida lineal estándar $f(x) = w^T h + b$ puede producir salidas no admisibles ($\hat{y} < 0$ o $\hat{y} > 1$). 

Se introduce en la capa final del perceptrón multicapa la función de activación **Sigmoide logística**:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Propiedades que fundamentan su elección:
1. **Acotamiento intrínseco:** Su rango es estrictamente $(0, 1)$, mapeando cualquier activación pre-salida $z \in (-\infty, \infty)$ en una fracción válida de pérdida económica sin necesidad de truncamientos arbitrarios que anulen el gradiente.
2. **Diferenciabilidad y suavidad:** Su derivada analítica $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ es continua y garantiza gradientes estables en el entrenamiento.

### 3.2. Derivación de la función de pérdida a partir de Máxima Verosimilitud (MLE)
Consideremos que el valor observado de LGD condicional a las características de entrada $x_i$ sigue un modelo de regresión no lineal:

$$y_i = f_\theta(x_i) + \epsilon_i$$

Donde $\epsilon_i \overset{\text{iid}}{\sim} \mathcal{N}(0, \sigma^2)$ representa el error de estimación con distribución gaussiana. La densidad de probabilidad condicional de cada observación es:

$$p(y_i \mid x_i; \theta, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(y_i - f_\theta(x_i))^2}{2\sigma^2} \right)$$

Bajo el supuesto de independencia entre los $N$ préstamos del conjunto de entrenamiento, la función de verosimilitud de la muestra es:

$$L(\theta, \sigma^2) = \prod_{i=1}^N p(y_i \mid x_i; \theta, \sigma^2) = \left(2\pi\sigma^2\right)^{-N/2} \exp\left( -\frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - f_\theta(x_i))^2 \right)$$

Aplicando logaritmo natural para obtener la log-verosimilitud ($\ln L$):

$$\ln L(\theta, \sigma^2) = -\frac{N}{2} \ln(2\pi) - \frac{N}{2} \ln(\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - f_\theta(x_i))^2$$

Para estimar los parámetros $\theta$ de la red neuronal mediante el principio de Máxima Verosimilitud, maximizamos $\ln L(\theta)$, lo cual equivale a minimizar la pérdida por Máxima Verosimilitud Negativa (NLL):

$$\min_\theta -\ln L(\theta) \iff \min_\theta \frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - f_\theta(x_i))^2 \iff \min_\theta \frac{1}{N} \sum_{i=1}^N (y_i - f_\theta(x_i))^2$$

Queda así formalmente demostrado que el **Error Cuadrático Medio (MSE)** es el estimador de máxima verosimilitud para la media condicional bajo supuestos de perturbaciones gaussianas.

Adicionalmente, se exploró la quasi-verosimilitud binaria (**BCE**) propuesta por Papke y Wooldridge (1996) para respuestas fraccionarias $y \in [0, 1]$, cuya minimización es consistente para $E[y|x]$ incluso si la distribución real no es binomial:

$$\mathcal{L}_{\text{BCE}}(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^N \Big[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \Big]$$

### 3.3. Algoritmo de optimización: AdamW
Se utiliza el optimizador AdamW (Loshchilov & Hutter, 2019), el cual desacopla la regularización por decaimiento de pesos ($L_2$) del gradiente adaptativo de los momentos. Para el gradiente $g_t = \nabla_\theta \mathcal{L}_t(\theta_t)$, las ecuaciones de actualización son:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(primer momento)}$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(segundo momento)}$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{(corrección de sesgo)}$$

$$\theta_{t+1} = \theta_t - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right) \quad \text{(actualización desacoplada con decaimiento } \lambda)$$

### 3.4. Regularización
- **Dropout (Srivastava et al., 2014):** Durante el entrenamiento, las activaciones de neuronas intermedias se desactivan aleatoriamente con probabilidad $p=0.20$, forzando a la red a no depender de coadaptaciones espurias.
- **Parada Temprana (*Early Stopping*):** El error en validación se monitorea en cada época. Si la pérdida de validación no mejora tras una paciencia de 8 épocas, el entrenamiento se interrumpe y se restauran los pesos del mejor punto de control, previniendo el sobreajuste.

---

## 4. Resultados experimentales actualizados

Benchmark recalculado el 30/10/2026: cinco configuraciones, 23.85 s de búsqueda y treinta semillas (101–3030). Todos los modelos comparten matrices y particiones. La arquitectura elegida es [64, 32], con pérdida mse. El checkpoint 1212 se elige por RMSE de validación 0.093975; solo después se evalúan las corridas en prueba. El JSON registra en cada configuración paciencia, tamaño de lote, weight decay y activaciones, además de la configuración efectiva del benchmark. Detalle y política: [selección LGD](lgd-seleccion-validacion.md).

| Modelo | Parámetros | MAE media ± DE | MAE mín–máx | RMSE media ± DE | RMSE mín–máx |
|---|---:|---:|---:|---:|---:|
| Lineal | 45 | 0.077110 | — | 0.086769 | — |
| MLP, 30 semillas | 5185 | 0.077056 ± 0.000659 | 0.075657–0.078373 | 0.088701 ± 0.000619 | 0.087570–0.089977 |

Fuente: `artifacts/models/lgd_benchmark_results.json`, incluyendo `per_seed_metrics`.
El entrenamiento de las treinta corridas acumuló 128.21 s; no incluye la búsqueda. La desviación se calcula con ddof=0.

### 4.1 Distribución

La media observada es 0.945151; la media predicha del checkpoint seleccionado es 0.918765. La figura `reports/figures/distribucion_lgd_observada_vs_predicha.svg` se regeneró con ese modelo. Las métricas del checkpoint individual no se confunden con los promedios entre semillas.

## 5. Discusión del benchmark actualizado

El MLP tiene 5185 parámetros frente a 45 de la regresión lineal. Su MAE promedio es parecido y su RMSE promedio mayor; no hay ventaja clara que justifique afirmar superioridad de la red. Las predicciones de media condicional no necesariamente reproducen la distribución marginal de pérdidas. La inferencia causal o las explicaciones definitivas sobre el bajo poder predictivo requieren análisis adicional.

La selección de semilla se corrige para usar validación. Como prueba 2014 se consultó históricamente, la corrección no la convierte retrospectivamente en un holdout nuevo. Una evaluación externa requiere reservar otro periodo.

## 6. Pérdida Esperada ($E[L] = PD \times LGD \times EAD$)

### 6.1. Definición y justificación de EAD
En concordancia con los marcos de Basilea y el ciclo de vida del crédito:
- **EAD al Otorgamiento ($EAD_0$):** Al momento de suscripción, el momento del impago es incierto. Para créditos a plazo fijo de consumo, la exposición máxima contractual comprometida es $EAD_0 = \text{funded\_amnt}$.
- **EAD al Incumplimiento ($EAD_{obs}$):** Para la cartera en seguimiento moratorio, la exposición real no amortizada es $EAD_{obs} = \text{funded\_amnt} - \text{total\_rec\_prncp}$.

### 6.2. Alcance de la cuantificación

El benchmark actualizado evalúa LGD condicionada a créditos castigados. La integración versionada por ID puntuó los 235.628 préstamos originados en 2014 con PD y LGD reales, usando `funded_amnt` al otorgamiento. Sus cifras (pérdida esperada total de USD 774.349.671,31; USD 208.561.086,83 en la cohorte resuelta y USD 168.195.585,76 realizados) corresponden al checkpoint anterior, semilla 808, cuyo hash está registrado en `reports/lgd_evaluation/perdida_esperada_pd_lgd_2014.json`. El benchmark regenerado selecciona ahora la semilla 1212; el resultado de cartera debe recalcularse con `python -m credit_risk.lgd.portfolio --year 2014` cuando `loan.csv` esté disponible. La comparación sigue siendo exploratoria: PD no corresponde a un horizonte fijo y las pérdidas usan exposiciones de momentos distintos.


## 7. Instrucciones de reproducción, comandos en vivo y verificación de requisitos

### 7.1. Cumplimiento de requisitos de evaluación

| Requisito evaluado | Estado | Implementación y verificación técnica |
| :--- | :---: | :--- |
| **Cada experimento se lanza con un solo comando** | **Cumple** | `python -m credit_risk.lgd.experiments` o con `--demo`. Carga automáticamente las matrices versionadas en Git o las autogenera desde `loan.csv` si no existieran, sin pasos previos obligatorios. |
| **Parámetros modificables sin tocar código** | **Cumple en la implementación** | YAML y CLI permiten modificar paciencia (`early_stopping_patience` / `--patience`), tamaño de lote (`batch_size` / `--batch-size`) y decaimiento de pesos (`weight_decay` / `--weight-decay`); se propagan a la búsqueda y al entrenamiento final. La demo usa 15 épocas por defecto (`demo.epochs`) y permite cambiar el máximo con `--epochs`. Las pruebas cubren la resolución de épocas y el paso de parámetros a los modelos. |
| **Demostración en menos de 2 minutos** | **Cumple (Medido)** | **37,19 s** en clon limpio. Entorno medido: CPU AMD Ryzen 7 3700U (4c/8t), 6.7 GB RAM, Linux Debian x86_64. |
| **Cacería de la alucinación (15 %)** | **Cumple** | Consolidada en `docs/caceria_alucinaciones.md` y Sección 8 del reporte técnico, con aportes en LGD (H1, H2, H5). |

### 7.2. Comandos de ejecución

```bash
# 1. Demostración rápida en vivo (< 2 minutos, 37.19 s medidos)
PYTHONPATH=src python -m credit_risk.lgd.experiments --demo

# 2. Modificación en vivo (3 minutos): cambiar arquitectura y tasa de aprendizaje sin tocar código
PYTHONPATH=src python -m credit_risk.lgd.experiments --demo --hidden-dims 128,64 --lr 0.005

# 3. Benchmark completo de 30 semillas (usando configs/lgd_mlp_best.yaml)
PYTHONPATH=src python -m credit_risk.lgd.experiments --seeds 30

# 4. Generación explícita de matrices procesadas (autónoma y robusta ante datos atípicos)
PYTHONPATH=src python -m credit_risk.lgd.data

# 5. Suite completa de pruebas unitarias de LGD
pytest tests/test_lgd.py tests/test_lgd_checkpoint_integration.py -v

# 6. Regenerar integración PD × LGD × EAD por ID (requiere data/raw/loan/loan.csv)
PYTHONPATH=src python -m credit_risk.lgd.portfolio --year 2014
```

---

## 8. Cacería de la alucinación (Aportes del Integrante 3)

Sara Acevedo Maya identificó y documentó con evidencia empírica y fuentes primarias tres de los cinco hallazgos del equipo (ver `docs/caceria_alucinaciones.md`):
1. **Hallazgo H1 (Datos del mundo real):** Fuga de información en la fórmula de LGD sugerida por la IA (inclusión de variables ex-post como `recoveries` en $X$ y omisión de amortizaciones previas). Corregido con el diccionario oficial de Lending Club y normativa de Basilea II.
2. **Hallazgo H2 (Matemáticas):** Sugerencia errónea de usar Softmax escalar o capa lineal no acotada, afirmando falsamente que la Sigmoide no sirve para regresión. Corregido con la literatura de quasi-verosimilitud bernoulli (Papke & Wooldridge, 1996) y activación logística diferenciable.
3. **Hallazgo H5 (Conceptos):** Afirmación dogmática de que el MLP superaría drásticamente a la regresión lineal en datos tabulares de LGD. No se sostiene en este experimento: el MAE medio del MLP es muy similar y ligeramente mayor (0.077140 frente a 0.077110), y su RMSE medio también es mayor.

---

## 9. Referencias bibliográficas (Normas APA 7.ª edición)

- Basel Committee on Banking Supervision. (2005). *An explanatory note on the Basel II IRB risk-weight functions*. Bank for International Settlements. https://www.bis.org/bcbs/irbriskweight.pdf
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. En *International Conference on Learning Representations (ICLR)*. https://openreview.net/forum?id=Bkg6RiCqY7
- Papke, L. E., & Wooldridge, J. M. (1996). Econometric methods for fractional response variables with an application to 401(k) plan participation rates. *Journal of Applied Econometrics, 11*(6), 619–632. https://doi.org/10.1002/(SICI)1099-1255(199611)11:6<619::AID-JAE418>3.0.CO;2-1
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research, 15*(56), 1929–1958. https://www.jmlr.org/papers/v15/srivastava14a.html
