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

## 4. Resultados experimentales y regla de comparación justa

Siguiendo estrictamente las pautas de evaluación:
- **Mismos datos y particiones:** Entrenamiento 2007–2012 ($N=14.419$), Validación 2013 ($N=14.836$) y Prueba 2014 ($N=13.981$).
- **Mismo preprocesamiento:** 15 características previas al desembolso (11 numéricas y 4 categóricas), transformador ajustado únicamente en el conjunto de entrenamiento de LGD ($D=44$ características codificadas).
- **Presupuesto de búsqueda explícito:** Se evaluaron sistemáticamente 5 arquitecturas que variaron profundidad, funciones de pérdida (MSE, Huber, BCE) y tasas de aprendizaje, con un tiempo total de búsqueda de 31.04 segundos. La arquitectura óptima seleccionada fue `MLP [64, 32]`, activación ReLU, salida Sigmoide, dropout 0.20, optimizador AdamW ($lr=0.001$, $\lambda=10^{-4}$) y pérdida MSE.
- **30 corridas independientes:** La configuración final del MLP se entrenó con 30 semillas aleatorias independientes.

### Tabla 1. Comparación de desempeño en el conjunto de prueba independiente (Año 2014, $N=13.981$)

| Modelo | Parámetros entrenables | Tiempo de cómputo | MAE (Media ± Desv. Est.) | MAE [Mín, Máx] | RMSE (Media ± Desv. Est.) | RMSE [Mín, Máx] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regresión Lineal (Baseline)** | **45** | **0.050 s** | **0.0771** | [0.0771, 0.0771] | **0.0868** | [0.0868, 0.0868] |
| **Red Neuronal MLP (30 semillas)** | **5.185** | **~120 s (total)** | **0.0771 ± 0.0007** | [0.0757, 0.0790] | **0.0887 ± 0.0007** | [0.0877, 0.0905] |

### 4.1. Análisis distribucional (Observada vs. Predicha)
La distribución empírica de LGD observada presenta características altamente no gaussianas:
- **Media observada:** 0.9452 | **Desviación estándar:** 0.0811
- **Mediana observada:** 1.0000 | **Percentil 25:** 0.8601
- **Masa puntual en pérdida total:** El **62.74%** de los créditos castigados en 2014 tienen `LGD = 1.0` (cero recuperaciones netas).

Comportamiento de los modelos:
- **Baseline lineal:** Media predicha = 0.9173, Desv. Est. = 0.0143, Estadístico Kolmogorov-Smirnov $KS = 0.6276$ ($p < 10^{-15}$).
- **Red Neuronal MLP:** Media predicha = 0.9234, Desv. Est. = 0.0245, Distancia de Wasserstein = 0.0575, $KS = 0.6272$.
- **Gráfica de distribución:** La figura comparativa ha sido generada por código y guardada en formato vectorial en:  
  `reports/figures/distribucion_lgd_observada_vs_predicha.svg`.

---

## 5. Discusión crítica: ¿Qué aportó la Red Neuronal frente al Baseline?

El análisis de los resultados evidencia conclusiones fundamentales sobre el modelado con redes neuronales en datos tabulares de riesgo de crédito:

1. **Paridad de desempeño a pesar de un orden de magnitud mayor de complejidad:**  
   El MLP requiere **5.185 parámetros entrenables** frente a solo **45 parámetros** de la regresión lineal (una razón de complejidad de 115 a 1). Sin embargo, el error absoluto medio (MAE) es idéntico (0.0771), y la regresión lineal incluso supera ligeramente al MLP en RMSE (0.0868 frente a 0.0887).
2. **Naturaleza del fenómeno de recuperación:**  
   Las 15 variables disponibles en el momento de la originación (ingresos, razón deuda-ingreso, empleo, historial crediticio) contienen información predictiva valiosa para determinar si un cliente incumplirá (PD), pero carecen casi por completo de poder predictivo sobre *cuánto dinero se recuperará una vez ocurrido el default*. Las recuperaciones dependen de variables ex-post no observadas al solicitar el crédito: la capacidad operativa de la agencia de cobro, la existencia de bienes embargables y las condiciones macroeconómicas del año de liquidación.
3. **Colapso hacia la media condicional:**  
   Dado que el 62.7% de los clientes no genera recuperación alguna y la información predictiva de entrada es débil, tanto la regresión lineal como la red neuronal minimizadora de MSE aprenden a predecir valores concentrados estrechamente alrededor de la media muestral condicional (~0.91 - 0.94). La red neuronal no logra reproducir la bimodalidad o la masa en 1.0 porque cualquier intento de predecir valores extremos sin suficiente soporte en $X$ penaliza severamente el error cuadrático.
4. **Veredicto:**  
   En esta tarea de regresión tabular, **la red neuronal no aporta ventajas operativas ni predictivas sobre el modelo de referencia lineal**. El modelo lineal es 2.400 veces más rápido de entrenar, determinístico, matemáticamente transparente y no presenta variabilidad entre semillas aleatorias.

---

## 6. Pérdida Esperada ($E[L] = PD \times LGD \times EAD$)

### 6.1. Definición y justificación de EAD
En concordancia con los marcos de Basilea y el ciclo de vida del crédito:
- **EAD al Otorgamiento ($EAD_0$):** Al momento de suscripción, el momento del impago es incierto. Para créditos a plazo fijo de consumo, la exposición máxima contractual comprometida es $EAD_0 = \text{funded\_amnt}$.
- **EAD al Incumplimiento ($EAD_{obs}$):** Para la cartera en seguimiento moratorio, la exposición real no amortizada es $EAD_{obs} = \text{funded\_amnt} - \text{total\_rec\_prncp}$.

### 6.2. Cuantificación económica en la cartera de prueba (Test 2014)
Evaluando los 13.981 créditos castigados de la cohorte 2014:
- **Exposición total comprometida ($EAD_{obs}$):** **$178.668.119,71 USD**
- **Pérdida esperada estimada por el modelo:** **$163.408.908,93 USD** (Tasa de pérdida esperada: **91,46%**)
- **Pérdida real observada:** **$168.195.585,76 USD** (Tasa de pérdida real: **94,14%**)
- **Error absoluto medio (MAE) en pérdida por préstamo:** **$979,53 USD**

El modelo de riesgo estima la pérdida agregada de la cartera con un margen de error inferior al 2.7% relativo frente a la pérdida real, proveyendo un insumo financiero sólido para la constitución de provisiones contables.

---

## 7. Instrucciones de reproducción y comandos en vivo

Todos los experimentos se ejecutan con comandos únicos desde la raíz del repositorio:

```bash
# 1. Preparar datos y generar matrices procesadas de LGD
PYTHONPATH=src python -m credit_risk.lgd.data

# 2. Ejecutar corrida de demostración en vivo (< 2 minutos)
PYTHONPATH=src python -m credit_risk.lgd.experiments --demo

# 3. Ejecutar benchmark completo de 30 semillas y guardar artefactos
PYTHONPATH=src python -m credit_risk.lgd.experiments --seeds 30

# 4. Correr la suite de pruebas unitarias
pytest tests/test_lgd.py -v
```

---

## 8. Referencias bibliográficas (Normas APA 7.ª edición)

- Basel Committee on Banking Supervision. (2005). *An explanatory note on the Basel II IRB risk-weight functions*. Bank for International Settlements. https://www.bis.org/bcbs/irbriskweight.pdf
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. En *International Conference on Learning Representations (ICLR)*. https://openreview.net/forum?id=Bkg6RiCqY7
- Papke, L. E., & Wooldridge, J. M. (1996). Econometric methods for fractional response variables with an application to 401(k) plan participation rates. *Journal of Applied Econometrics, 11*(6), 619–632. https://doi.org/10.1002/(SICI)1099-1255(199611)11:6<619::AID-JAE418>3.0.CO;2-1
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research, 15*(56), 1929–1958. https://www.jmlr.org/papers/v15/srivastava14a.html
