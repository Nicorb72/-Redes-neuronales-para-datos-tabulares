# Cacería de la Alucinación — Trabajo 2: Redes Neuronales para Datos Tabulares

**Curso:** Redes Neuronales y Algoritmos Bioinspirados  
**Semestre:** 2026-02  
**Universidad Nacional de Colombia — Facultad de Minas**  
**Componente de Evaluación:** Cacería de la Alucinación (15 % de la calificación grupal)  
**Fecha de consolidación:** Octubre de 2026  

---

## Introducción y Criterios Docentes

De acuerdo con las instrucciones de la Sección 7 del enunciado (*Trabajo 2: Redes neuronales para datos tabulares*):
> *"Las herramientas de IA producen errores que suenan convincentes. Su misión es cazarlos. Documenten al menos cinco hallazgos en los que una IA generó algo incorrecto, impreciso, desactualizado o subóptimo durante el desarrollo del trabajo. Deben cubrir al menos tres categorías (...). Para cada hallazgo reporten: prompt literal, respuesta relevante, motivo de sospecha, evidencia demostrativa, corrección y lección aprendida. Un hallazgo inventado o sin evidencia se penaliza más que un hallazgo faltante."*

Este documento consolida cinco hallazgos técnicos genuinos identificados durante las fases de diseño, preprocesamiento, modelado econométrico de LGD, calibración del MLP y contrastación empírica frente al baseline lineal, cubriendo las cinco categorías temáticas exigidas por la cátedra.

---

## Hallazgo 1: Categoría «Datos del Mundo Real y Negocio de Crédito»
### Fuga de información (*data leakage*) ex-post en la definición de variables y la fórmula de LGD

- **Responsable:** Sara Acevedo Maya (Integrante 3: LGD y Pérdida Esperada).
- **Categoría:** Datos del mundo real, definiciones del negocio de crédito y normativa bancaria.

#### 1. Prompt utilizado (literal)
> *"Explícame cómo calcular la variable objetivo LGD (Loss Given Default) a partir del dataset de Lending Club y qué variables predictoras debería incluir mi red neuronal para estimarla con alta precisión en clientes que piden crédito."*

#### 2. Respuesta relevante de la IA
> *"Para calcular la LGD, simplemente toma `1 - (recoveries / funded_amnt)`. Para predecir la LGD con alta precisión usando tu red neuronal, debes alimentar variables clave del préstamo como `loan_amnt`, `funded_amnt`, `total_pymnt`, `recoveries`, `collection_recovery_fee`, `last_pymnt_amnt` y `last_pymnt_d`. Estas variables tienen altísima correlación con el monto recuperado y permitirán que la red minimice drásticamente el error de regresión."*

#### 3. Cómo se sospechó del error
1. La fórmula propuesta por la IA divide las recuperaciones entre el monto inicial financiado (`funded_amnt`) sin restar las amortizaciones de capital que el cliente ya había pagado legalmente antes de caer en mora (`total_rec_prncp`). Si un cliente ya pagó el 90 % de su deuda y luego incumple, el banco no pierde sobre el 100 % del monto original, sino sobre el saldo insoluto al momento del impago.
2. Al sugerir incluir `recoveries`, `collection_recovery_fee` o `last_pymnt_amnt` como *variables predictoras* de entrada al modelo, la IA introduce una **fuga masiva de información (*data leakage*)**: esas variables solo se generan *meses o años después de que el préstamo ha entrado en impago*. En el momento de la originación (cuando el banco decide otorgar el crédito), estas variables son físicamente desconocidas y deben valer cero o no existir.

#### 4. Evidencia que lo demuestra
- **Fuente primaria oficial:** Hoja de cálculo `LCDataDictionary.xlsx` provista en Kaggle, pestaña `LoanStats`:
  - `funded_amnt`: *"The total amount committed to that loan at that point in time."* (Monto inicial financiado).
  - `total_rec_prncp`: *"Principal received to date."* (Amortización acumulada de capital).
  - `recoveries`: *"Post charge off gross recovery."* (Recuperación bruta generada **con posterioridad** al castigo del crédito).
  - `collection_recovery_fee`: *"Post charge off collection fee."* (Comisión pagada a casas de cobranza **tras** el castigo).
- **Normativa de Basilea II/III (Comité de Supervisión Bancaria de Basilea, 2005):** La exposición al incumplimiento ($EAD_{obs}$) es el capital residual insoluto en el evento del default:
  $$EAD_{obs} = \text{funded\_amnt} - \text{total\_rec\_prncp}$$
  Dividir las recuperaciones brutas entre `funded_amnt` sin deducir los honorarios de cobranza ni la amortización previa distorsiona gravemente la tasa de recuperación económica real ($RR$).

#### 5. Corrección implementada
1. Se definió la variable objetivo LGD respetando estrictamente los fondos netos recuperados y la exposición al incumplimiento real:
   $$LGD_i = \text{clip}\left( 1 - \frac{\text{recoveries}_i - \text{collection\_recovery\_fee}_i}{\text{funded\_amnt}_i - \text{total\_rec\_prncp}_i}, \, 0, \, 1 \right)$$
2. Se purgaron del espacio de características predictoras $X$ todas las variables generadas *ex-post* a la originación. El modelo de LGD solo utiliza las 15 características previas al desembolso (idénticas a las del modelo PD), preservando la regla de causalidad temporal y no anticipación.

#### 6. Lección aprendida
Las correlaciones elevadas reportadas por un modelo sobre datos tabulares con frecuencia señalan fuga de información (*target leakage*) y no verdadero poder predictivo. En riesgo de crédito, cualquier variable debe confrontarse con la línea temporal del negocio: **¿esta columna existía el día en que el cliente firmó el pagaré?** Si la respuesta es no, no puede ser una predictora.

---

## Hallazgo 2: Categoría «Matemáticas»
### Uso erróneo de capa de salida lineal sin acotamiento y función Softmax para regresión de LGD continua

- **Responsable:** Sara Acevedo Maya (Integrante 3: LGD y Pérdida Esperada).
- **Categoría:** Matemáticas: funciones de activación de salida y funciones de pérdida para variables fraccionarias acotadas.

#### 1. Prompt utilizado (literal)
> *"Diseña la arquitectura de una red neuronal en PyTorch para estimar LGD en riesgo de crédito. ¿Qué activación de salida debo usar si LGD es un porcentaje entre 0 y 1, y qué función de pérdida es la matemáticamente correcta?"*

#### 2. Respuesta relevante de la IA
> *"Para estimar LGD en PyTorch debes usar una capa lineal de salida estándar `nn.Linear(hidden_dim, 1)` sin ninguna activación, o bien aplicar una activación `nn.Softmax(dim=-1)` a la salida para asegurarte de que represente una distribución de probabilidad continua. Luego entrena con `nn.CrossEntropyLoss()` o `nn.MSELoss()`. Si usas Sigmoide vas a saturar los gradientes porque Sigmoide es solo para clasificación binaria booleana, no para regresión."*

#### 3. Cómo se sospechó del error
1. La función `Softmax` sobre un tensor unidimensional escalar $\mathbb{R}^1$ colapsa trivialmente a la constante $1.0$ ($\text{Softmax}([z]) = \frac{e^z}{e^z} = 1.0$), por lo que la red no podría aprender ningún valor distinto de 1.
2. Una capa lineal sin acotamiento produce salidas no restringidas en $(-\infty, \infty)$. En experimentos preliminares con regresión lineal no acotada, el modelo predijo valores de LGD negativos (ej. $-0.14$) o superiores a 1 (ej. $1.08$), lo que en finanzas equivale al absurdo de afirmar que el banco recuperó más del 100 % de la deuda o perdió más dinero del que prestó.
3. La afirmación de que la activación Sigmoide es «exclusiva para clasificación binaria» es matemáticamente falsa: la función logística $\sigma(z) = \frac{1}{1 + e^{-z}}$ es un biyección diferenciable de $\mathbb{R} \to (0, 1)$, ideal para acotar la salida de una regresión en el soporte físico unitario $[0, 1]$.

#### 4. Evidencia que lo demuestra
- **Derivación de salida acotada:** Para cualquier pre-activación $z \in \mathbb{R}$, $\lim_{z \to -\infty} \sigma(z) = 0$ y $\lim_{z \to \infty} \sigma(z) = 1$. Su derivada es analítica y suave: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.
- **Fundamento econométrico de Quasi-Verosimilitud:** Papke & Wooldridge (1996) (*Econometric methods for fractional response variables with an application to 401(k) plan participation rates*) demuestran formalmente que para variables dependientes continuas acotadas $y \in [0, 1]$, la combinación de enlace logístico $\hat{y} = \sigma(x^T \beta)$ con quasi-verosimilitud bernoulli / entropía cruzada binaria (BCE) produce estimadores asintóticamente consistentes de la media condicional $E[y|x]$, incluso cuando $y$ no sigue una distribución Bernoulli discreta.
- **Prueba unitaria:** Implementada en `tests/test_lgd.py::test_mlp_regressor_salida_sigmoide_acotada`. Al evaluar 80 observaciones sintéticas arbitrarias con valores extremos en $X$, el 100 % de las predicciones satisfacen $0 \le \hat{y}_i \le 1$.

#### 5. Corrección implementada
En `src/credit_risk/lgd/neural_network.py` (`MLPRegressor`):
```python
# Capa de salida con activación Sigmoide para acotamiento intrínseco en [0, 1]
layers.append(nn.Linear(prev_dim, 1))
if output_activation == "sigmoid":
    layers.append(nn.Sigmoid())
```
Adicionalmente, se implementó en `BaselineLGD` una proyección (*clip*) determinística a $[0, 1]$ para garantizar una comparación equitativa con el modelo de referencia.

#### 6. Lección aprendida
Las funciones de activación y pérdida deben elegirse con base en el soporte matemático de la variable dependiente y los principios de verosimilitud de la familia exponencial, no por analogías superficiales de bibliotecas de software.

---

## Hallazgo 3: Categoría «Código y Metodología Experimental»
### Fuga temporal en el preprocesamiento (*data leakage* al hacer `fit` en todo el dataset)

- **Responsable:** Equipo (Integración de datos PD y LGD).
- **Categoría:** Código: errores de implementación, preprocesamiento defectuoso y validación cruzada incorrecta.

#### 1. Prompt utilizado (literal)
> *"Escribe una función en Python usando Scikit-Learn para aplicar StandardScaler a las variables numéricas y OneHotEncoder a las categóricas sobre el DataFrame de préstamos antes de entrenar la red neuronal."*

#### 2. Respuesta relevante de la IA
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def preprocesar_y_dividir(df, numeric_cols, cat_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='first'), cat_cols)
        ]
    )
    # Ajustamos y transformamos todo el conjunto de datos
    X_processed = preprocessor.fit_transform(df[numeric_cols + cat_cols])
    y = df['default_flag'].values
    
    # Dividimos aleatoriamente en train y test
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
```

#### 3. Cómo se sospechó del error
1. La función aplica `preprocessor.fit_transform(...)` sobre **todo** el DataFrame `df` *antes* de realizar la partición. Esto significa que las medias $\mu$ y desviaciones estándar $\sigma$ de `StandardScaler`, así como las categorías de `OneHotEncoder`, se calculan utilizando información del conjunto de prueba y validación (fuga de información del futuro hacia el pasado).
2. Usa `train_test_split(..., random_state=42)` con partición aleatoria uniforme, ignorando la naturaleza cronológica de los préstamos. En riesgo de crédito, un banco en 2013 no puede evaluar a sus clientes utilizando datos aleatorios extraídos de préstamos de 2014 o 2015.

#### 4. Evidencia que lo demuestra
- **Protocolo de partición del equipo:** El flujo del proyecto usa una separación **temporal**: Entrenamiento (2007–2012), Validación (2013) y Prueba (2014). La consigna exige justificar la partición temporal, pero estos años concretos corresponden a la decisión metodológica del equipo.
- **Regla de comparación justa:** En `tests/test_preprocesamiento.py::test_transformador_solo_ajusta_con_train` se verifica formalmente que `preprocesador.fit()` se ejecute **únicamente** sobre la matriz de entrenamiento $X_{\text{train}}$. Cualquier categoría o media observada en 2013–2014 no debe influir en la transformación.

#### 5. Corrección implementada
En `src/credit_risk/data/prepare_data.py` y `src/credit_risk/lgd/data.py`:
```python
# Partición temporal estricta por año de originación (issue_year)
df_cast.loc[df_cast["issue_year"].between(2007, 2012), "split"] = "train"
df_cast.loc[df_cast["issue_year"] == 2013, "split"] = "validation"
df_cast.loc[df_cast["issue_year"] == 2014, "split"] = "test"

# Ajustar transformador EXCLUSIVAMENTE en train
preprocesador = construir_preprocesador()
X_trans["train"] = preprocesador.fit_transform(X_raw["train"])

# Solo transformar (sin fit) en validación y prueba
X_trans["validation"] = preprocesador.transform(X_raw["validation"])
X_trans["test"] = preprocesador.transform(X_raw["test"])
```

#### 6. Lección aprendida
Nunca debe ejecutarse un transformador de datos sobre el conjunto global previo a la partición. En datos estructurados temporales, la separación cronológica debe preceder a cualquier estimación de parámetros estadísticos.

---

## Hallazgo 4: Categoría «Bibliografía»
### Referencias académicas inventadas y DOI no resolubles (*hallucinated citations*)

- **Responsable:** Equipo e Integrante 5 (Revisión bibliográfica y rigor APA 7).
- **Categoría:** Bibliografía: referencias inexistentes, autores y años equivocados, DOI rotos.

#### 1. Prompt utilizado (literal)
> *"Dame 3 referencias académicas en formato APA 7 con su enlace DOI sobre el uso de redes neuronales profundas para la estimación de Loss Given Default (LGD) en crédito de consumo con datos de Lending Club."*

#### 2. Respuesta relevante de la IA
> 1. *Goodfellow, I., Bengio, Y., & Courville, A. (2020). Deep learning architectures for financial default and recovery estimation. Journal of Banking & Finance, 114, 105–118. https://doi.org/10.1016/j.jbankfin.2020.105789*  
> 2. *Hastie, T., Tibshirani, R., & Friedman, J. (2018). Non-linear neural scoring models for bounded tabular responses. Journal of Risk and Financial Management, 11(3), 45–59. https://doi.org/10.3390/jrfm11030045*  
> 3. *Zhang, L., & Wang, H. (2021). Multilayer perceptrons vs. fractional response models for credit recovery rate prediction. European Journal of Operational Research, 289(2), 654–669. https://doi.org/10.1016/j.ejor.2021.03.042*

#### 3. Cómo se sospechó del error
1. Ian Goodfellow y Yoshua Bengio son reconocidos autores del libro fundamental *Deep Learning* (MIT Press, 2016), pero nunca publicaron un artículo conjunto en 2020 en el *Journal of Banking & Finance* sobre LGD y recuperación de crédito.
2. Hastie, Tibshirani y Friedman son los autores del clásico libro *The Elements of Statistical Learning* (2001/2009); el título y revista indicados no coinciden con su producción bibliográfica.
3. Al consultar directamente los tres enlaces DOI en el navegador (`https://doi.org/...`), los dos primeros arrojaron un error HTTP 404 de DOI no encontrado en el directorio central de CrossRef, mientras que el tercer DOI resolvía a un artículo totalmente ajeno sobre logística portuaria y transporte marítimo.

#### 4. Evidencia que lo demuestra
- **Resolución oficial de identificadores digitales en CrossRef (doi.org):**
  - Consulta a `https://doi.org/10.1016/j.jbankfin.2020.105789`: Devuelve *DOI Not Found*.
  - Búsqueda en los índices Scopus y Web of Science: Ninguno de los tres títulos existe en la literatura científica internacional.

#### 5. Corrección implementada
Se eliminaron por completo las referencias alucinadas y se construyó una bibliografía 100 % verificable con DOI y enlaces oficiales a repositorios primarios:
- **Papke, L. E., & Wooldridge, J. M. (1996).** Econometric methods for fractional response variables with an application to 401(k) plan participation rates. *Journal of Applied Econometrics, 11*(6), 619–632. https://doi.org/10.1002/(SICI)1099-1255(199611)11:6<619::AID-JAE418>3.0.CO;2-1
- **Basel Committee on Banking Supervision. (2005).** *An explanatory note on the Basel II IRB risk-weight functions*. Bank for International Settlements. https://www.bis.org/bcbs/irbriskweight.pdf
- **Loshchilov, I., & Hutter, F. (2019).** Decoupled weight decay regularization. En *International Conference on Learning Representations (ICLR)*. https://openreview.net/forum?id=Bkg6RiCqY7
- **Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).** Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research, 15*(56), 1929–1958. https://www.jmlr.org/papers/v15/srivastava14a.html

#### 6. Lección aprendida
Nunca debe aceptarse una cita, autor, año o identificador DOI sugerido por un modelo de lenguaje sin abrirlo y verificarlo activamente en la base de datos de CrossRef o en el repositorio editorial de la revista.

---

## Hallazgo 5: Categoría «Conceptos»
### Afirmación dogmática sobre la superioridad universal de redes neuronales en datos tabulares de LGD

- **Responsable:** Sara Acevedo Maya (Integrante 3) y Equipo de Modelado.
- **Categoría:** Conceptos: afirmaciones falsas sobre las redes neuronales, su sesgo inductivo y su desempeño real en datos tabulares.

#### 1. Prompt utilizado (literal)
> *"¿Por qué una red neuronal profunda MLP es categóricamente superior a una regresión lineal para predecir la pérdida dado el incumplimiento (LGD) en datos tabulares de crédito?"*

#### 2. Respuesta relevante de la IA
> *"Una red neuronal profunda MLP siempre supera ampliamente a los modelos lineales tradicionales como la regresión lineal en datos tabulares. Gracias al Teorema de Aproximación Universal, la red es capaz de capturar todas las complejas interacciones no lineales entre ingresos, empleo y deuda, lo que garantiza una reducción drástica del MAE y del RMSE frente a cualquier baseline lineal. Si tu baseline obtiene métricas similares a la red, significa que la red está mal entrenada o le faltan capas."*

#### 3. Cómo se sospechó del error
1. La literatura reciente sobre aprendizaje profundo en datos tabulares (ej. Shwartz-Ziv & Armon, 2022; Grinsztajn et al., 2022; Borisov et al., 2022) demuestra empíricamente que las redes neuronales no son inductivamente superiores a los modelos tabulares clásicos cuando la relación señal-ruido es moderada o baja.
2. En la tarea de LGD, las variables financieras observables al momento de originar el crédito (ingreso, DTI, empleo) ofrecen muy poca información explicativa sobre el monto que una agencia de cobro logrará recuperar *años después* tras agotar instancias judiciales. Por ende, la hipótesis de que mayor complejidad arquitectónica generaría un salto cualitativo en desempeño era altamente dudosa.

#### 4. Evidencia que lo demuestra
- **Resultados empíricos del experimento sobre la partición de prueba (Año 2014, $N=13.981$ préstamos castigados):**
  - **Regresión Lineal (Baseline):** 45 parámetros entrenables | Tiempo de ajuste: **0,040 s** | **MAE = 0,07711** | **RMSE = 0,08677**.
  - **Red Neuronal MLP (30 semillas independientes):** 5.185 parámetros entrenables | Búsqueda de cinco configuraciones: **31,98 s** | Ajuste acumulado de 30 corridas: **165,07 s** | **MAE = 0,07714 ± 0,00074** | **RMSE = 0,08876 ± 0,00070**.
  - **Comparación de complejidad:** El MLP posee aproximadamente **115 veces más parámetros**; los MAE y RMSE medios en test son ligeramente mayores que los del baseline. Los tiempos comparan un ajuste lineal con la búsqueda y 30 ajustes neuronales, por lo que no son tiempos por modelo equivalentes.
  - **Distribución de predicciones:** El baseline predice con media 0,9173 y desviación 0,0143. El MLP guardado, seleccionado por RMSE de validación (semilla 808), tiene media 0,9184 y desviación 0,0237. Ambas distribuciones son más estrechas que la LGD observada, con media 0,9452 y 62,74 % de valores iguales a 1.

#### 5. Corrección implementada
Se rechazó la afirmación dogmática de la IA en la sección de discusión del reporte técnico (Sección 2.3 en `reports/technical-report/index.md` y en su versión HTML). La corrida completa y registrada de 30 semillas no respalda que el MLP supere al baseline en esta tarea y partición. La conclusión se limita a estas métricas y datos; no se generaliza a todo uso bancario ni a todos los modelos tabulares.

#### 6. Lección aprendida
La capacidad matemática de aproximación universal no equivale a eficiencia estadística de generalización. En datos tabulares con variables de bajo poder explicativo ex-ante, la complejidad adicional de una red neuronal solo incrementa el costo computacional y la varianza entre semillas, sin mejorar las decisiones financieras del negocio.

---

## Tabla Resumen de Cumplimiento de la Rúbrica

| N.º | Categoría cubierta | Integrante | Error cazado de la IA | Evidencia primaria utilizada | Corrección implementada |
| :---: | :--- | :---: | :--- | :--- | :--- |
| **H1** | **Datos del mundo real y negocio de crédito** | Sara Acevedo (Int. 3) | Sugirió incluir variables ex-post (`recoveries`, `last_pymnt`) en $X$ y omitió amortización previa en LGD. | Diccionario `LCDataDictionary.xlsx` y normativa de Basilea II/III (BCBS, 2005). | Derivación con $EAD_{obs} = \text{funded} - \text{prncp}$; purga estricta de variables posteriores al desembolso. |
| **H2** | **Matemáticas y funciones de pérdida** | Sara Acevedo (Int. 3) | Propuso Softmax unidimensional o salida lineal no acotada; afirmó que Sigmoide no sirve para regresión. | Papke & Wooldridge (1996) sobre respuestas fraccionarias; derivación de rango $(0, 1)$ de $\sigma(z)$. | Capa final con activación Sigmoide intrínseca y prueba unitaria de acotamiento en $[0, 1]$. |
| **H3** | **Código y metodología experimental** | Equipo (Int. 1 y 2) | Aplicó `fit_transform` sobre todo el dataset antes de dividir y propuso división aleatoria temporalmente ciega. | Regla de no anticipación temporal; prueba `test_preprocesamiento.py::test_transformador_solo_ajusta_con_train`. | Partición temporal estricta (2007-2012, 2013, 2014); ajuste de transformadores exclusivamente en `train`. |
| **H4** | **Bibliografía y rigor académico** | Equipo (Int. 5) | Generó tres referencias académicas inventadas con autores falsos y códigos DOI que arrojaban HTTP 404. | Verificación directa en el registro central de CrossRef (`doi.org`) e índices de Scopus. | Eliminación de citas espurias; adopción de fuentes primarias reales verificadas (Basilea, Loshchilov, Papke). |
| **H5** | **Conceptos de redes en datos tabulares** | Sara Acevedo (Int. 3) | Afirmó que el MLP siempre supera por amplio margen a un modelo lineal en datos tabulares de LGD. | Benchmark registrado: 5 configuraciones y 30 semillas; baseline MAE/RMSE 0,07711/0,08677; MLP 0,07714 ± 0,00074 / 0,08876 ± 0,00070. | La afirmación no se sostiene en esta tarea y partición; el MLP no mejora las métricas del baseline. |
