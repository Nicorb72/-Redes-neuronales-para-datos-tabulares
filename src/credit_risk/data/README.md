# Módulo de datos — Riesgo crediticio

## 1. Objetivo

Este módulo se encarga de cargar, explorar, preparar y transformar los datos de préstamos de Lending Club para el desarrollo de modelos predictivos de riesgo crediticio.

La implementación actual permite generar los conjuntos de entrenamiento, validación y prueba para la estimación de la probabilidad de incumplimiento (PD).

## 2. Dataset

Se utiliza el archivo `loan.csv`, que contiene 887.379 registros de préstamos originados entre junio de 2007 y diciembre de 2015.

El archivo original se ubica en `data/raw/loan/loan.csv` y no se incluye en GitHub debido a su tamaño.

### Procedencia y diccionario de variables

- **Fuente del conjunto:** *Credit Risk Analysis*, datos de préstamos de Lending Club publicados en [Kaggle](https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset).
- **Dimensiones originales:** 887.379 registros y 74 columnas en `loan.csv`.
- **Diccionario consultado:** `data/raw/LCDataDictionary.xlsx`, hoja `LoanStats`, suministrado con el conjunto de datos.
- **Fecha de descarga de la copia de referencia:** 24/10/2026, según el registro del equipo. No se registraron el identificador de versión ni el checksum de Kaggle; la fecha no se infiere de la modificación o compresión del archivo.

Las definiciones de las variables de la sección 5 se basan en la hoja `LoanStats` del diccionario suministrado.

## 3. Variable objetivo

La variable `default_flag` se construye a partir de `loan_status`:

- **0 — Buen pagador:** `Fully Paid` y `Does not meet the credit policy. Status:Fully Paid`.
- **1 — Mal pagador:** `Charged Off`, `Default`, `Late (31-120 days)` y `Does not meet the credit policy. Status:Charged Off`.
- **Sin etiqueta:** `Current`, `Issued`, `In Grace Period` y `Late (16-30 days)`.

Se obtienen 209.711 buenos pagadores, 58.819 malos pagadores y 618.849 créditos sin etiqueta. Para PD se utilizan 268.530 créditos etiquetados.

## 4. Particiones temporales

| Conjunto       | Originación | Registros |
| -------------- | ------------ | --------: |
| Entrenamiento  | 2007–2012   |    90.347 |
| Validación    | 2013         |    73.194 |
| Prueba         | 2014         |    74.154 |
| Complementario | 2015         |    30.835 |

### Justificación de la partición temporal

Se utilizó una partición temporal basada en el año de originación de los créditos, con el propósito de conservar el orden cronológico de los datos y evaluar el desempeño de los modelos en periodos posteriores a los utilizados durante el entrenamiento.

Los créditos originados entre 2007 y 2012 se asignaron al conjunto de entrenamiento, mientras que los correspondientes a 2013 se reservaron para validación y ajuste de hiperparámetros. El año 2014 se destinó a la evaluación final, manteniendo un conjunto de prueba independiente. Los créditos de 2015 se conservaron para un análisis complementario, debido a la elevada proporción de registros sin una clasificación definitiva.

Esta distribución permite aplicar una estrategia de evaluación común a los diferentes modelos, manteniendo separados los procesos de entrenamiento, validación y prueba.

Las particiones se realizan por año de originación. No representan una validación histórica estricta, ya que los estados finales de los créditos pueden haberse observado después de su fecha de originación.

## 5. Variables predictoras

Se utilizan 15 características antes de la codificación:

**Numéricas:** `loan_amnt`, `annual_inc`, `dti`, `delinq_2yrs`, `inq_last_6mths`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc` y `antiguedad_crediticia_meses`.

**Categóricas:** `term`, `purpose`, `emp_length` y `home_ownership`.

La antigüedad crediticia se calcula a partir de la diferencia en meses entre `earliest_cr_line` e `issue_d`.

### Diccionario de características seleccionadas

Las descripciones siguientes proceden de `LCDataDictionary.xlsx`, hoja `LoanStats`; se presenta una traducción resumida al español. Para la antigüedad crediticia se documenta una variable derivada por este proyecto.

| Variable | Significado de la fuente o definición derivada | Unidad o formato |
|---|---|---|
| `loan_amnt` | Monto del préstamo solicitado y listado; puede reflejar ajustes del área de crédito. | USD |
| `annual_inc` | Ingreso anual declarado por el solicitante. | USD/año |
| `dti` | Relación entre obligaciones de deuda mensuales y el ingreso mensual declarado, según la definición de Lending Club. | Razón, expresada en la escala porcentual del CSV |
| `delinq_2yrs` | Incidentes de morosidad de 30 días o más durante los últimos dos años. | Conteo |
| `inq_last_6mths` | Consultas crediticias durante los últimos seis meses, excluyendo las de autos e hipotecas. | Conteo |
| `open_acc` | Líneas de crédito abiertas registradas en el historial del solicitante. | Conteo |
| `pub_rec` | Registros públicos negativos del historial crediticio. | Conteo |
| `revol_bal` | Saldo total de crédito revolvente. | USD |
| `revol_util` | Utilización del crédito revolvente respecto al crédito disponible. | Porcentaje |
| `total_acc` | Total de líneas de crédito registradas en el historial. | Conteo |
| `antiguedad_crediticia_meses` | Meses entre `earliest_cr_line` (primera línea de crédito reportada) e `issue_d` (mes de financiación del préstamo). | Meses; variable derivada |
| `term` | Número de pagos/plazo del préstamo: 36 o 60 meses. | Categoría (meses) |
| `purpose` | Propósito del préstamo declarado por el solicitante. | Categoría |
| `emp_length` | Antigüedad laboral declarada, con categorías desde menos de un año hasta diez años o más. | Categoría (años) |
| `home_ownership` | Situación de tenencia de vivienda declarada al registrarse. | Categoría |

**Nota metodológica:** `issue_d` representa el mes de financiación según el diccionario; se emplea como referencia temporal para las particiones y para calcular la antigüedad crediticia. Las categorías efectivamente utilizadas en la codificación se aprenden del conjunto de entrenamiento.

## 6. Preprocesamiento

Las transformaciones incluyen:

- Imputación por mediana para características numéricas.
- Estandarización mediante `StandardScaler`.
- Imputación categórica con el valor `Desconocido`.
- Codificación de categorías mediante `OneHotEncoder`.

El preprocesador se ajusta exclusivamente con los registros de entrenamiento y posteriormente se aplica a los demás conjuntos.

Después de la transformación se obtienen 44 características.

## 7. Auditoría de calidad

No se identificaron identificadores de crédito duplicados.

La principal variable con valores faltantes es `emp_length`, con 44.825 registros faltantes (5,05 %) en el dataset original.

También se identificaron observaciones financieras extremas. Estas se conservan inicialmente para no descartar información potencialmente válida. Su efecto sobre los modelos debe evaluarse posteriormente.

## 8. Limitaciones

El 67,82 % de los créditos originales aparece como `Current`.

La proporción de créditos sin etiqueta aumenta considerablemente en los años recientes y alcanza el 92,68 % en 2015.

Esto puede introducir sesgo de selección en los conjuntos etiquetados y limita la interpretación de las tasas de incumplimiento por año.

La comprobación de fuga de información realizada excluye variables de resultados posteriores al desembolso, pero no constituye una auditoría completa de la disponibilidad temporal de cada dato.

## 9. Ejecución

Un clon limpio incluye las matrices, etiquetas y preprocesador de PD
versionados; por eso los experimentos PD y la aplicación pueden ejecutarse sin
`loan.csv`. El comando siguiente es distinto: reconstruye esas matrices desde
los registros originales y requiere el CSV en `data/raw/loan/loan.csv`.

Desde la raíz del proyecto, con las dependencias instaladas y el CSV ubicado en
la carpeta correspondiente, ejecutar:

```powershell
$env:PYTHONPATH = "src"
python -m credit_risk.data.prepare_data
```

Para regenerar las figuras que usa el reporte, ejecutar:

```powershell
$env:PYTHONPATH = "src"
python -m credit_risk.data.generate_eda_figures
```

Si `loan.csv` está disponible, el comando recalcula las figuras desde el archivo
original. Si no está, utiliza `data/processed/eda_pd_resumen.json`, versionado
con los conteos verificados, para generar las dos figuras de PD. En ambos casos,
guarda `distribucion_etiquetas_pd.svg` y `particiones_temporales_pd.svg` en
`reports/figures/`. La alternativa permite regenerar las figuras, pero no
reconstruir el preprocesamiento completo ni verificar conteos nuevos sin el CSV.

## 10. Archivos generados

**En `data/processed/`:**

- `X_train_pd.npz` y `y_train_pd.npy`
- `X_validation_pd.npz` y `y_validation_pd.npy`
- `X_test_pd.npz` y `y_test_pd.npy`
- `X_supplementary_2015_pd.npz` y `y_supplementary_2015_pd.npy`

**En `artifacts/models/`:**

- `pd_preprocessor.joblib`
- `pd_feature_names.json`

Las matrices se guardan en formato disperso y pueden cargarse mediante `scipy.sparse.load_npz`.

## 11. Integración con el equipo

El integrante 2 puede utilizar las matrices y etiquetas PD para entrenar y evaluar los modelos de clasificación.

La tarea LGD requiere una preparación adicional y una definición explícita de la pérdida observada. No debe asumirse que las matrices PD son directamente adecuadas para entrenar LGD.
