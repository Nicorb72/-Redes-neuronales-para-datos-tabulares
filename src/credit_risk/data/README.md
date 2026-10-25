# Módulo de datos — Riesgo crediticio

## 1. Objetivo

Este módulo se encarga de cargar, explorar, preparar y transformar los datos de préstamos de Lending Club para el desarrollo de modelos predictivos de riesgo crediticio.

La implementación actual permite generar los conjuntos de entrenamiento, validación y prueba para la estimación de la probabilidad de incumplimiento (PD).

## 2. Dataset

Se utiliza el archivo `loan.csv`, que contiene 887.379 registros de préstamos originados entre junio de 2007 y diciembre de 2015.

El archivo original se ubica en `data/raw/loan/loan.csv` y no se incluye en GitHub debido a su tamaño.

## 3. Variable objetivo

La variable `default_flag` se construye a partir de `loan_status`:

- **0 — Buen pagador:** `Fully Paid` y `Does not meet the credit policy. Status:Fully Paid`.
- **1 — Mal pagador:** `Charged Off`, `Default`, `Late (31-120 days)` y `Does not meet the credit policy. Status:Charged Off`.
- **Sin etiqueta:** `Current`, `Issued`, `In Grace Period` y `Late (16-30 days)`.

Se obtienen 209.711 buenos pagadores, 58.819 malos pagadores y 618.849 créditos sin etiqueta. Para PD se utilizan 268.530 créditos etiquetados.

## 4. Particiones temporales

| Conjunto | Originación | Registros |
|---|---|---:|
| Entrenamiento | 2007–2012 | 90.347 |
| Validación | 2013 | 73.194 |
| Prueba | 2014 | 74.154 |
| Complementario | 2015 | 30.835 |

Las particiones se realizan por año de originación. No representan una validación histórica estricta, ya que los estados finales de los créditos pueden haberse observado después de su fecha de originación.

## 5. Variables predictoras

Se utilizan 15 características antes de la codificación:

**Numéricas:** `loan_amnt`, `annual_inc`, `dti`, `delinq_2yrs`, `inq_last_6mths`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc` y `antiguedad_crediticia_meses`.

**Categóricas:** `term`, `purpose`, `emp_length` y `home_ownership`.

La antigüedad crediticia se calcula a partir de la diferencia en meses entre `earliest_cr_line` e `issue_d`.

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

Desde la raíz del proyecto, con las dependencias instaladas y `loan.csv` ubicado en la carpeta correspondiente, ejecutar:

```powershell
$env:PYTHONPATH = "src"
python -m credit_risk.data.prepare_data
```

## 10. Archivos generados

**En `data/processed/`:**

- `X_train_pd.npz` y `y_train_pd.npy`
- `X_validation_pd.npz` y `y_validation_pd.npy`
- `X_test_pd.npz` y `y_test_pd.npy`
- `X_supplementary_2015_pd.npz` y `y_supplementary_2015_pd.npy`

**En `artifacts/models/`:**

- `preprocesador_pd.joblib`
- `columnas_pd.json`

Las matrices se guardan en formato disperso y pueden cargarse mediante `scipy.sparse.load_npz`.

## 11. Integración con el equipo

El integrante 2 puede utilizar las matrices y etiquetas PD para entrenar y evaluar los modelos de clasificación.

La tarea LGD requiere una preparación adicional y una definición explícita de la pérdida observada. No debe asumirse que las matrices PD son directamente adecuadas para entrenar LGD.