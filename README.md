# Riesgo de crédito con redes neuronales para datos tabulares

Proyecto académico para estimar la probabilidad de incumplimiento (PD) y la
pérdida dado incumplimiento (LGD) en préstamos de consumo de Lending Club.
Incluye comparación de modelos, scorecard, interpretación y aplicación web.

## Resultados principales

El conjunto Credit Risk Analysis de Kaggle contiene 887.379 préstamos
originados entre junio de 2007 y diciembre de 2015. El conjunto de datos
original se utiliza desde `data/raw/loan/loan.csv`, se descargó el 24 de
octubre de 2026 y no se versiona en Git.
El tratamiento de PD parte de 15 variables originales y genera 44 variables
después de la imputación, estandarización y codificación one-hot. Las
particiones temporales documentadas son train 2007–2012 (90.347 registros),
validación 2013 (73.194) y prueba 2014 (74.154).

En la prueba PD de 2014, el MLP ejecutado con 30 semillas obtuvo AUC
promedio de 0,6694 ± 0,0043 y Brier de 0,1818 ± 0,0010. La regresión
logística obtuvo AUC de 0,6648 y Brier de 0,1872. El modelo final guardado
corresponde a la semilla 202612.

El benchmark LGD completo usa 13.981 créditos castigados de prueba 2014 y
30 semillas. La regresión lineal obtiene MAE 0.07711 y RMSE
0.08677; el MLP obtiene MAE 0.07714 ± 0.00074
y RMSE 0.08876 ± 0.00070. El checkpoint guardado, semilla
808, se selecciona por el menor RMSE de validación, sin usar test para
seleccionar. Las métricas por semilla y la decisión quedan en
`artifacts/models/lgd_benchmark_results.json` y se explican en
[selección LGD](docs/lgd-seleccion-validacion.md). Estas métricas evalúan LGD
condicionada al incumplimiento; la integración ex-ante se resume a continuación.

La integración PD × LGD × EAD sí se ejecutó para los préstamos originados en
2014: se puntuaron 235.628 filas y la pérdida esperada exploratoria fue
USD 774.349.671,31. El detalle por préstamo y el resumen están en
`reports/lgd_evaluation/`; la PD no tiene un horizonte fijo, así que la cifra
no es una provisión regulatoria.

La scorecard usa puntaje base 600, odds buenas:malas de 50:1 y PDO de 20.
La aplicación presenta PD, puntaje, percentil respecto a la referencia de
train y simulación de cambios en las variables.

## Enlaces

- [Reporte técnico tipo blog](https://nicorb72.github.io/-Redes-neuronales-para-datos-tabulares/reports/technical-report/)
- [Video promocional](https://youtu.be/TijgcPZNboU)
- [Aplicación web](https://rna-trabajo-2.streamlit.app/)
- [Repositorio en GitHub](https://github.com/Nicorb72/-Redes-neuronales-para-datos-tabulares)
- [Conjunto de datos en Kaggle](https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset)

## Requisitos

- Python 3.11.9, especificado en `.python-version`.
- Git.
- El CSV original se necesita para reconstruir matrices desde la fuente (`prepare_data` o
  `lgd.data`) y para regenerar la integración PD–LGD por préstamo. No está versionado por su
  tamaño y debe obtenerse por separado.

**Un clon limpio permite ejecutar la aplicación y los experimentos PD/LGD sobre matrices
versionadas sin descargar `loan.csv`**. Para reconstruir matrices desde la fuente o volver a
generar la integración PD–LGD por ID sí se requiere el CSV. El resumen
`data/processed/eda_pd_resumen.json` permite regenerar las figuras EDA sin depender de él.

## Instalación

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS y Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Los siguientes comandos se ejecutan desde la raíz con el entorno activado.

## Ejecución

### 1. Clonar el repositorio

Desde la carpeta donde quieras guardar el proyecto:

**Windows PowerShell**

```powershell
git clone https://github.com/Nicorb72/-Redes-neuronales-para-datos-tabulares.git
Set-Location .\-Redes-neuronales-para-datos-tabulares
```

**macOS/Linux**

```bash
git clone https://github.com/Nicorb72/-Redes-neuronales-para-datos-tabulares.git
cd ./-Redes-neuronales-para-datos-tabulares
```

Con Python 3.11 instalado, crea el entorno virtual e instala las dependencias
siguiendo los comandos de la sección [Instalación](#instalación). Mantén esa
terminal activa o vuelve a activar el entorno virtual si abres otra.

Ejecuta los siguientes comandos desde la raíz del repositorio y con el entorno
virtual activado. Los módulos bajo `src/` necesitan que esa carpeta esté en
`PYTHONPATH`.

**Windows PowerShell**

```powershell
$env:PYTHONPATH = "src"
```

**macOS/Linux**

```bash
export PYTHONPATH=src
```

### 2. Comprobar el clon y abrir la aplicación

El clon incluye los artefactos PD y las matrices procesadas; no necesita
`loan.csv` para estos pasos. Valida primero el paquete y luego inicia la app:

```console
python run_app.py --check
python run_app.py
```

El primer comando debe mostrar `PD listo`. La aplicación estará disponible en
`http://127.0.0.1:8501`; detenla con `Ctrl+C`. Para usar otro puerto, ejecuta
`python run_app.py --port 8502`.

### 3. Regenerar las figuras EDA

El comando usa `loan.csv` si está disponible; de lo contrario, utiliza los
conteos guardados en `data/processed/eda_pd_resumen.json`. En ambos casos
genera las dos figuras principales de PD en `reports/figures/`.

```console
python -m credit_risk.data.generate_eda_figures
```

### 4. Ejecutar los experimentos

Los experimentos usan las matrices versionadas y no requieren el CSV. El
modo completo de PD y LGD evalúa 30 semillas; `--demo` ejecuta una corrida
reducida para una comprobación rápida. Los hiperparámetros se encuentran en
`configs/pd_mlp_best.yaml` y `configs/lgd_mlp_best.yaml`.

**PD — corrida completa o demo**

```console
python -m credit_risk.pd.experiments
python -m credit_risk.pd.experiments --demo
```

**LGD — corrida completa, demo o parámetros alternativos**

```console
python -m credit_risk.lgd.experiments --seeds 30
python -m credit_risk.lgd.experiments --demo
python -m credit_risk.lgd.experiments --demo --hidden-dims 128,64 --lr 0.005 --epochs 20
```

La corrida completa de PD puede tardar varios minutos. El experimento LGD
reconstruye sus matrices si faltan y encuentra `loan.csv`; en ese caso, sí
necesitarás el archivo en `data/raw/loan/loan.csv`. Los resultados LGD se
guardan en `artifacts/models/` y las figuras en `reports/figures/`.

**Integración PD × LGD × EAD por préstamo**

Con `loan.csv` en `data/raw/loan/loan.csv`, este comando vuelve a puntuar la
cohorte de originación 2014 con los modelos guardados y regenera el CSV de
predicciones y su resumen en `reports/lgd_evaluation/`. El resumen registra
la ruta y el SHA-256 de la copia del CSV utilizada:

```console
python -m credit_risk.lgd.portfolio --year 2014
```

### 5. Actualizar el paquete PD después de entrenar

Después de una corrida completa de PD —descrita en el paso 4—, reconstruye el
manifiesto antes de iniciar la app. La exportación calcula los hashes y valida
el paquete. Este paso también es necesario después de reconstruir las matrices
PD en el paso 6.

```console
python -m credit_risk.scorecard.export --model artifacts/models/pd_mlp_final.pt --model-format pd_mlp --model-id pd_mlp_final-seed202612 --training-data "Matrices PD train 2007-2012" --reference-population "90347 creditos etiquetados de train 2007-2012"
python run_app.py --check
```

### 6. Reconstruir matrices desde el CSV original (opcional)

Solo necesitas este paso si quieres rehacer las matrices desde los registros
originales. Descarga `loan.csv` desde [Credit Risk Analysis en
Kaggle](https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset) y
colócalo en `data/raw/loan/loan.csv`. El resumen EDA no contiene los registros
ni las variables predictoras. La copia de referencia se descargó el
24/10/2026, pero en ese momento no se registraron su versión ni checksum. La
integración PD–LGD guarda en su JSON el SHA-256 de la copia local usada para
esa ejecución; conserva además la procedencia, pues otra descarga podría
producir resultados distintos.

```console
python -m credit_risk.data.prepare_data
```

El comando recalcula las cuatro particiones PD y sobrescribe matrices,
etiquetas y preprocesador. El preprocesador se ajusta solo con train. Después,
ejecuta la corrida completa de PD del paso 4 y actualiza el paquete según el
paso 5. Para preparar las matrices LGD explícitamente, ejecuta
`python -m credit_risk.lgd.data`.

### 7. Interpretabilidad PD

Con el paquete PD y las matrices y etiquetas de validación disponibles:

```console
python -m credit_risk.scorecard.interpretability --seed 42 --repeats 10 --max-rows 5000
```

Los resultados se guardan en `reports/interpretability/` como JSON, CSV y SVG.

### 8. Pruebas y lint

Instala las dependencias de desarrollo y ejecuta la suite desde la raíz:

```console
python -m pip install -r requirements-test.txt
python -m pytest -v
python -m ruff check src tests
```

## Datos, modelos y artefactos

`data/raw/` se excluye de Git por tamaño. El repositorio incluye matrices
procesadas y artefactos de PD y LGD para reproducir los experimentos sin el
CSV, además del paquete PD para la aplicación. No incluyas credenciales,
tokens, datos personales ni archivos temporales en el repositorio.

## Cacería de la alucinación

Los hallazgos se documentan en la sección 8 del [reporte técnico](reports/technical-report/index.md)
y en [`docs/caceria_alucinaciones.md`](docs/caceria_alucinaciones.md).

## Estructura del repositorio

```text
configs/                  Configuración PD, scorecard y aplicación
data/raw/                 CSV original local, excluido de Git
data/processed/           Matrices y etiquetas PD versionadas
src/credit_risk/data/     Carga, preparación, particiones y figuras EDA
src/credit_risk/pd/       Clasificación de probabilidad de incumplimiento
src/credit_risk/lgd/      Objetivo y modelos de pérdida dado incumplimiento
src/credit_risk/scorecard/Scorecard, exportación e interpretación
src/credit_risk/app/      Aplicación web
artifacts/models/         Modelos PD, preprocesador y contrato de inferencia
reports/                  Evaluaciones, figuras y blog técnico
tests/                    Pruebas automatizadas
```

## Equipo

| Integrante | Responsabilidad |
|---|---|
| Nicole Guaranguay | Datos y análisis |
| Nicolas Ruiz | Clasificación y PD |
| Sara Acevedo | Regresión y LGD |
| Sebastián Cataño | Scorecard, interpretación y aplicación web |
| Ivan Dario Gomez Cabrera | Integración, reproducibilidad y reporte |
