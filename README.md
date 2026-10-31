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

Para LGD, los resultados reportados en `docs/integrante-3.md` corresponden a
13.981 préstamos castigados en la prueba de 2014. La regresión lineal obtuvo
MAE 0,0771 y RMSE 0,0868; el MLP de 30 semillas obtuvo MAE 0,0771 ± 0,0007
y RMSE 0,0887 ± 0,0007. Estos resultados describen pérdidas condicionadas a
incumplimiento; no corresponden a una estimación ex-ante de PD × LGD × EAD.

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
- El CSV original solo es necesario si se desea reconstruir desde cero las matrices desde la
  fuente (`prepare_data` o `lgd.data`). El archivo no está versionado por su tamaño y debe
  descargarse por separado.

**Un clon limpio permite ejecutar la aplicación y los experimentos de PD y LGD sin
descargar `loan.csv`**, porque incluye matrices procesadas y artefactos de ambos
modelos versionados en el repositorio. Esto reproduce el flujo completo desde esos artefactos. El resumen `data/processed/eda_pd_resumen.json`
permite regenerar las figuras EDA sin depender del CSV original.

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

### Flujo recomendado: reproducir desde un clon limpio, sin CSV

Este recorrido usa los archivos versionados y no requiere descargar `loan.csv`.

- El módulo de datos documenta el número de filas y variables, el periodo, la
  fecha de descarga, el diccionario consultado y cómo genera los datos
  procesados desde el archivo original.
- PD modela la clase positiva como incumplimiento (`1`). Los estados sin
  resultado definitivo se excluyen según la consigna; no se deben etiquetar
  como buenos.
- LGD documenta la fórmula, los campos usados y sus limitaciones. La
  exposición al incumplimiento (EAD) debe definirse explícitamente antes de
  combinar los modelos como `pérdida esperada = PD × LGD × EAD`.
- Las familias de modelos que se comparen usan las mismas particiones y el
  mismo preprocesamiento. Reporten el presupuesto de búsqueda, tiempos,
  complejidad y, para la configuración final de cada red, resultados sobre
  al menos 30 semillas.
- Las semillas, hiperparámetros y parámetros de scorecard deben poder
  cambiarse desde configuración o línea de comandos, no editando la lógica
  del modelo. `configs/default.yaml` propone una partición temporal,
  configuración de scorecard (odds buenas:malas) y semilla, conectada al
  modelo final PD mediante el manifiesto de artefactos.
- Los modelos guardados permiten abrir la app y regenerar figuras sin
  reentrenar. Documenten el procedimiento de regeneración cuando se integren
  los experimentos.

## Datos, modelos y artefactos

`data/raw/` se excluye de Git por tamaño. Se versionan las matrices train y
validación PD necesarias para inferencia e interpretación, sus etiquetas de
validación y el paquete del modelo final. El archivo original y los artefactos
LGD siguen teniendo sus propias reglas. Las excepciones explícitas de
`.gitignore` evitan incluir archivos temporales junto a las entregas finales.

No incluyan credenciales, tokens ni datos personales. No suban datasets,
modelos o figuras que no sean necesarios para reproducir el trabajo.

## Reproducibilidad y entrega

Cada experimento principal se lanza con un **único comando** desde la raíz del repositorio, con parámetros configurables externamente en archivos YAML o por línea de comandos, y demostraciones rápidas medidas en menos de dos minutos:

### 1. Experimento de PD (Probabilidad de Incumplimiento)
```bash
# Corrida completa (30 semillas)
python -m credit_risk.pd.experiments

# Demostración rápida (< 2 minutos, 21.14 s medidos en CPU local)
python -m credit_risk.pd.experiments --demo
```
Configuración en `configs/pd_mlp_best.yaml` y `configs/default.yaml`.

### 2. Experimento de LGD (Pérdida Dado el Incumplimiento - Integrante 3)
```bash
# Corrida completa (30 semillas)
python -m credit_risk.lgd.experiments --seeds 30

# Demostración rápida (< 2 minutos, 37.19 s medidos en CPU local)
python -m credit_risk.lgd.experiments --demo

# Modificación de parámetros en vivo (sin editar código)
python -m credit_risk.lgd.experiments --demo --hidden-dims 128,64 --lr 0.005 --epochs 20
```
Configuración en `configs/lgd_mlp_best.yaml`. Las matrices procesadas están versionadas en Git o se autogeneran si no estuvieran presentes.

### 3. Aplicación web y scorecard
```bash
# Verificación de artefactos
python run_app.py --check

# Inicio de la aplicación
python run_app.py
```

### 4. Cacería de la alucinación (15 % grupal)
Documentada en la Sección 4 del reporte técnico y detallada con prompts literales, respuestas y evidencias en [`docs/caceria_alucinaciones.md`](docs/caceria_alucinaciones.md). Cubre 5 hallazgos técnicos en las 5 categorías exigidas por la rúbrica.

## Pruebas

Ejecuten la suite completa de pruebas desde la raíz del repositorio con:

```bash
python -m pytest -v
1. Clonar el repositorio y entrar en su carpeta:

```powershell
git clone https://github.com/Nicorb72/-Redes-neuronales-para-datos-tabulares.git
Set-Location .\-Redes-neuronales-para-datos-tabulares
```

2. Instalar las dependencias siguiendo la sección [Instalación](#instalación).
   Luego, desde la raíz, configurar el acceso al código:

```powershell
$env:PYTHONPATH = "src"
```

3. Regenerar las dos figuras EDA. Si no está el CSV, se usan los conteos del
   resumen versionado:

```powershell
python -m credit_risk.data.generate_eda_figures
```

4. Verificar el paquete PD y las matrices incluidas en el clon:

```powershell
python run_app.py --check
```

Debe aparecer `PD listo`. Para ejecutar la aplicación:

```powershell
python run_app.py
```

Abrir `http://127.0.0.1:8501` y detener el servidor con `Ctrl+C`.

5. Ejecutar las pruebas:

```powershell
python -m pip install -r requirements-test.txt
python -m pytest -q
```

### Reentrenar y evaluar PD usando las matrices versionadas

Este paso tampoco requiere el CSV: lee las matrices PD de train, validation y
test que están en el repositorio y ejecuta la regresión logística y el MLP con
30 semillas. Puede tardar varios minutos. Escribe resultados en
`reports/pd_evaluation/` y actualiza el modelo y el preprocesador en
`artifacts/models/`.

```powershell
$env:PYTHONPATH = "src"
python -m credit_risk.pd.experiments
```

Al regenerar el modelo o el preprocesador, hay que reconstruir el manifiesto
del paquete PD antes de iniciar la app. Usa el comando de [Exportar el paquete
PD](#exportar-el-paquete-pd) y luego vuelve a ejecutar `python run_app.py
--check`.

### Reconstruir matrices PD desde el CSV original (opcional)

Este recorrido reproduce la preparación desde los datos originales y **sí
requiere** el archivo de Lending Club en `data/raw/loan/loan.csv`. El resumen
EDA no contiene registros ni variables predictoras y no sirve para este paso.
Descarga el archivo desde [Credit Risk Analysis en
Kaggle](https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset) y
colócalo en esa ruta. La copia de referencia del proyecto se descargó el
24/10/2026; no se registró un identificador de versión o checksum de Kaggle.
Conserva la procedencia de la copia que utilices y no asumas que una descarga
posterior producirá matrices idénticas.

```powershell
$env:PYTHONPATH = "src"
python -m credit_risk.data.prepare_data
```

Este comando recalcula las cuatro particiones PD y sobrescribe las matrices,
etiquetas y el preprocesador en `data/processed/` y `artifacts/models/`.
**No lo ejecutes si solo quieres usar el clon limpio**: además de requerir el
CSV, reemplazar el preprocesador puede dejar desactualizado el manifiesto del
modelo existente. Para completar una reconstrucción desde la fuente, continúa
con el entrenamiento PD, exporta de nuevo el paquete y valida la app como se
describe arriba.

En macOS/Linux, configura una vez la variable y ejecuta el mismo módulo:

```bash
export PYTHONPATH=src
python -m credit_risk.data.prepare_data
```

El preprocesador se ajusta únicamente con train.

### Regenerar las figuras de datos

El comando usa el CSV para recalcular conteos si está disponible; si no, usa
el resumen versionado para generar las dos figuras principales de PD. Los
archivos se escriben en `reports/figures/`.

```powershell
$env:PYTHONPATH = "src"
python -m credit_risk.data.generate_eda_figures
```

### Ejecutar experimentos LGD (Comando único, parámetros externos y demo < 2 min)

Las matrices procesadas de LGD están versionadas en Git (`data/processed/*_lgd.*`), por lo que los experimentos de LGD se ejecutan directamente con un solo comando en un clon limpio sin necesidad de `loan.csv`:

```powershell
$env:PYTHONPATH = "src"
# Demostración rápida (< 2 minutos, ~37 s medidos en CPU local con 5 semillas)
python -m credit_risk.lgd.experiments --demo

# Corrida completa del benchmark (30 semillas)
python -m credit_risk.lgd.experiments --seeds 30

# Modificación de hiperparámetros en vivo sin tocar código
python -m credit_risk.lgd.experiments --demo --hidden-dims 128,64 --lr 0.005 --epochs 20
```

Los hiperparámetros están externalizados en `configs/lgd_mlp_best.yaml`. Si las matrices faltaran y estuviera presente `loan.csv`, el script las reconstruye automáticamente. Si se desea reconstruir las matrices desde la fuente de manera explícita: `python -m credit_risk.lgd.data`.

Los resultados y modelos LGD se guardan bajo `artifacts/models/` y las figuras bajo `reports/figures/`.

### Exportar el paquete PD

Después de cambiar o regenerar el modelo PD, su preprocesador o las matrices de
referencia, ejecuta este comando para reconstruir el manifiesto. Calcula los
hashes y valida el paquete:

```powershell
$env:PYTHONPATH = "src"
python -m credit_risk.scorecard.export --model artifacts/models/pd_mlp_final.pt --model-format pd_mlp --model-id pd_mlp_final-seed202612 --training-data "Matrices PD train 2007-2012" --reference-population "90347 creditos etiquetados de train 2007-2012"
```

### Interpretabilidad PD

Con el paquete PD y las matrices y etiquetas de validación:

```powershell
$env:PYTHONPATH = "src"
python -m credit_risk.scorecard.interpretability --seed 42 --repeats 10 --max-rows 5000
```

Los resultados se guardan en `reports/interpretability/` como JSON, CSV y SVG.

### Aplicación web

Validar el paquete y arrancar Streamlit:

```powershell
python run_app.py --check
python run_app.py
```

La aplicación se abre en `http://127.0.0.1:8501`. Para cambiar el puerto:

```powershell
python run_app.py --port 8502
```

El enlace al reporte y al video promocional aparecen en la barra lateral.

### Pruebas y lint

```powershell
python -m pip install -r requirements-test.txt
python -m pytest -q
python -m ruff check src tests
```

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
