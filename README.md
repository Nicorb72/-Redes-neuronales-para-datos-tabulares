# Riesgo de crédito con redes neuronales para datos tabulares

Proyecto académico para estimar la probabilidad de incumplimiento (PD), la
pérdida dado incumplimiento (LGD) y la pérdida esperada, y para presentar los
resultados mediante una scorecard y una aplicación web.

> **Estado:** modelo PD final, preprocesador, matrices de inferencia y scorecard
> integrados. La app calcula PD, puntaje, percentil y escenarios sin reentrenar.
> LGD tiene su implementación propia; publicación de la app y video pendientes.

## Aplicación y scorecard

Con el entorno activado, instalar `python -m pip install -r requirements-test.txt`
y arrancar con un único comando:

```powershell
python run_app.py
```

Abrir `http://127.0.0.1:8501`. Usar `python run_app.py --check` para comprobar
los artefactos sin navegador. `configs/default.yaml` permite cambiar puntaje
base, odds buenas:malas, PDO, ruta del paquete PD y enlaces al reporte/video.
La app usa `pd_mlp_final.pt` y muestra PD, puntaje, percentil de la
referencia train y simulación de cambios, siempre con el mismo modelo.

La [guía del integrante 4](docs/integrante-4.md) contiene los bloqueos por
responsable, el contrato de inferencia, el comando de empaquetado de modelos,
la derivación matemática, las pruebas y el método de interpretabilidad con
referencias verificables. No se necesitan datos originales ni reentrenamiento
para abrir la app; las consultas requieren los artefactos reales guardados.

## Reporte tipo blog

La página inicial del reporte está en
[`reports/technical-report/index.html`](reports/technical-report/index.html)
y puede abrirse directamente desde un clon local. Al publicar los cambios en
`main`, GitHub Actions desplegará esa carpeta en
[GitHub Pages](https://nicorb72.github.io/-Redes-neuronales-para-datos-tabulares/technical-report/).
El reporte se publica en la ruta `/technical-report/` del sitio para conservar
también los recursos gráficos de `reports/figures/`.
En la configuración del repositorio, seleccionen **Settings → Pages → Build
and deployment → Source: GitHub Actions** para habilitar el primer despliegue.
El enlace público estará disponible cuando termine correctamente esa ejecución.

## Requisitos previos

- Python **3.11.9** (versión fijada en `.python-version`).
- Git.
- El archivo de datos Credit Risk Analysis de Kaggle. Descárguenlo desde
  [la fuente del conjunto](https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset)
  y colóquenlo en `data/raw/loan/loan.csv`, como espera el módulo de carga.
  No se requiere una API de Kaggle para ejecutar el proyecto una vez que el
  archivo esté disponible localmente.

## Instalación

Desde la raíz del repositorio, creen y activen un entorno virtual:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En macOS o Linux, activen el entorno con
`source .venv/bin/activate`. Las versiones de las dependencias están fijadas
en `requirements.txt`.

## Estructura

```text
configs/
  default.yaml          Configuración inicial compartida
data/
  raw/                  Archivo original local; no se versiona
  processed/            Datos derivados reproducibles y metadatos
src/credit_risk/
  data/                 Validación, limpieza y preparación de datos
  pd/                   Clasificación de probabilidad de incumplimiento
  lgd/                  Regresión de pérdida dado incumplimiento
  scorecard/            Conversión de PD a puntaje e interpretación
  evaluation/           Particiones, métricas, semillas y tiempos
  app/                  Aplicación web
artifacts/models/       Modelos entrenados para la aplicación y las figuras
reports/
  figures/              Figuras regenerables desde código
  technical-report/     Fuente y recursos del reporte técnico tipo blog
tests/                  Pruebas de integración y de los módulos
```

## División de trabajo e integración

| Integrante | Responsabilidad                                                   | Ubicación principal                                                   |
| ---------- | ----------------------------------------------------------------- | ---------------------------------------------------------------------- |
| 1          | Datos, diccionario de variables y análisis descriptivo           | `src/credit_risk/data/`, `data/processed/`                         |
| 2          | Clasificación y estimación de PD                                | `src/credit_risk/pd/`                                                |
| 3          | Construcción de LGD y modelos de regresión                      | `src/credit_risk/lgd/`                                               |
| 4          | Scorecard, interpretación y aplicación web                      | `src/credit_risk/scorecard/`, `src/credit_risk/app/`               |
| 5          | Integración, reproducibilidad, evaluación comparativa y reporte | `src/credit_risk/evaluation/`, `reports/`, README y configuración |

Antes de integrar, acuerden entre los responsables las columnas y el formato
de los datos de entrada, las particiones que compartirán PD y LGD, las rutas
de los modelos guardados y la interfaz de predicción que consumirá la app.
Registren esas decisiones y sus cambios en commits pequeños y descriptivos;
integren cada módulo mediante una rama y un pull request, en lugar de acumular
todo en un único commit final.

### Contratos que deben respetarse

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
```

Las pruebas que requieren el CSV o matrices locales se omiten si esos archivos
no están disponibles; las pruebas unitarias de etiquetado, particiones,
preprocesamiento y exclusión de variables posteriores al desembolso no
requieren el dataset original.
