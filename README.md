# Riesgo de crédito con redes neuronales para datos tabulares

Proyecto académico para estimar la probabilidad de incumplimiento (PD), la
pérdida dado incumplimiento (LGD) y la pérdida esperada, y para presentar los
resultados mediante una scorecard y una aplicación web.

> **Estado:** datos y estructura implementados; scorecard y aplicación local
> disponibles. Faltan los modelos entrenados y los artefactos procesados en el
> clon. La app muestra integración pendiente hasta recibir un paquete PD válido;
> no genera predicciones ficticias. Entrenamiento PD/LGD y publicación pendientes.

## Aplicación y scorecard

Con el entorno activado, instalar `python -m pip install -r requirements-test.txt`
y arrancar con un único comando:

```powershell
python run_app.py
```

Abrir `http://127.0.0.1:8501`. Usar `python run_app.py --check` para comprobar
los artefactos sin navegador. `configs/default.yaml` permite cambiar puntaje
base, odds buenas:malas, PDO, ruta del paquete PD y enlaces al reporte/video.
Una vez integrado el modelo, la app muestra PD, puntaje, percentil de la
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
  configuración inicial de scorecard (odds buenas:malas) y semilla; aún no
  está conectada a modelos.
- Los modelos guardados permiten abrir la app y regenerar figuras sin
  reentrenar. Documenten el procedimiento de regeneración cuando se integren
  los experimentos.

## Datos, modelos y artefactos

`data/raw/` se excluye de Git para evitar subir el archivo original sin
revisar tamaño y condiciones de distribución. No se excluyen los datos
procesados, modelos ni figuras: la configuración actual de `.gitignore` sí
excluye `data/processed/`, los archivos de `artifacts/models/` y las figuras
generadas. Antes de entregar, el equipo debe decidir cuáles artefactos finales
versionará, comprobar el tamaño del repositorio y documentar su origen. Si el
archivo original es demasiado grande, versionen la versión procesada y
comprimida que use el proyecto, junto con el código y los metadatos necesarios
para reproducirla.

No incluyan credenciales, tokens ni datos personales. No suban datasets,
modelos o figuras que no sean necesarios para reproducir el trabajo.

## Reproducibilidad y entrega

La aplicación local se inicia con `python run_app.py`. La ejecución completa
y la demostración de los experimentos de entrenamiento siguen pendientes.
La demostración debe tardar menos de dos minutos; su tiempo con modelos reales
debe medirse al integrar los artefactos. Los comandos de interpretación y
validación se describen en `docs/integrante-4.md`.

Antes de entregar:

- Ejecuten instalación, pruebas, demostración y aplicación en un clon limpio y
  sin conexión a servicios externos.
- Verifiquen que las figuras se regeneran desde código y que la app carga los
  modelos guardados.
- Comprueben que el reporte tipo blog cita figuras y tablas en el texto,
  utiliza referencias verificables en APA 7 y contiene la metodología,
  resultados, discusión, uso de IA y la cacería de alucinaciones con evidencia.
- Incluyan en el reporte los enlaces al repositorio, aplicación y video
  promocional; entreguen también el video de contribución individual de cada
  integrante.

## Pruebas

Ejecuten las pruebas desde la raíz del repositorio con:

```powershell
python -m pytest
```

Las pruebas que requieren el CSV o matrices locales se omiten si esos archivos
no están disponibles; las pruebas unitarias de etiquetado, particiones,
preprocesamiento y exclusión de variables posteriores al desembolso no
requieren el dataset original.
