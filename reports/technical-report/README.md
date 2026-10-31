# Reporte técnico

La fuente única es `index.md`; `index.html` es generado. No editar el HTML a mano.
`template.html` conserva el diseño, y `evidence.json` registra hashes y resultados
usados en las tablas de métricas y escenarios. El generador no entrena modelos.

Desde la raíz del repositorio, con el entorno del proyecto activo:

```console
python -m pip install -r requirements-report.txt
python scripts/report_evidence.py
python scripts/build_report.py --refresh-evidence
python scripts/build_report.py --check
```

La primera instalación agrega únicamente el generador Markdown. El cálculo de
evidencia usa dependencias y modelos del proyecto, matrices de validación y las
30 corridas PD guardadas. No requiere `loan.csv` ni conexión externa.

Para una edición de texto que no cambie las tablas:

```console
python scripts/build_report.py
python scripts/build_report.py --check
```

La comprobación detecta HTML desactualizado, hashes de evidencia incompatibles,
tablas derivadas desactualizadas, numeración no consecutiva y enlaces locales
inexistentes. No certifica contenido de videos ni autenticidad de conversaciones.

Para previsualizar el reporte con imágenes desde la raíz:

```console
python -m http.server 8000
```

Abrir `http://localhost:8000/reports/technical-report/`. Las fórmulas se presentan
con MathJax 3.2.2 desde CDN; sin conexión quedan sus expresiones TeX legibles.
La app y los experimentos no dependen de ese recurso de presentación del blog.

GitHub Pages publica el HTML versionado al integrar el PR en `main`. Antes de
integrarlo, revisar también el HTML y `python scripts/build_report.py --check`.
Actualizar JSON/CSV de resultados cuando se resuelvan los pendientes de modelos;
no presentar la demo LGD de cinco semillas como benchmark de treinta.

- [Reporte publicado](https://nicorb72.github.io/-Redes-neuronales-para-datos-tabulares/reports/technical-report/)
- [Video promocional](https://youtu.be/TijgcPZNboU)
