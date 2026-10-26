# Figuras

Guarden aquí las figuras reproducibles del reporte. Cada figura debe tener
número, título, fuente y una referencia explícita en el texto del reporte.

Las figuras `distribucion_etiquetas_pd.svg` y
`particiones_temporales_pd.svg` tienen funciones de generación en
`src/credit_risk/data/eda.py`. Se pueden regenerar desde la raíz del proyecto
con los siguientes comandos de PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m credit_risk.data.generate_eda_figures
```

El CSV original no está en el clon de trabajo, así que las figuras actuales
reflejan los conteos resumidos en `src/credit_risk/data/README.md` y no se
recalcularon independientemente. Regenerenlas con el archivo fuente antes de
la entrega.
