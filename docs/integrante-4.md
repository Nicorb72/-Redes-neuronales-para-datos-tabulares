# Integrante 4: scorecard, interpretación y aplicación

## Revisión inicial y bloqueos

Revisión del 26 de octubre de 2026, sobre `main` en `d4d95b8`.
Rama de trabajo: `feature/integrante-4-scorecard-app`. Este documento define
una propuesta de integración verificable; no implica que los otros integrantes
ya hayan aprobado el contrato.

| Responsable | Contrato encontrado | Entrega pendiente |
| --- | --- | --- |
| Datos (1) | `default_flag=1` significa incumplimiento; 11 numéricas y 4 categóricas; `ColumnTransformer` ajustado solo en train | Archivos `preprocesador_pd.joblib`, `columnas_pd.json` y matrices procesadas; no están en el clon |
| PD (2) | `pd/` contiene solamente `__init__.py` | Modelo entrenado, versiones del entorno, identificación de corrida y procedencia de train; interfaz de probabilidad explícita |
| LGD (3) | `lgd/` contiene solamente `__init__.py` | Modelo, fórmula de LGD y definición de EAD; la app no muestra LGD ni pérdida esperada |
| Integración (5) | Reporte inicial y configuración de scorecard | Publicación de app, video real, verificación en otra máquina y artefactos finales versionados |

Las matrices definidas por `data.prepare_data` son `X_train_pd.npz`,
`X_validation_pd.npz`, `X_test_pd.npz` y la partición complementaria 2015,
con etiquetas en archivos `y_<split>_pd.npy`. Son matrices transformadas,
no entradas adecuadas para un formulario. No se presupone que siempre tengan
44 columnas: se verifica la dimensión efectiva de los artefactos.
La división implementada usa train 2007–2012, validación 2013 y test 2014.
Los porcentajes generales de `configs/default.yaml` no gobiernan esa división.

El PDF exige guardar modelos y datos procesados, pero `.gitignore` los excluye.
Esta rama no añade modelos ni datos inventados y tampoco cambia esa política
global: el equipo debe versionar explícitamente los archivos finales aprobados
(o modificar las exclusiones) antes de entregar. El arranque funciona sin ellos,
mostrando integración pendiente; la consulta real queda bloqueada.

## Instalación y comandos

Desde la raíz del clon y un entorno Python 3.11.9 activado:

```powershell
python -m pip install -r requirements-test.txt
python run_app.py
```

`requirements-app.txt` contiene solo las dependencias directas de inferencia y
web, fijadas a las mismas versiones del entorno original. `requirements-test.txt`
añade pruebas y generación de figuras. Para entrenar o cargar un objeto que
dependa de PyTorch se necesita además el entorno completo `requirements.txt`.
Las dependencias transitivas no constituyen un lock completo.

La app usa `http://127.0.0.1:8501`. Se puede cambiar el puerto y la configuración:

```powershell
python run_app.py --port 8502 --config configs/default.yaml
python run_app.py --check
python -m pytest
```

`--check` retorna 0 si el paquete PD es compatible y 2 si falta o es inválido.
No necesita navegador. El lanzador configura la ruta del paquete, sin exigir
instalación editable. Los comandos de módulos siguientes requieren la ruta
`src` (en macOS/Linux: `export PYTHONPATH=src`):

```powershell
$env:PYTHONPATH = "src"
python -m credit_risk.scorecard.export --model artifacts/models/pd_model.joblib --model-id ID_REAL_DE_CORRIDA --training-data VERSION_REAL_DE_TRAIN --reference-population "Créditos etiquetados de train 2007-2012"
python -m credit_risk.scorecard.interpretability --repeats 10 --max-rows 5000
```

Los identificadores en mayúsculas son argumentos que debe sustituir quien
entrenó el modelo; no corresponden a una corrida existente. El exportador
solo empaqueta artefactos existentes y comprueba su compatibilidad.
Nunca entrena, ajusta el preprocesador ni descarga datos.
No sobrescribe un manifiesto válido si la nueva validación falla.

## Contrato PD versión 1

Entradas del formulario: exactamente las columnas de `VARIABLES_NUMERICAS +
VARIABLES_CATEGORICAS` en `data.preprocessing`. La antigüedad se expresa en
meses al originar el crédito, no en meses hasta hoy. Se introducen variables
antes de transformar. Las categorías se obtienen del encoder ajustado,
conservando incluso espacios en los códigos originales de `term`.
El formulario exige datos completos, números finitos no negativos y conteos
enteros; no inventa valores para campos faltantes. Sus valores iniciales
provienen de las medianas (redondeadas para conteos) y categorías del preprocesador: deben revisarse.
No impone techos financieros arbitrarios ni garantiza que un perfil esté dentro
de la distribución de entrenamiento. Los rótulos de la app son orientativos;
el equipo de datos debe contrastar su significado con el diccionario original.

El preprocesador debe ser el `ColumnTransformer` guardado por el módulo de
datos, con `feature_names_in_` y `get_feature_names_out()`. El modelo recibe
las columnas **ya transformadas**, declara `n_features_in_`, `classes_` con
las clases numéricas 0 y 1, y expone `predict_proba(X)` con salida `n × 2`.
Se busca la posición de la clase 1; no se presupone que siempre sea la segunda.
Se comprueban límites, valores finitos y suma por fila. Puede ser una regresión
logística o un MLP con esa interfaz. No se presenta una logística como red.
Los modelos que requieran entrada densa usan `--input-format dense`.

Si el integrante 2 entrega únicamente un `state_dict` de PyTorch, **aún hace
falta un adaptador** con arquitectura, pesos, modo `eval`, salida sigmoid o
softmax y esta interfaz; no es correcto deducirlos de los pesos ni volver
a entrenar un modelo sustituto. Se debe guardar la clase del adaptador en un
módulo importable del repositorio. La probabilidad de la clase 1 debe incluir
la calibración seleccionada por el equipo de PD, si existe.

`pd_bundle.json` guarda versión, modelo/corrida, versión de train, población
de referencia, clase positiva, columnas originales y formato de entrada. Para
cada uno de modelo, preprocesador, columnas y matriz train guarda ruta relativa
al manifiesto y SHA-256. El exportador construye esos hashes. La carga comprueba
dimensiones, columnas y hashes antes de inferir. Los hashes detectan cambios;
no demuestran que el equipo entrenara ese modelo con ese train. Esa trazabilidad
debe conservarse en el registro de entrenamiento. Solo se deben cargar archivos
joblib confiables del equipo: deserializarlos puede ejecutar código.

## Derivación de la scorecard

Sea `p = P(default_flag = 1 | x)`. Se definen las odds buenas:malas como
`O = (1-p)/p`. Buscamos `S = A + B ln(O)` con dos condiciones:

1. En las odds de referencia `O0`, el puntaje es `S0`.
2. Duplicar las odds buenas:malas suma exactamente `PDO` puntos.

Entonces `S(2O) - S(O) = B ln(2) = PDO`, de donde
`B = PDO / ln(2)` y `A = S0 - B ln(O0)`. Por tanto:

```text
S(p) = S0 + (PDO/ln(2)) · [ln(1-p) - ln(p) - ln(O0)]
p(S) = 1 / [1 + exp((S-A)/B)]
dS/dp = -B / [p(1-p)] < 0
```

Configuración inicial: `S0=600`, `O0=50:1`, `PDO=20`. Son convenciones de
escala, no parámetros estimados ni resultados de desempeño. En `p=1/51`,
el puntaje es 600; en `p=1/101`, es 620. Un puntaje mayor indica menor PD.
`base_score`, `base_good_to_bad_odds` y
`points_to_double_good_to_bad_odds` se cambian en el YAML. Se validan finitud,
odds y PDO positivos. Los cálculos conservan precisión; se redondea solo al mostrar.

Para PD exactamente 0 o 1 se recorta únicamente en la transformación de
puntaje con `epsilon=1e-9`, configurable dentro de `scorecard`. Se evita un
puntaje infinito, sin alterar la PD que se muestra. La inversa de esos extremos
devuelve el límite recortado, no el extremo original. La inversa usa una
formulación numéricamente estable.

Esta es una scorecard continua derivada de la salida del modelo. Para un MLP
no hay puntos aditivos constantes por variable: las interacciones hacen que
el cambio de puntaje dependa del perfil. La app obtiene los puntos de un cambio
como `S(p(x_modificado)) - S(p(x_original))`, pasando ambos perfiles por el
mismo preprocesador y modelo. No se atribuye a cada variable una descomposición
exacta del puntaje total.

## Percentil y escenarios

La referencia es la matriz **train** del mismo paquete PD. Se calcula su PD
en lotes de hasta 2048 filas y se transforma con la misma scorecard de la consulta.
No se acepta `reference_split=test`. Se usa la distribución empírica inclusiva:
`percentil(s) = 100 · número de puntajes de referencia <= s / n`.
Los empates reciben el extremo superior; puede llegar a 0 o 100. Se calcula
con puntajes sin redondear. Cambiar la escala no debe cambiar el percentil,
salvo empates inducidos por el recorte de probabilidades extremas.

El percentil se refiere a los créditos etiquetados seleccionados en train,
no a la población general ni a la población contemporánea. No es PD,
probabilidad de aprobación ni evidencia de calibración.

Cada escenario modifica una variable de la última consulta calculada y
reutiliza modelo, referencia y escala. La app muestra PD, puntaje, percentil
y sus diferencias. Mantener todo lo demás constante permite explorar la
dirección local de la respuesta del modelo. No asegura factibilidad del
escenario ni identifica efectos de una intervención real.

## Interpretabilidad: método, límites y referencias

Se usa importancia por permutación agrupada en validación temporal, con
modelo congelado. Es independiente de la arquitectura y mide cuánto cambia
el error al romper la relación entre una variable y las otras columnas/etiqueta.
Este uso sigue el enfoque de dependencia predictiva por permutación de
Fisher et al. (2019); no implementa su estimación de dependencia de toda
una clase de modelos.

Para cada variable original `j` y repetición `r` se permutan **juntas** todas
sus columnas one-hot con el mismo orden de filas. Así no se crean vectores
categóricos imposibles. Para una muestra fija de validación:
`I_jr = Brier(y, p(X_permutado_jr)) - Brier(y, p(X))`, con
`Brier = media((p-y)^2)`. La métrica responde a errores de probabilidad, por
eso resulta coherente con una app que muestra PD, aunque no reemplaza el
análisis de discriminación o calibración de PD.

El comando guarda JSON con semilla, índices muestreados, repeticiones,
hashes, modelo, Brier original y valores por repetición; también CSV y SVG.
No reajusta ni selecciona el modelo con el conjunto test. La semilla por
defecto se toma de `configs/default.yaml`; `--seed` permite cambiarla.
El límite de 5000 filas controla el tiempo, pero no garantiza dos minutos
con cualquier arquitectura o máquina. Las etiquetas deben corresponder
exactamente al orden de filas de la matriz: el hash no prueba esa alineación.

Una importancia positiva alta indica que el modelo depende de esa variable
para predecir en la muestra. No informa por sí sola si valores altos aumentan
riesgo. Para esa dirección se usa la simulación local del modelo, con el perfil
y el cambio explícitos. No se afirma que el efecto sea monotónico o igual en
todos los clientes. Importancias negativas se conservan y deben investigarse.
La desviación entre permutaciones describe aleatoriedad de la permutación;
no es un intervalo de confianza poblacional ni variabilidad entre 30 redes.

Las variables correlacionadas pueden compartir o esconder importancia,
y permutarlas puede crear combinaciones numéricas fuera de distribución.
Esta limitación está documentada también por scikit-learn (s. f.). Para
conclusiones finales se deben examinar correlaciones, estabilidad temporal,
desempeño del modelo y escenarios plausibles con el equipo de PD/datos.

Referencias APA 7:

- Fisher, A., Rudin, C., & Dominici, F. (2019). All models are wrong, but many
  are useful: Learning a variable’s importance by studying an entire class of
  prediction models simultaneously. *Journal of Machine Learning Research,
  20*(177), 1–81. https://www.jmlr.org/papers/v20/18-760.html
- scikit-learn developers. (s. f.). *Permutation importance with multicollinear
  or correlated features* (versión 1.5.2).
  https://scikit-learn.org/1.5/auto_examples/inspection/plot_permutation_importance_multicollinear.html

## Resultados y entregables aún pendientes

No hay AUC, Brier real, ranking de variables ni perfiles reales calculados
en esta rama porque faltan los artefactos entrenados. Las pruebas generan
datos sintéticos únicamente en directorios temporales y los identifican
como `SYNTHETIC-TEST-ONLY`; nunca se distribuyen como resultados del trabajo.

Al recibir el paquete real:

1. Verificar procedencia, diccionario, entorno y clase positiva con integrantes 1 y 2.
2. Ejecutar exportador, `--check`, pruebas y app sin reentrenar.
3. Generar interpretación con matrices/etiquetas de validación alineadas.
4. Revisar resultados y citar en el blog la Tabla I4-1 (CSV generado) y
   Figura I4-1 (SVG), con título, fuente y explicación de sus límites.
5. Configurar el enlace real al video. Probar el enlace al reporte tras su despliegue.
6. Publicar la app en un servidor Python/Streamlit con los artefactos del paquete
   y versiones fijadas; el HTML estático de GitHub Pages no ejecuta Python.
   La publicación y su URL siguen pendientes.
7. Medir la demostración real (<120 s) y repetir en un clon limpio en otra máquina.

## Uso de IA y verificaciones

El encargo fue implementar scorecard configurable, interpretabilidad justificada
y app con PD/puntaje/percentil/simulación, sin modificar main ni inventar resultados.
La IA ayudó con derivación, implementación, pruebas y documentación. El equipo
debe conservar el prompt literal de esta conversación y revisar el código antes
de defenderlo. No se inventan cinco alucinaciones para completar el reporte.

Verificaciones implementadas: signo de la transformación, punto de referencia,
duplicación de odds, inversa, límites, empates de percentil, búsqueda de clase 1,
contratos dañados, agrupación one-hot, repetibilidad, consulta y simulación web.
Estas comprobaciones son evidencia de revisión, no cinco errores observados
atribuibles a una respuesta de IA. Los resultados de ejecución se documentan
en la entrega y en el PR.
