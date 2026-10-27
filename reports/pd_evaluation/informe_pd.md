# Evaluación de probabilidad de incumplimiento (PD)

## Diseño experimental

Se reutilizaron las particiones temporales y el preprocesador del proyecto: entrenamiento 2007–2012, validación 2013, prueba 2014 y conjunto complementario 2015. El preprocesador se ajustó solo con entrenamiento. La validación controla parada temprana del MLP; la prueba 2014 se reserva para esta evaluación final. El conjunto de 2015 no se usa por su alta proporción de resultados sin etiqueta.

| Partición | Préstamos etiquetados | Tasa de incumplimiento |
|---|---:|---:|
| Entrenamiento 2007–2012 | 90.347 | 16.30% |
| Validación 2013 | 73.194 | 22.95% |
| Prueba 2014 | 74.154 | 26.22% |

La tasa de incumplimiento aumenta de entrenamiento a validación y prueba. Esto evidencia un cambio temporal en la población etiquetada; por ello la evaluación de 2014 es más exigente y las probabilidades requieren cautela.

La regresión logística es el modelo de referencia: `C=1.0`, solver `lbfgs`, sin ponderación de clases y ajustada sobre las mismas observaciones y características transformadas que el MLP. El MLP usa 2 capas ocultas de 64 neuronas, activación relu, dropout 0.3327, optimizador nadam, tasa de aprendizaje 0.009872, lote 2048 y regularización L1 con λ=4.13e-06. Se ejecutaron 30 semillas (202612–202641). El checkpoint entregado se ajustó con la semilla fija 202612.

Las corridas comparten los mismos conjuntos temporales e hiperparámetros; solo cambia la semilla de inicialización y el orden de minibatches. La desviación estándar describe variabilidad por aleatoriedad del entrenamiento, no incertidumbre muestral. El modelo entregado usa la semilla configurada 202612; no se seleccionó una semilla usando la prueba.

## Resultados en prueba temporal 2014

| Métrica | Regresión logística | MLP: media ± desviación estándar (30 semillas) |
|---|---:|---:|
| AUC | 0.66481 | 0.66941 ± 0.00435 |
| Gini | 0.32962 | 0.33883 ± 0.00870 |
| KS | 0.24071 | 0.24698 ± 0.00726 |
| Brier | 0.18726 | 0.18180 ± 0.00096 |
| Log-loss | 0.56362 | 0.54471 ± 0.00267 |

El AUC medio del MLP supera al de la logística en +0.00460; el Brier medio cambia en -0.00546 (negativo favorece al MLP). La mejora es moderada y debe interpretarse junto a la variabilidad entre semillas y al cambio de tasa de incumplimiento entre periodos. El MLP de semilla fija 202612 obtuvo: AUC=0.67588, Gini=0.35175, KS=0.25870, Brier=0.18191 y log-loss=0.54394. La regresión logística obtuvo: AUC=0.66481, Gini=0.32962, KS=0.24071, Brier=0.18726 y log-loss=0.56362.

## Definiciones matemáticas

Para cada préstamo i, yᵢ ∈ {0,1} es el resultado y pᵢ ∈ [0,1] la PD estimada. La regresión logística calcula pᵢ = σ(β₀ + βᵀxᵢ), donde σ(z)=1/(1+e⁻ᶻ), y estima β minimizando la entropía cruzada binaria con penalización L2 de scikit-learn; `C=1.0` es la inversa de la intensidad de regularización. El MLP calcula pᵢ = σ(fθ(xᵢ)); sus pesos se entrenan minimizando la entropía cruzada binaria más la penalización L1: L(θ)=−(1/n)Σᵢ[yᵢlog(pᵢ)+(1−yᵢ)log(1−pᵢ)] + λΣⱼ|wⱼ|. El dropout desactiva aleatoriamente unidades ocultas durante entrenamiento con probabilidad 0.3327.

- **AUC:** área bajo ROC; mide la probabilidad de ordenar un incumplidor por encima de un buen pagador.
- **Gini:** Gini = 2·AUC − 1.
- **KS:** KS = maxₜ |TPR(t) − FPR(t)|, máxima separación entre las distribuciones acumuladas de las dos clases.
- **Brier:** (1/n)Σᵢ(pᵢ−yᵢ)²; menor valor indica mejor precisión probabilística.
- **Calibración:** compara, por grupos de riesgo, la PD media predicha con la tasa observada. La diagonal representa calibración ideal.

## Lectura y límites

La curva de calibración en prueba muestra que ambos modelos subestiman la tasa observada en varios grupos, especialmente entre los de mayor PD predicha. No se ajustó un calibrador sobre la prueba. Si se calibra el modelo, el calibrador debe ajustarse con datos de desarrollo y su efecto evaluarse posteriormente en una partición independiente.

La comparación debe considerar discriminación y calibración, no solo AUC. Las semillas miden sensibilidad a la aleatoriedad del MLP sobre una partición fija; no reemplazan una validación temporal adicional. La prueba se originó en 2014, pero los estados crediticios del archivo pueden haberse observado después de la originación; por tanto, este experimento es una evaluación temporal de originaciones y no demuestra por sí solo una validación histórica con horizonte de desempeño fijo. Las probabilidades dependen también de la definición de la etiqueta y de la selección de préstamos con estado observable.

## Conclusiones: por qué la mejora del MLP es limitada

El MLP **sí supera ligeramente a la regresión logística en promedio** en esta prueba: la diferencia de AUC es +0.00460, mientras que la desviación estándar del AUC entre semillas del MLP es 0.00435. Esa desviación mide sensibilidad a la inicialización sobre el mismo conjunto y no es un intervalo de confianza; por tanto, no permite afirmar que la ventaja sea estadísticamente concluyente o que se mantenga en otros periodos. El MLP también obtiene Brier y log-loss medios menores, aunque la curva de calibración muestra subestimación en varios grupos de riesgo.

En la búsqueda de hiperparámetros, 150 ensayos iniciales alcanzaron AUC de validación 0.67338 y los 30 ensayos de refinamiento 0.67471; la ganancia puntual fue 0.00134. La configuración seleccionada obtuvo AUC medio 0.66911 ± 0.00244 al repetir cinco semillas. El refinamiento aportó una mejora pequeña frente a la variación entre semillas, señal de rendimientos decrecientes en el espacio explorado.

La configuración elegida terminó siendo una red compacta de 2 capas de 64 neuronas y 7.105 parámetros, con dropout 0.333 y regularización L1. Que la búsqueda no favorezca una arquitectura más grande es compatible con que las 44 variables procesadas disponibles aporten relaciones que la regresión logística ya captura en buena medida, o con que las interacciones no lineales adicionales no generalicen mejor. La optimización no demuestra cuál de estas explicaciones es la causa.

También hay un cambio temporal marcado en la tasa observada: 16.30% en entrenamiento, 22.95% en validación y 26.22% en prueba. Esto apunta a cambio de población o de madurez/observabilidad de las etiquetas. La curva de aprendizaje muestra que la pérdida de entrenamiento continúa bajando mientras la de validación queda más alta y alcanza su mínimo cerca de la época 11. Aunque se aplicaron dropout y L1, ese patrón sugiere que el límite principal puede estar en la generalización temporal y la información predictiva disponible, no solo en encontrar otra combinación de hiperparámetros.

Por ello, los resultados también pueden estar limitados por la estructura y cobertura de los datos y por cómo se forman los conjuntos de entrenamiento, validación y prueba. La separación temporal es apropiada para probar el desempeño en cohortes posteriores, pero aquí coincide con diferencias grandes en la tasa etiquetada y los estados de los préstamos pueden haberse observado después de originarlos. Además, una parte importante del archivo no tiene etiqueta definitiva. Estos factores pueden cambiar qué préstamos entran al análisis y qué patrones aprende el modelo; con esta única comparación no es posible atribuirles causalmente el rendimiento observado.

Antes de llevar una PD a producción sería necesario un análisis y procesamiento más exhaustivo: revisar madurez y horizonte de las etiquetas, disponibilidad de cada predictor al momento de originación, faltantes y distribuciones por cohorte, y estabilidad temporal de la población. También convendría construir particiones que reproduzcan el uso previsto y estudiar transformaciones o variables derivadas justificadas por el negocio. Ese trabajo podría mejorar la información disponible y la generalización, aunque no garantiza por sí solo un aumento de métricas.

En conjunto, los resultados no indican que la red sea peor: muestran una **ganancia modesta y sensible al periodo**, sin evidencia suficiente para justificar mayor complejidad o declarar superioridad general frente al baseline. La comparación final debe conservar la prueba temporal como evaluación única y reconocer que el análisis utiliza estados crediticios observados en una fecha posterior a la originación.

## Artefactos

- `metricas_por_semilla.csv`: métricas de validación y prueba por semilla del MLP.
- `metricas_resumen.csv` y `resultados.json`: resumen MLP frente a regresión logística.
- `calibracion_deciles_test.csv`: tasas observadas y PD medias por decil.
- `predicciones_test.csv`: probabilidades de ambos modelos en la prueba temporal.
- `curva_aprendizaje_final.png`, `curva_roc_test.png` y `calibracion_test.png`: gráficos de diagnóstico y comparación.
- `artifacts/models/pd_mlp_final.pt`, `pd_logistic.joblib` y `pd_preprocessor.joblib`: modelos y preprocesador.
