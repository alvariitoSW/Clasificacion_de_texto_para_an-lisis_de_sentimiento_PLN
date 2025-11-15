# Clasificacion de texto para analisis de sentimiento (PLN)

Proyecto desarrollado para la asignatura de Procesamiento de Lenguaje Natural (curso 2025-26) con el objetivo de construir un pipeline reproducible que clasifique la polaridad (negativa, neutral, positiva) de reseñas de BoardGameGeek. El trabajo resume y operacionaliza los entregables descritos en `memoria.pdf`, incorporando procesos de ingenieria de atributos, entrenamiento de modelos y evaluacion cuantitativa.

## Objetivos trazados en la memoria
- Desplegar un flujo integral que vaya desde la ingesta del corpus crudo hasta la evaluacion de modelos en un conjunto hold-out.
- Disenar caracteristicas linguisticas especificas del dominio de juegos de mesa para complementar representaciones basadas en conteo de terminos.
- Comparar varios algoritmos supervisados (Multinomial Naive Bayes, Linear SVM y Random Forest) bajo una misma metodologia de validacion y seleccion de hiperparametros.
- Documentar resultados, hallazgos y recomendaciones para iteraciones futuras.

## Alcance y entregables
1. `pln_p2_7461_01_e1.py`: exploracion del corpus, pipeline de preprocesamiento (limpieza HTML, normalizacion, lematizacion) y extraccion de las 23 caracteristicas linguisticas basadas en VADER y vocabulario de dominio.
2. `pln_p2_7461_01_e2.py`: generacion de representaciones TF-IDF (uni/bi/trigramas), ensamblados hibridos con rasgos linguisticos, balanceo del dataset y persistencia en `data/vector_representations`.
3. `pln_p2_7461_01_e3.py`: gestion de particiones estratificadas (train/validation/test) almacenadas en `data/data_splits`, junto con estadisticas descriptivas del corpus balanceado.
4. `pln_p2_7461_01_e4.py`: entrenamiento y ajuste de hiperparametros mediante grid search para los modelos seleccionados usando las distintas vistas vectoriales.
5. `pln_p2_7461_01_e5.py`: evaluacion final sobre el set de prueba, generacion de reportes (`data/results/*.txt` y `.csv`) y consolidacion de metricas comparativas.

## Datos y preprocesamiento
- Fuente: `data/raw_data/reviews.csv`, reseñas con rating numerico de usuarios de BGG.
- Etiquetado: mapa de ratings a clases (`0-4.99` negativa, `5-6.99` neutral, `7-10` positiva) aplicado con `Corpus.assign_labels`.
- Limpieza: eliminacion de HTML y URLs, lowercasing y normalizacion de espacios como pasos basicos; se extienden con stopwords removal y lematizacion para los experimentos vectoriales.
- Balanceo: estrategia de submuestreo estratificado para igualar las tres clases antes de entrenar modelos supervisados.

## Representaciones vectoriales
- `tfidf_1gram`, `tfidf_2gram`, `tfidf_3gram`: vocabularios de 5k, 10k y 15k terminos respectivamente, con filtros `min_df=2`, `max_df=0.95`.
- `linguistic`: vector denso de 23 caracteristicas (contadores de opinion words, negaciones, intensificadores, senales de dominio, longitudes, puntuacion, scores VADER, etc.).
- `combined_{1,2,3}gram`: concatenacion de TF-IDF con el vector linguistic, preservando magnitudes mediante `scipy.sparse` y almacenando salidas en formato `.npz`.

## Modelos y entrenamiento
- Algoritmos: Multinomial Naive Bayes, LinearSVC y Random Forest.
- Tunearon sus hiperparametros via grid search sobre el conjunto de validacion (ej.: `C`, `loss`, `max_iter` para SVM; `n_estimators`, `max_depth`, `min_samples_split/leaf` para RF).
- La metrica objetivo fue F1-macro, reportando adicionalmente accuracy, precision y recall ponderados para facilitar comparativas entre clases balanceadas.

## Resultados (resumen de `data/results/informe_tecnico.txt`)
- Mejor configuracion: Random Forest con caracteristicas linguisticas puras.
- Rendimiento en test: accuracy 1.0000, precision 1.0000, recall 1.0000, F1 1.0000 (macro y weighted), con una unica confusion entre clases.
- Las vistas combinadas (n-gramas + rasgos linguisticos) ofrecieron mejoras incrementales sobre TF-IDF puro, pero no superaron al modelo exclusivamente linguistic.
- Se recomienda revisar potenciales fugas de informacion si se replica en otros dominios, aunque la lista de features y particiones actuales no incluye identificadores ni etiquetas.

## Estructura del repositorio
- `corpus/`: modulos reutilizables para lectura (`reader.py`), preprocesamiento (`preprocessing.py`), representacion (`vector_manager.py`), ingenieria de rasgos (`feature_extractor.py`) y persistencia (`persistence.py`).
- `data/raw_data/`: fuente original.
- `data/processed_data/`: estadisticas, corpus balanceado y splits etiquetados.
- `data/vector_representations/`: matrices dispersas `.npz` para train/test por tipo de vector.
- `data/results/`: reportes cuantitativos (`classification_results.csv`, `best_model_report.txt`, `informe_tecnico.txt`).
- `memoria.pdf`: documento academico con la justificacion metodologica y el relato completo del proyecto.

## Requisitos y ejecucion
1. Crear entorno e instalar dependencias: `pip install -r requirements.txt`.
2. Ejecutar los scripts `pln_p2_7461_01_e{1..5}.py` en orden para regenerar el flujo completo (cada script imprime trazas guiadas y guarda sus salidas en `data/`).
3. Ajustar rutas o limites de documentos mediante los argumentos definidos en `Corpus.load` y configuraciones internas si se trabaja con nuevas particiones.

## Documentacion
- `memoria.pdf`: referencia normativa del trabajo, objetivos y conclusiones.
- `data/results/informe_tecnico.txt`: resumen ejecutiv o auto-generado con metricas y ranking de modelos.
- `data/results/best_model_report.txt` y `classification_results.csv`: bitacora de hiperparametros y resultados por combinacion modelo-representacion.