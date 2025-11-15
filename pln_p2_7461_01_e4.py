from corpus import (
    Document,
    PreprocessingPipeline,
    LinguisticAnalyzer,
    FeatureExtractor
)
import sys
from pathlib import Path
import numpy as np
import polars as pl
from scipy.sparse import hstack, csr_matrix
import pickle
import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def load_documents_from_csv(data_dir, split='train'):

    processed_dir = Path(data_dir) / 'processed_data'
    file_path = processed_dir / f"{split}_set.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")

    # Leer con schema_overrides para evitar errores de parsing
    df = pl.read_csv(file_path, schema_overrides={'user': pl.Utf8})

    # Crear objetos Document
    documents = []
    labels = []

    for row in df.iter_rows(named=True):
        doc = Document(
            doc_id=row['doc_id'],
            raw_text=row['raw_text'],
            rating=row['rating'],
            metadata={
                'game_id': row['game_id'],
                'user': row['user'],
                'timestamp': row.get('timestamp'),
                'text_length': row.get('text_length')
            }
        )
        doc.label = row['label']
        documents.append(doc)
        labels.append(row['label'])

    return documents, np.array(labels)


def load_vectorizer(data_dir, ngram_type):
    vectors_dir = Path(data_dir) / 'vector_representations'
    vectorizer_path = vectors_dir / f"vectorizer_{ngram_type}.pkl"

    if not vectorizer_path.exists():
        raise FileNotFoundError(
            f"No se encontró el vectorizador: {vectorizer_path}")

    with open(vectorizer_path, 'rb') as f:
        return pickle.load(f)


def extract_linguistic_features(documents, preprocessing_pipeline, analyzer, feature_extractor):
    features_list = []

    for doc in documents:
        # Establecer referencias en el documento
        doc.set_preprocessing_pipeline(preprocessing_pipeline)
        doc.set_linguistic_analyzer(analyzer)

        # Extraer características
        features = feature_extractor.generate_feature_dict(doc)

        # Convertir a vector (excluir 'doc_id' y 'label')
        feature_vector = [
            v for k, v in features.items()
            if k not in ['doc_id', 'label'] and isinstance(v, (int, float))
        ]

        features_list.append(feature_vector)

    return np.array(features_list)


def create_representations(documents, labels, data_dir, preprocessing_pipeline, analyzer, feature_extractor):
    representations = {}

    # Preprocesar todos los textos
    processed_texts = [preprocessing_pipeline.process(
        doc.raw_text) for doc in documents]

    # 1. TF-IDF Unigramas
    print("\n  Generando TF-IDF Unigramas...")
    vectorizer_1gram = load_vectorizer(data_dir, '1gram')
    tfidf_1gram = vectorizer_1gram.transform(processed_texts)
    representations['tfidf_1gram'] = tfidf_1gram
    print(f"    ✓ Shape: {tfidf_1gram.shape}")

    # 2. TF-IDF Bigramas
    print("\n  Generando TF-IDF Bigramas...")
    vectorizer_2gram = load_vectorizer(data_dir, '2gram')
    tfidf_2gram = vectorizer_2gram.transform(processed_texts)
    representations['tfidf_2gram'] = tfidf_2gram
    print(f"    ✓ Shape: {tfidf_2gram.shape}")

    # 3. TF-IDF Trigramas
    print("\n  Generando TF-IDF Trigramas...")
    vectorizer_3gram = load_vectorizer(data_dir, '3gram')
    tfidf_3gram = vectorizer_3gram.transform(processed_texts)
    representations['tfidf_3gram'] = tfidf_3gram
    print(f"    ✓ Shape: {tfidf_3gram.shape}")

    # 4. Características Lingüísticas
    print("\n  Generando Características Lingüísticas...")
    linguistic_features = extract_linguistic_features(
        documents, preprocessing_pipeline, analyzer, feature_extractor
    )
    linguistic_sparse = csr_matrix(linguistic_features)
    representations['linguistic'] = linguistic_sparse
    print(f"    ✓ Shape: {linguistic_sparse.shape}")

    # 5. Combinadas: Unigramas + Lingüísticas
    print("\n  Generando Combinación Unigramas + Lingüísticas...")
    combined_1gram = hstack([tfidf_1gram, linguistic_sparse])
    representations['combined_1gram'] = combined_1gram
    print(f"    ✓ Shape: {combined_1gram.shape}")

    # 6. Combinadas: Bigramas + Lingüísticas
    print("\n  Generando Combinación Bigramas + Lingüísticas...")
    combined_2gram = hstack([tfidf_2gram, linguistic_sparse])
    representations['combined_2gram'] = combined_2gram
    print(f"    ✓ Shape: {combined_2gram.shape}")

    # 7. Combinadas: Trigramas + Lingüísticas
    print("\n  Generando Combinación Trigramas + Lingüísticas...")
    combined_3gram = hstack([tfidf_3gram, linguistic_sparse])
    representations['combined_3gram'] = combined_3gram
    print(f"    ✓ Shape: {combined_3gram.shape}")

    return representations


def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name, representation_name):
    print(f"\n  Entrenando {model_name}...")

    # Entrenar
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Predecir
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"    ✓ Accuracy:  {accuracy:.4f}")
    print(f"    ✓ Precision: {precision:.4f}")
    print(f"    ✓ Recall:    {recall:.4f}")
    print(f"    ✓ F1-Score:  {f1:.4f}")
    print(f"    ✓ Tiempo entrenamiento: {train_time:.2f}s")
    print(f"    ✓ Tiempo predicción:    {predict_time:.2f}s")

    return {
        'model_name': model_name,
        'representation': representation_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'train_time': train_time,
        'predict_time': predict_time,
        'predictions': y_pred,
        'model': model
    }


def print_header(title, char='='):
    """Imprimir encabezado formateado."""
    print("\n" + char * 80)
    print(title)
    print(char * 80)


def main():
    """Programa principal para construcción de modelos de clasificación."""

    print("="*80)
    print("EJERCICIO 4: Construcción de modelos de clasificación")
    print("="*80)

    data_dir = Path('./data')

    # =========================================================================
    # 1. CONFIGURACIÓN DE MODELOS
    # =========================================================================
    models = {
        'Naive Bayes': MultinomialNB(),
        'SVM': LinearSVC(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }

    # =========================================================================
    # 2. CONFIGURACIÓN DE REPRESENTACIONES
    # =========================================================================
    representation_names = {
        'tfidf_1gram': 'TF-IDF Unigramas',
        'tfidf_2gram': 'TF-IDF Bigramas',
        'tfidf_3gram': 'TF-IDF Trigramas',
        'linguistic': 'Características Lingüísticas',
        'combined_1gram': 'Unigramas + Lingüísticas',
        'combined_2gram': 'Bigramas + Lingüísticas',
        'combined_3gram': 'Trigramas + Lingüísticas'
    }

    # =========================================================================
    # 3. CARGA DE DATOS
    # =========================================================================
    print_header("[3] CARGA DE DATOS")

    print("\nCargando documentos desde CSV...")
    print("-" * 80)

    # Cargar documentos de train y test
    train_docs, y_train = load_documents_from_csv(data_dir, 'train')
    test_docs, y_test = load_documents_from_csv(data_dir, 'test')

    print(f"  ✓ Train: {len(train_docs)} documentos")
    print(f"  ✓ Test:  {len(test_docs)} documentos")

    # Mostrar distribución de clases
    print("\n  Distribución en TRAIN:")
    for label in ['negative', 'neutral', 'positive']:
        count = np.sum(y_train == label)
        percentage = (count / len(y_train)) * 100
        print(f"    • {label}: {count} ({percentage:.2f}%)")

    print("\n  Distribución en TEST:")
    for label in ['negative', 'neutral', 'positive']:
        count = np.sum(y_test == label)
        percentage = (count / len(y_test)) * 100
        print(f"    • {label}: {count} ({percentage:.2f}%)")

    # =========================================================================
    # 4. GENERACIÓN DE REPRESENTACIONES
    # =========================================================================
    print_header("[4] GENERACIÓN DE REPRESENTACIONES VECTORIALES")

    # Inicializar componentes necesarios
    preprocessing = PreprocessingPipeline([
        'remove_html',
        'remove_urls',
        'lowercase',
        'remove_extra_whitespace',
        'remove_stopwords',
        'lemmatize'
    ])

    analyzer = LinguisticAnalyzer()
    feature_extractor = FeatureExtractor()

    # Generar representaciones para train
    print("\nGenerando representaciones para TRAIN...")
    print("-" * 80)
    train_representations = create_representations(
        train_docs, y_train, data_dir, preprocessing, analyzer, feature_extractor
    )

    # Generar representaciones para test
    print("\nGenerando representaciones para TEST...")
    print("-" * 80)
    test_representations = create_representations(
        test_docs, y_test, data_dir, preprocessing, analyzer, feature_extractor
    )

    # =========================================================================
    # 5. ENTRENAMIENTO Y EVALUACIÓN
    # =========================================================================
    print_header("[5] ENTRENAMIENTO Y EVALUACIÓN DE MODELOS")

    results = []

    # Evaluar cada combinación de modelo y representación
    for repr_key, repr_name in representation_names.items():
        print_header(f"REPRESENTACIÓN: {repr_name}", char='=')

        X_train = train_representations[repr_key]
        X_test = test_representations[repr_key]

        print(f"\n  Train shape: {X_train.shape}")
        print(f"  Test shape:  {X_test.shape}")

        # Verificar si hay valores negativos (importante para Naive Bayes)
        if repr_key in ['linguistic', 'combined_1gram', 'combined_2gram', 'combined_3gram']:
            X_train_dense = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
            if np.any(X_train_dense < 0):
                print(
                    "\n  ⚠️  Detectados valores negativos, normalizando para Naive Bayes...")
                scaler = MinMaxScaler()
                X_train_scaled = scaler.fit_transform(X_train_dense)
                X_test_scaled = scaler.transform(
                    X_test.toarray() if hasattr(X_test, 'toarray') else X_test)
                X_train_nb = csr_matrix(X_train_scaled)
                X_test_nb = csr_matrix(X_test_scaled)
            else:
                X_train_nb = X_train
                X_test_nb = X_test
        else:
            X_train_nb = X_train
            X_test_nb = X_test

        # Entrenar cada modelo
        for model_name, model in models.items():
            try:
                # Usar datos normalizados para Naive Bayes si es necesario
                if model_name == 'Naive Bayes' and 'nb' in locals():
                    result = train_and_evaluate(
                        model, X_train_nb, y_train, X_test_nb, y_test,
                        model_name, repr_name
                    )
                else:
                    result = train_and_evaluate(
                        model, X_train, y_train, X_test, y_test,
                        model_name, repr_name
                    )
                results.append(result)
            except Exception as e:
                print(f"  ✗ Error entrenando {model_name}: {str(e)}")
                continue

    # =========================================================================
    # 6. ANÁLISIS DE RESULTADOS
    # =========================================================================
    print_header("[6] ANÁLISIS DE RESULTADOS")

    if not results:
        print("\n✗ No se obtuvieron resultados para analizar")
        return

    # Ordenar por F1-Score
    results_sorted = sorted(results, key=lambda x: x['f1_score'], reverse=True)

    # Top 10 modelos
    print("\nTop 10 mejores configuraciones (por F1-Score):")
    print("-" * 80)
    print(f"{'Rank':<6} {'Modelo':<18} {'Representación':<30} {'F1':<8} {'Acc':<8}")
    print("-" * 80)

    for i, result in enumerate(results_sorted[:10], 1):
        print(f"{i:<6} {result['model_name']:<18} {result['representation']:<30} "
              f"{result['f1_score']:.4f}   {result['accuracy']:.4f}")

    # Mejor modelo
    best_result = results_sorted[0]
    print("\n\nMejor configuración:")
    print("=" * 80)
    print(f"  Modelo:         {best_result['model_name']}")
    print(f"  Representación: {best_result['representation']}")
    print(f"  Accuracy:       {best_result['accuracy']:.4f}")
    print(f"  Precision:      {best_result['precision']:.4f}")
    print(f"  Recall:         {best_result['recall']:.4f}")
    print(f"  F1-Score:       {best_result['f1_score']:.4f}")
    print(f"  Tiempo train:   {best_result['train_time']:.2f}s")
    print(f"  Tiempo test:    {best_result['predict_time']:.2f}s")

    # Reporte de clasificación detallado
    print("\n\nReporte de clasificación del mejor modelo:")
    print("-" * 80)
    print(classification_report(y_test, best_result['predictions'],
                                target_names=['negative', 'neutral', 'positive']))

    # Matriz de confusión
    print("\nMatriz de confusión:")
    print("-" * 80)
    cm = confusion_matrix(y_test, best_result['predictions'],
                          labels=['negative', 'neutral', 'positive'])
    print("\n                 Predicho")
    print("                 neg    neu    pos")
    print("           neg  ", end="")
    for val in cm[0]:
        print(f"{val:4d}   ", end="")
    print("\nReal       neu  ", end="")
    for val in cm[1]:
        print(f"{val:4d}   ", end="")
    print("\n           pos  ", end="")
    for val in cm[2]:
        print(f"{val:4d}   ", end="")
    print()

    # =========================================================================
    # 7. GUARDAR RESULTADOS
    # =========================================================================
    print_header("[7] GUARDANDO RESULTADOS")

    results_dir = data_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    # Guardar mejor modelo
    best_model_path = results_dir / 'best_model.pkl'
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_result['model'], f)
    print(f"\n  ✓ Mejor modelo guardado en: {best_model_path}")

    # Guardar tabla de resultados
    results_csv_path = results_dir / 'classification_results.csv'
    with open(results_csv_path, 'w', encoding='utf-8') as f:
        f.write(
            "model,representation,accuracy,precision,recall,f1_score,train_time,predict_time\n")
        for result in results_sorted:
            f.write(f"{result['model_name']},{result['representation']},"
                    f"{result['accuracy']:.4f},{result['precision']:.4f},"
                    f"{result['recall']:.4f},{result['f1_score']:.4f},"
                    f"{result['train_time']:.2f},{result['predict_time']:.2f}\n")
    print(f"  ✓ Resultados guardados en: {results_csv_path}")

    # Guardar reporte detallado del mejor modelo
    report_path = results_dir / 'best_model_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MEJOR MODELO - REPORTE DETALLADO\n")
        f.write("="*80 + "\n\n")
        f.write(f"Modelo:         {best_result['model_name']}\n")
        f.write(f"Representación: {best_result['representation']}\n")
        f.write(f"Accuracy:       {best_result['accuracy']:.4f}\n")
        f.write(f"Precision:      {best_result['precision']:.4f}\n")
        f.write(f"Recall:         {best_result['recall']:.4f}\n")
        f.write(f"F1-Score:       {best_result['f1_score']:.4f}\n")
        f.write(f"Tiempo train:   {best_result['train_time']:.2f}s\n")
        f.write(f"Tiempo test:    {best_result['predict_time']:.2f}s\n\n")
        f.write("Reporte de clasificación:\n")
        f.write("-"*80 + "\n")
        f.write(classification_report(y_test, best_result['predictions'],
                                      target_names=['negative', 'neutral', 'positive']))


if __name__ == '__main__':
    main()
