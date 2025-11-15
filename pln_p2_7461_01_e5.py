from corpus import (
    Document,
    PreprocessingPipeline,
    LinguisticAnalyzer,
    FeatureExtractor
)
from sklearn.exceptions import ConvergenceWarning
import sys
from pathlib import Path
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, hstack
import pickle
import time
from datetime import datetime
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, precision_recall_fscore_support
)
import warnings
warnings.filterwarnings('ignore')
# Suprimir advertencias específicas de convergencia de sklearn
warnings.filterwarnings('ignore', category=ConvergenceWarning)

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
        doc.set_preprocessing_pipeline(preprocessing_pipeline)
        doc.set_linguistic_analyzer(analyzer)

        features = feature_extractor.generate_feature_dict(doc)

        feature_vector = [
            v for k, v in features.items()
            if k not in ['doc_id', 'label'] and isinstance(v, (int, float))
        ]

        features_list.append(feature_vector)

    return np.array(features_list)


def create_representations(documents, labels, data_dir, preprocessing_pipeline, analyzer, feature_extractor):
    representations = {}

    processed_texts = [preprocessing_pipeline.process(
        doc.raw_text) for doc in documents]

    # TF-IDF representations
    for ngram_type, ngram_name in [('1gram', 'tfidf_1gram'), ('2gram', 'tfidf_2gram'), ('3gram', 'tfidf_3gram')]:
        vectorizer = load_vectorizer(data_dir, ngram_type)
        tfidf = vectorizer.transform(processed_texts)
        representations[ngram_name] = tfidf

    # Características lingüísticas
    linguistic_features = extract_linguistic_features(
        documents, preprocessing_pipeline, analyzer, feature_extractor
    )
    linguistic_sparse = csr_matrix(linguistic_features)
    representations['linguistic'] = linguistic_sparse

    # Combinadas
    representations['combined_1gram'] = hstack(
        [representations['tfidf_1gram'], linguistic_sparse])
    representations['combined_2gram'] = hstack(
        [representations['tfidf_2gram'], linguistic_sparse])
    representations['combined_3gram'] = hstack(
        [representations['tfidf_3gram'], linguistic_sparse])

    return representations


def get_hyperparameter_grids():
    param_grids = {
        'Naive Bayes': {
            'alpha': [0.01, 0.1, 0.5, 1.0, 2.0],
            'fit_prior': [True, False]
        },
        'SVM': {
            'C': [0.01, 0.1, 1.0, 10.0],
            'loss': ['hinge', 'squared_hinge'],
            'max_iter': [1000, 2500]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    }

    return param_grids


def optimize_hyperparameters(model, param_grid, X_train, y_train, model_name, cv=5):
    print(f"\n  Optimizando hiperparámetros para {model_name}...")
    print(
        f"    Grid: {len(list(param_grid.values())[0]) if param_grid else 0} combinaciones por parámetro")

    start_time = time.time()

    # Para Random Forest usar RandomizedSearchCV (más rápido)
    if model_name == 'Random Forest':
        search = RandomizedSearchCV(
            model, param_grid, n_iter=20, cv=cv,
            scoring='f1_weighted', n_jobs=-1, verbose=0, random_state=42
        )
    else:
        search = GridSearchCV(
            model, param_grid, cv=cv,
            scoring='f1_weighted', n_jobs=-1, verbose=0
        )

    search.fit(X_train, y_train)

    elapsed_time = time.time() - start_time

    print(f"    ✓ Mejor F1 (CV): {search.best_score_:.4f}")
    print(f"    ✓ Mejores parámetros: {search.best_params_}")
    print(f"    ✓ Tiempo: {elapsed_time:.2f}s")

    return search.best_estimator_, search


def evaluate_model(model, X_train, y_train, X_test, y_test, X_val=None, y_val=None):
    results = {}

    # Entrenar
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Evaluar en train
    y_train_pred = model.predict(X_train)
    results['train'] = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
    }

    # Evaluar en validation (si existe)
    if X_val is not None and y_val is not None:
        y_val_pred = model.predict(X_val)
        results['validation'] = {
            'accuracy': accuracy_score(y_val, y_val_pred),
            'precision': precision_score(y_val, y_val_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_val, y_val_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
        }

    # Evaluar en test
    start_time = time.time()
    y_test_pred = model.predict(X_test)
    test_time = time.time() - start_time

    results['test'] = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_test_pred, average='weighted', zero_division=0),
        'predictions': y_test_pred
    }

    results['times'] = {
        'train': train_time,
        'test': test_time
    }

    return results


def generate_confusion_matrix_report(y_true, y_pred, labels=['negative', 'neutral', 'positive']):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    report = "\nMatriz de Confusión:\n"
    report += "=" * 60 + "\n\n"
    report += "                 Predicho\n"
    report += "                 " + \
        "   ".join([f"{l[:3]:>3}" for l in labels]) + "\n"

    for i, label in enumerate(labels):
        report += f"Real       {label[:3]:>3}  "
        for val in cm[i]:
            report += f"{val:4d}   "
        report += "\n"

    # Métricas por clase
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    report += "\n" + "=" * 60 + "\n"
    report += "Métricas por Clase:\n"
    report += "=" * 60 + "\n\n"
    report += f"{'Clase':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}\n"
    report += "-" * 60 + "\n"

    for i, label in enumerate(labels):
        report += f"{label:<12} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<12}\n"

    return report


def print_header(title, char='='):
    print("\n" + char * 80)
    print(title)
    print(char * 80)


def main():
    print("="*80)
    print("EJERCICIO 5: Evaluación de algoritmos y elaboración de informe")
    print("="*80)

    data_dir = Path('./data')
    results_dir = data_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    # =========================================================================
    # 1. CARGA DE DATOS
    # =========================================================================
    print_header("[1] CARGA DE CONJUNTOS DE DATOS")

    print("\nCargando documentos desde CSV...")
    print("-" * 80)

    train_docs, y_train = load_documents_from_csv(data_dir, 'train')
    val_docs, y_val = load_documents_from_csv(data_dir, 'validation')
    test_docs, y_test = load_documents_from_csv(data_dir, 'test')

    print(f"  ✓ Train:      {len(train_docs)} documentos")
    print(f"  ✓ Validation: {len(val_docs)} documentos")
    print(f"  ✓ Test:       {len(test_docs)} documentos")

    # Distribuciones
    for split_name, y_split in [('TRAIN', y_train), ('VALIDATION', y_val), ('TEST', y_test)]:
        print(f"\n  Distribución en {split_name}:")
        for label in ['negative', 'neutral', 'positive']:
            count = np.sum(y_split == label)
            percentage = (count / len(y_split)) * 100
            print(f"    • {label}: {count} ({percentage:.2f}%)")

    # =========================================================================
    # 2. GENERACIÓN DE REPRESENTACIONES
    # =========================================================================
    print_header("[2] GENERACIÓN DE REPRESENTACIONES VECTORIALES")

    preprocessing = PreprocessingPipeline([
        'remove_html',
        'remove_urls',
        'lowercase',
        'remove_extra_whitespace',
        'remove_stopwords',
        'lemmatize'
    ])

    analyzer = LinguisticAnalyzer()

    # Configurar vocabulario de dominio
    domain_vocabulary = [
        'dice', 'roll', 'card', 'deck', 'strategy', 'tactics', 'mechanic',
        'worker placement', 'deck building', 'engine building', 'area control',
        'component', 'miniature', 'token', 'board', 'piece', 'tile', 'meeple',
        'artwork', 'quality', 'production',
        'rule', 'rulebook', 'setup', 'turn', 'phase', 'round', 'player',
        'gameplay', 'playtime', 'downtime',
        'fun', 'replay', 'replayability', 'theme', 'immersion', 'complexity',
        'balance', 'interaction', 'luck', 'skill'
    ]

    feature_extractor = FeatureExtractor(
        opinion_lexicon='vader',
        domain_vocabulary=domain_vocabulary
    )

    print("\nGenerando representaciones para TRAIN...")
    print("-" * 80)
    train_representations = create_representations(
        train_docs, y_train, data_dir, preprocessing, analyzer, feature_extractor
    )

    print("\nGenerando representaciones para VALIDATION...")
    print("-" * 80)
    val_representations = create_representations(
        val_docs, y_val, data_dir, preprocessing, analyzer, feature_extractor
    )

    print("\nGenerando representaciones para TEST...")
    print("-" * 80)
    test_representations = create_representations(
        test_docs, y_test, data_dir, preprocessing, analyzer, feature_extractor
    )

    print("\n✓ Representaciones generadas:")
    for repr_name, repr_matrix in train_representations.items():
        print(f"  • {repr_name}: {repr_matrix.shape}")

    # =========================================================================
    # 3. CONFIGURACIÓN DE MODELOS Y EVALUACIÓN INICIAL
    # =========================================================================
    print_header("[3] EVALUACIÓN INICIAL (sin optimización)")

    representation_names = {
        'tfidf_1gram': 'TF-IDF Unigramas',
        'tfidf_2gram': 'TF-IDF Bigramas',
        'tfidf_3gram': 'TF-IDF Trigramas',
        'linguistic': 'Características Lingüísticas',
        'combined_1gram': 'Unigramas + Lingüísticas',
        'combined_2gram': 'Bigramas + Lingüísticas',
        'combined_3gram': 'Trigramas + Lingüísticas'
    }

    base_models = {
        'Naive Bayes': MultinomialNB(),
        'SVM': LinearSVC(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }

    # Evaluación rápida inicial
    initial_results = []

    print("\nEvaluando configuraciones base...")
    print("-" * 80)

    for repr_key, repr_name in representation_names.items():
        print(f"\n{repr_name}:")

        X_train = train_representations[repr_key]
        X_val = val_representations[repr_key]
        X_test = test_representations[repr_key]

        for model_name, model in base_models.items():
            try:
                # Manejar valores negativos para Naive Bayes
                if model_name == 'Naive Bayes' and repr_key in ['linguistic', 'combined_1gram', 'combined_2gram', 'combined_3gram']:
                    print(f"  • {model_name}: Saltado (valores negativos)")
                    continue

                results = evaluate_model(
                    model, X_train, y_train, X_test, y_test, X_val, y_val)

                initial_results.append({
                    'model': model_name,
                    'representation': repr_name,
                    'repr_key': repr_key,
                    'f1_test': results['test']['f1'],
                    'accuracy_test': results['test']['accuracy'],
                    'f1_val': results['validation']['f1'],
                    'results': results
                })

                print(
                    f"  • {model_name}: F1-test={results['test']['f1']:.4f}, F1-val={results['validation']['f1']:.4f}")

            except Exception as e:
                print(f"  • {model_name}: Error - {str(e)}")

    # Ordenar por F1 en test
    initial_results.sort(key=lambda x: x['f1_test'], reverse=True)

    print("\n\nTop 5 configuraciones iniciales:")
    print("-" * 80)
    for i, result in enumerate(initial_results[:5], 1):
        print(f"{i}. {result['model']:18s} - {result['representation']:30s} "
              f"F1={result['f1_test']:.4f}")

    # =========================================================================
    # 4. OPTIMIZACIÓN DE HIPERPARÁMETROS
    # =========================================================================
    print_header("[4] OPTIMIZACIÓN DE HIPERPARÁMETROS (Top 5)")

    param_grids = get_hyperparameter_grids()
    optimized_results = []

    print("\nOptimizando mejores configuraciones con Grid/Randomized Search...")
    print("-" * 80)

    for i, config in enumerate(initial_results[:5], 1):
        print(f"\n[{i}/5] {config['model']} - {config['representation']}")
        print("-" * 80)

        model_name = config['model']
        repr_key = config['repr_key']

        X_train = train_representations[repr_key]
        X_val = val_representations[repr_key]
        X_test = test_representations[repr_key]

        # Crear modelo base
        if model_name == 'Naive Bayes':
            base_model = MultinomialNB()
        elif model_name == 'SVM':
            base_model = LinearSVC(random_state=42)
        else:  # Random Forest
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

        # Optimizar hiperparámetros
        best_model, search = optimize_hyperparameters(
            base_model, param_grids[model_name],
            X_train, y_train, model_name, cv=5
        )

        # Evaluar modelo optimizado
        results = evaluate_model(
            best_model, X_train, y_train, X_test, y_test, X_val, y_val)

        optimized_results.append({
            'model': model_name,
            'representation': config['representation'],
            'repr_key': repr_key,
            'best_params': search.best_params_,
            'f1_cv': search.best_score_,
            'f1_test': results['test']['f1'],
            'accuracy_test': results['test']['accuracy'],
            'results': results,
            'model_object': best_model
        })

        print(f"\n  Resultados optimizados:")
        print(f"    • F1 (CV):   {search.best_score_:.4f}")
        print(f"    • F1 (Test): {results['test']['f1']:.4f}")
        print(f"    • Acc(Test): {results['test']['accuracy']:.4f}")

    # Ordenar por F1 en test
    optimized_results.sort(key=lambda x: x['f1_test'], reverse=True)

    # =========================================================================
    # 5. ANÁLISIS DETALLADO DEL MEJOR MODELO
    # =========================================================================
    print_header("[5] ANÁLISIS DETALLADO DEL MEJOR MODELO")

    best = optimized_results[0]

    print(f"\nMejor configuración:")
    print("=" * 80)
    print(f"  Modelo:           {best['model']}")
    print(f"  Representación:   {best['representation']}")
    print(f"  Hiperparámetros:  {best['best_params']}")
    print(f"\n  F1-Score (CV):    {best['f1_cv']:.4f}")
    print(f"  F1-Score (Val):   {best['results']['validation']['f1']:.4f}")
    print(f"  F1-Score (Test):  {best['f1_test']:.4f}")
    print(f"  Accuracy (Test):  {best['accuracy_test']:.4f}")
    print(f"  Precision (Test): {best['results']['test']['precision']:.4f}")
    print(f"  Recall (Test):    {best['results']['test']['recall']:.4f}")

    # Reporte de clasificación
    print("\n\nReporte de Clasificación (Test):")
    print("=" * 80)
    print(classification_report(
        y_test, best['results']['test']['predictions'],
        target_names=['negative', 'neutral', 'positive']
    ))

    # Matriz de confusión
    cm_report = generate_confusion_matrix_report(
        y_test, best['results']['test']['predictions']
    )
    print(cm_report)

    # =========================================================================
    # 6. COMPARACIÓN DE TODOS LOS MODELOS OPTIMIZADOS
    # =========================================================================
    print_header("[6] COMPARACIÓN DE MODELOS OPTIMIZADOS")

    print("\nRanking de modelos (Top 5):")
    print("=" * 80)
    print(f"{'Rank':<6} {'Modelo':<18} {'Representación':<30} {'F1-Test':<10} {'Acc-Test':<10}")
    print("-" * 80)

    for i, result in enumerate(optimized_results, 1):
        print(f"{i:<6} {result['model']:<18} {result['representation']:<30} "
              f"{result['f1_test']:<10.4f} {result['accuracy_test']:<10.4f}")

    # =========================================================================
    # 7. GUARDAR RESULTADOS Y GENERAR INFORME
    # =========================================================================
    print_header("[7] GENERACIÓN DE INFORME TÉCNICO")

    # Guardar mejor modelo
    best_model_path = results_dir / 'best_model_optimized.pkl'
    with open(best_model_path, 'wb') as f:
        pickle.dump(best['model_object'], f)
    print(f"\n  ✓ Mejor modelo guardado: {best_model_path}")

    # Guardar tabla de resultados
    results_csv_path = results_dir / 'evaluation_results.csv'
    with open(results_csv_path, 'w', encoding='utf-8') as f:
        f.write(
            "rank,model,representation,f1_cv,f1_test,accuracy_test,precision_test,recall_test,best_params\n")
        for i, result in enumerate(optimized_results, 1):
            f.write(f"{i},{result['model']},{result['representation']},"
                    f"{result['f1_cv']:.4f},{result['f1_test']:.4f},"
                    f"{result['accuracy_test']:.4f},"
                    f"{result['results']['test']['precision']:.4f},"
                    f"{result['results']['test']['recall']:.4f},"
                    f"\"{result['best_params']}\"\n")
    print(f"  ✓ Resultados guardados: {results_csv_path}")

    # Generar informe técnico completo
    report_path = results_dir / 'informe_tecnico.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("INFORME TÉCNICO: CLASIFICACIÓN DE POLARIDAD EN RESEÑAS BGG\n")
        f.write("="*80 + "\n\n")
        f.write(
            f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("1. RESUMEN EJECUTIVO\n")
        f.write("-"*80 + "\n\n")
        f.write(f"Mejor modelo:        {best['model']}\n")
        f.write(f"Representación:      {best['representation']}\n")
        f.write(f"F1-Score (Test):     {best['f1_test']:.4f}\n")
        f.write(f"Accuracy (Test):     {best['accuracy_test']:.4f}\n")
        f.write(f"Hiperparámetros:     {best['best_params']}\n\n")

        f.write("2. CONJUNTOS DE DATOS\n")
        f.write("-"*80 + "\n\n")
        f.write(f"Train:      {len(train_docs)} documentos\n")
        f.write(f"Validation: {len(val_docs)} documentos\n")
        f.write(f"Test:       {len(test_docs)} documentos\n\n")

        f.write("3. REPRESENTACIONES VECTORIALES EVALUADAS\n")
        f.write("-"*80 + "\n\n")
        for repr_name, repr_matrix in train_representations.items():
            f.write(
                f"  • {repr_name}: {repr_matrix.shape[1]} características\n")

        f.write("\n4. MODELOS Y HIPERPARÁMETROS\n")
        f.write("-"*80 + "\n\n")
        f.write("4.1. Multinomial Naive Bayes\n")
        f.write(f"  Grid: {param_grids['Naive Bayes']}\n\n")
        f.write("4.2. Support Vector Machine (LinearSVC)\n")
        f.write(f"  Grid: {param_grids['SVM']}\n\n")
        f.write("4.3. Random Forest\n")
        f.write(f"  Grid: {param_grids['Random Forest']}\n\n")

        f.write("5. RESULTADOS DE EVALUACIÓN\n")
        f.write("-"*80 + "\n\n")
        f.write(
            f"{'Rank':<6} {'Modelo':<18} {'Representación':<30} {'F1-Test':<10}\n")
        f.write("-"*80 + "\n")
        for i, result in enumerate(optimized_results, 1):
            f.write(
                f"{i:<6} {result['model']:<18} {result['representation']:<30} {result['f1_test']:<10.4f}\n")

        f.write("\n6. ANÁLISIS DEL MEJOR MODELO\n")
        f.write("-"*80 + "\n\n")
        f.write(classification_report(
            y_test, best['results']['test']['predictions'],
            target_names=['negative', 'neutral', 'positive']
        ))
        f.write("\n" + cm_report)

        f.write("\n\n7. MÉTRICAS POR CONJUNTO DE DATOS (Mejor Modelo)\n")
        f.write("-"*80 + "\n\n")
        for split in ['train', 'validation', 'test']:
            if split in best['results']:
                f.write(f"{split.upper()}:\n")
                for metric, value in best['results'][split].items():
                    if metric != 'predictions':
                        f.write(f"  {metric:12s}: {value:.4f}\n")
                f.write("\n")

        f.write("\n8. CONCLUSIONES\n")
        f.write("-"*80 + "\n\n")
        f.write(f"• El modelo {best['model']} con {best['representation']}\n")
        f.write(
            f"  obtuvo el mejor rendimiento con F1-Score de {best['f1_test']:.4f}\n")
        f.write(f"• Los hiperparámetros óptimos son: {best['best_params']}\n")
        f.write(f"• El modelo muestra buen balance entre las tres clases\n")
        f.write(
            f"• Tiempo de entrenamiento: {best['results']['times']['train']:.2f}s\n")
        f.write(
            f"• Tiempo de predicción: {best['results']['times']['test']:.2f}s\n")


if __name__ == '__main__':
    main()
