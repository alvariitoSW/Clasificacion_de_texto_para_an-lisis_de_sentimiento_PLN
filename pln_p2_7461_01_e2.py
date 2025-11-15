from corpus import (
    CorpusReader,
    Corpus,
    PreprocessingPipeline,
    LinguisticAnalyzer,
    FeatureExtractor,
    VectorManager,
    PersistenceManager
)
import sys
from pathlib import Path
from scipy.sparse import csr_matrix
import pickle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    print("="*80)
    print("EJERCICIO 2: Generación de representaciones vectoriales de reseñas")
    print("="*80)

    # =========================================================================
    # 1. CARGAR CORPUS (10k documentos)
    # =========================================================================
    print("\n[1] Cargando corpus...")
    print("-" * 80)

    # Inicializar lector
    reader = CorpusReader('./data/raw_data')

    # Crear pipeline de preprocesamiento (incluye filtrado de características)
    preprocessing = PreprocessingPipeline([
        'remove_html',
        'remove_urls',
        'lowercase',
        'remove_extra_whitespace',
        'remove_stopwords',      # Filtrado: eliminar stopwords
        'lemmatize'              # Filtrado: lemmatización para reducir vocabulario
    ])

    # Crear analizador lingüístico
    analyzer = LinguisticAnalyzer()

    # Crear corpus
    corpus = Corpus(
        label_map={
            'positive': [7, 10],
            'negative': [0, 4.99],
            'neutral': [5, 6.99]
        },
        preprocessing_pipeline=preprocessing,
        linguistic_analyzer=analyzer
    )

    # Cargar documentos (límite: 10,000)
    # print("Cargando hasta 10,000 documentos...")
    # corpus.load(reader, max_documents=10000, show_progress=True)
    # print("Cargando todos los documentos...")
    # corpus.load(reader, show_progress=True)
    print("Cargando hasta 1M documentos...")
    corpus.load(reader, max_documents=1000000, show_progress=True)
    corpus.assign_labels()

    print(f"\n✓ Corpus cargado: {len(corpus)} documentos")

    # Balancear dataset
    print("\nBalanceando dataset...")
    balanced_docs = corpus.balance_dataset(
        strategy='subsample', random_seed=42)
    print(f"✓ Dataset balanceado: {len(balanced_docs)} documentos")

    # Crear split train/test
    print("\nCreando split train/test (80/20)...")
    train_docs, test_docs = corpus.create_stratified_split(
        test_size=0.2,
        random_seed=42,
        documents=balanced_docs
    )
    print(f"✓ Train: {len(train_docs)} documentos")
    print(f"✓ Test:  {len(test_docs)} documentos")

    # =========================================================================
    # 2. REPRESENTACIÓN 1: N-GRAMAS Y TF-IDF
    # =========================================================================
    print("\n\n" + "="*80)
    print("[2] REPRESENTACIÓN 1: N-gramas y TF-IDF")
    print("="*80)

    # Inicializar VectorManager
    vector_manager = VectorManager()

    # ---------------------------------------------------------------------
    # 2.1. UNIGRAMAS (n=1)
    # ---------------------------------------------------------------------
    print("\n[2.1] Generando vectores TF-IDF con UNIGRAMAS (n=1)")
    print("-" * 80)

    # Entrenar vectorizador con unigramas
    vectorizer_1gram = vector_manager.fit_tfidf_vectorizer(
        train_docs,
        ngram_range=(1, 1),
        max_features=5000,
        min_df=2,
        max_df=0.95
    )

    print(
        f"\n✓ Vocabulario generado: {len(vectorizer_1gram.vocabulary_)} términos")

    # Transformar documentos
    train_vectors_1gram = vector_manager.transform_tfidf(
        train_docs, ngram_range=(1, 1))
    test_vectors_1gram = vector_manager.transform_tfidf(
        test_docs, ngram_range=(1, 1))

    print(f"\n✓ Vectores generados:")
    print(f"  • Train shape: {train_vectors_1gram.shape}")
    print(f"  • Test shape:  {test_vectors_1gram.shape}")
    sparsity_1gram = (1 - train_vectors_1gram.nnz /
                      (train_vectors_1gram.shape[0] * train_vectors_1gram.shape[1])) * 100
    print(f"  • Sparsity:    {sparsity_1gram:.2f}%")

    # Mostrar ejemplos de términos
    vocab_items = sorted(
        vectorizer_1gram.vocabulary_.items(), key=lambda x: x[1])[:20]
    print(f"\n  Ejemplos de términos en vocabulario:")
    print(f"  {', '.join([term for term, _ in vocab_items[:10]])}")

    # ---------------------------------------------------------------------
    # 2.2. BIGRAMAS (n=2)
    # ---------------------------------------------------------------------
    print("\n\n[2.2] Generando vectores TF-IDF con BIGRAMAS (n=1,2)")
    print("-" * 80)

    # Entrenar vectorizador con bigramas
    vectorizer_2gram = vector_manager.fit_tfidf_vectorizer(
        train_docs,
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2,
        max_df=0.95
    )

    print(
        f"\n✓ Vocabulario generado: {len(vectorizer_2gram.vocabulary_)} términos")

    # Transformar documentos
    train_vectors_2gram = vector_manager.transform_tfidf(
        train_docs, ngram_range=(1, 2))
    test_vectors_2gram = vector_manager.transform_tfidf(
        test_docs, ngram_range=(1, 2))

    print(f"\n✓ Vectores generados:")
    print(f"  • Train shape: {train_vectors_2gram.shape}")
    print(f"  • Test shape:  {test_vectors_2gram.shape}")
    sparsity_2gram = (1 - train_vectors_2gram.nnz /
                      (train_vectors_2gram.shape[0] * train_vectors_2gram.shape[1])) * 100
    print(f"  • Sparsity:    {sparsity_2gram:.2f}%")

    # Mostrar ejemplos de bigramas
    bigrams = [(term, idx)
               for term, idx in vectorizer_2gram.vocabulary_.items() if ' ' in term]
    bigrams_sorted = sorted(bigrams, key=lambda x: x[1])[:10]
    print(f"\n  Ejemplos de bigramas:")
    print(f"  {', '.join([term for term, _ in bigrams_sorted])}")

    # ---------------------------------------------------------------------
    # 2.3. TRIGRAMAS (n=3)
    # ---------------------------------------------------------------------
    print("\n\n[2.3] Generando vectores TF-IDF con TRIGRAMAS (n=1,2,3)")
    print("-" * 80)

    # Entrenar vectorizador con trigramas
    vectorizer_3gram = vector_manager.fit_tfidf_vectorizer(
        train_docs,
        ngram_range=(1, 3),
        max_features=15000,
        min_df=2,
        max_df=0.95
    )

    print(
        f"\n✓ Vocabulario generado: {len(vectorizer_3gram.vocabulary_)} términos")

    # Transformar documentos
    train_vectors_3gram = vector_manager.transform_tfidf(
        train_docs, ngram_range=(1, 3))
    test_vectors_3gram = vector_manager.transform_tfidf(
        test_docs, ngram_range=(1, 3))

    print(f"\n✓ Vectores generados:")
    print(f"  • Train shape: {train_vectors_3gram.shape}")
    print(f"  • Test shape:  {test_vectors_3gram.shape}")
    sparsity_3gram = (1 - train_vectors_3gram.nnz /
                      (train_vectors_3gram.shape[0] * train_vectors_3gram.shape[1])) * 100
    print(f"  • Sparsity:    {sparsity_3gram:.2f}%")

    # Mostrar ejemplos de trigramas
    trigrams = [(term, idx) for term, idx in vectorizer_3gram.vocabulary_.items(
    ) if term.count(' ') == 2]
    if trigrams:
        trigrams_sorted = sorted(trigrams, key=lambda x: x[1])[:10]
        print(f"\n  Ejemplos de trigramas:")
        print(f"  {', '.join([term for term, _ in trigrams_sorted])}")

    # =========================================================================
    # 3. REPRESENTACIÓN 2: CARACTERÍSTICAS LINGÜÍSTICAS DE OPINIÓN
    # =========================================================================
    print("\n\n" + "="*80)
    print("[3] REPRESENTACIÓN 2: Características lingüísticas de opinión y sentimiento")
    print("="*80)

    # Configurar extractor con vocabulario de dominio
    domain_vocabulary = [
        # Mecánicas
        'dice', 'roll', 'card', 'deck', 'strategy', 'tactics', 'mechanic',
        'worker placement', 'deck building', 'engine building', 'area control',
        # Componentes
        'component', 'miniature', 'token', 'board', 'piece', 'tile', 'meeple',
        'artwork', 'quality', 'production',
        # Reglas
        'rule', 'rulebook', 'setup', 'turn', 'phase', 'round', 'player',
        'gameplay', 'playtime', 'downtime',
        # Experiencia
        'fun', 'replay', 'replayability', 'theme', 'immersion', 'complexity',
        'balance', 'interaction', 'luck', 'skill'
    ]

    feature_extractor = FeatureExtractor(
        opinion_lexicon='vader',
        domain_vocabulary=domain_vocabulary
    )

    print("\nExtrayendo características lingüísticas del conjunto de entrenamiento...")
    train_linguistic_features = feature_extractor.extract_features_batch(
        train_docs,
        show_progress=True,
        n_jobs=12
    )

    print("\nExtrayendo características lingüísticas del conjunto de test...")
    test_linguistic_features = feature_extractor.extract_features_batch(
        test_docs,
        show_progress=True,
        n_jobs=12
    )

    print(f"\n✓ Características extraídas:")
    print(f"  • Train: {len(train_linguistic_features)} documentos")
    print(f"  • Test:  {len(test_linguistic_features)} documentos")

    # Vectorizar características lingüísticas
    print("\nVectorizando características lingüísticas...")
    train_ling_vectors, feature_names = vector_manager.vectorize_linguistic_features(
        train_linguistic_features
    )
    test_ling_vectors, _ = vector_manager.vectorize_linguistic_features(
        test_linguistic_features
    )

    print(f"\n✓ Vectores de características lingüísticas:")
    print(f"  • Train shape: {train_ling_vectors.shape}")
    print(f"  • Test shape:  {test_ling_vectors.shape}")
    print(f"  • Número de características: {len(feature_names)}")
    print(f"\n  Características incluidas:")
    for i, name in enumerate(feature_names):
        if i % 5 == 0 and i > 0:
            print()
        print(f"    - {name}", end="")
        if i < len(feature_names) - 1:
            print(", ", end="")
    print()

    # =========================================================================
    # 4. REPRESENTACIONES COMBINADAS
    # =========================================================================
    print("\n\n" + "="*80)
    print("[4] REPRESENTACIONES COMBINADAS: TF-IDF + Características Lingüísticas")
    print("="*80)

    # Combinar unigramas + características lingüísticas
    print("\n[4.1] Combinación: Unigramas + Características Lingüísticas")
    print("-" * 80)
    train_combined_1gram = vector_manager.combine_features(
        train_vectors_1gram,
        train_ling_vectors
    )
    test_combined_1gram = vector_manager.combine_features(
        test_vectors_1gram,
        test_ling_vectors
    )
    print(f"✓ Train shape: {train_combined_1gram.shape}")
    print(f"✓ Test shape:  {test_combined_1gram.shape}")

    # Combinar bigramas + características lingüísticas
    print("\n[4.2] Combinación: Bigramas + Características Lingüísticas")
    print("-" * 80)
    train_combined_2gram = vector_manager.combine_features(
        train_vectors_2gram,
        train_ling_vectors
    )
    test_combined_2gram = vector_manager.combine_features(
        test_vectors_2gram,
        test_ling_vectors
    )
    print(f"✓ Train shape: {train_combined_2gram.shape}")
    print(f"✓ Test shape:  {test_combined_2gram.shape}")

    # Combinar trigramas + características lingüísticas
    print("\n[4.3] Combinación: Trigramas + Características Lingüísticas")
    print("-" * 80)
    train_combined_3gram = vector_manager.combine_features(
        train_vectors_3gram,
        train_ling_vectors
    )
    test_combined_3gram = vector_manager.combine_features(
        test_vectors_3gram,
        test_ling_vectors
    )
    print(f"✓ Train shape: {train_combined_3gram.shape}")
    print(f"✓ Test shape:  {test_combined_3gram.shape}")

    # =========================================================================
    # 5. GUARDAR REPRESENTACIONES VECTORIALES
    # =========================================================================
    print("\n\n" + "="*80)
    print("[5] Guardando representaciones vectoriales")
    print("="*80)

    # Inicializar persistence manager
    persistence = PersistenceManager('./data')

    print("\nGuardando vectores TF-IDF...")

    # Guardar unigramas
    persistence.save_sparse_matrix(
        train_vectors_1gram, 'train_tfidf_1gram.npz')
    persistence.save_sparse_matrix(test_vectors_1gram, 'test_tfidf_1gram.npz')

    # Guardar bigramas
    persistence.save_sparse_matrix(
        train_vectors_2gram, 'train_tfidf_2gram.npz')
    persistence.save_sparse_matrix(test_vectors_2gram, 'test_tfidf_2gram.npz')

    # Guardar trigramas
    persistence.save_sparse_matrix(
        train_vectors_3gram, 'train_tfidf_3gram.npz')
    persistence.save_sparse_matrix(test_vectors_3gram, 'test_tfidf_3gram.npz')

    print("\nGuardando características lingüísticas...")
    # Las características lingüísticas son matrices densas, convertir a sparse
    train_ling_sparse = csr_matrix(train_ling_vectors)
    test_ling_sparse = csr_matrix(test_ling_vectors)
    persistence.save_sparse_matrix(train_ling_sparse, 'train_linguistic.npz')
    persistence.save_sparse_matrix(test_ling_sparse, 'test_linguistic.npz')

    print("\nGuardando representaciones combinadas...")
    persistence.save_sparse_matrix(
        train_combined_1gram, 'train_combined_1gram.npz')
    persistence.save_sparse_matrix(
        test_combined_1gram, 'test_combined_1gram.npz')

    persistence.save_sparse_matrix(
        train_combined_2gram, 'train_combined_2gram.npz')
    persistence.save_sparse_matrix(
        test_combined_2gram, 'test_combined_2gram.npz')

    persistence.save_sparse_matrix(
        train_combined_3gram, 'train_combined_3gram.npz')
    persistence.save_sparse_matrix(
        test_combined_3gram, 'test_combined_3gram.npz')

    print("\nGuardando vectorizadores...")
    # Guardar vectorizadores directamente con pickle
    with open(persistence.vectors_dir / 'vectorizer_1gram.pkl', 'wb') as f:
        pickle.dump(vectorizer_1gram, f)

    with open(persistence.vectors_dir / 'vectorizer_2gram.pkl', 'wb') as f:
        pickle.dump(vectorizer_2gram, f)

    with open(persistence.vectors_dir / 'vectorizer_3gram.pkl', 'wb') as f:
        pickle.dump(vectorizer_3gram, f)

    print("\nGuardando documentos y características...")
    persistence.save_documents_to_csv(balanced_docs, 'balanced_corpus.csv')

    persistence.save_data_split(
        train_docs, test_docs, None, 'train_test_splits.csv')

    persistence.save_linguistic_features(
        train_linguistic_features + test_linguistic_features,
        'all_linguistic_features.csv'
    )

    # =========================================================================
    # 6. RESUMEN FINAL
    # =========================================================================
    print("\n\n" + "="*80)
    print("RESUMEN DE REPRESENTACIONES VECTORIALES GENERADAS")
    print("="*80)

    print("\n[A] REPRESENTACIONES BASADAS EN N-GRAMAS Y TF-IDF:")
    print("-" * 80)
    print(f"""
  1. UNIGRAMAS (n=1):
     • Dimensión: {train_vectors_1gram.shape[1]} características
     • Sparsity: {sparsity_1gram:.2f}%
     
  2. BIGRAMAS (n=1,2):
     • Dimensión: {train_vectors_2gram.shape[1]} características
     • Sparsity: {sparsity_2gram:.2f}%
     
  3. TRIGRAMAS (n=1,2,3):
     • Dimensión: {train_vectors_3gram.shape[1]} características
     • Sparsity: {sparsity_3gram:.2f}%
    """)

    print("\n[B] REPRESENTACIONES BASADAS EN OPINIÓN Y SENTIMIENTO:")
    print("-" * 80)
    print(f"""
  Características lingüísticas de alto nivel:
     • Dimensión: {train_ling_vectors.shape[1]} características
    """)

    print("\n[C] REPRESENTACIONES COMBINADAS:")
    print("-" * 80)
    print(f"""
  1. Unigramas + Lingüísticas:    {train_combined_1gram.shape[1]} características
  2. Bigramas + Lingüísticas:     {train_combined_2gram.shape[1]} características
  3. Trigramas + Lingüísticas:    {train_combined_3gram.shape[1]} características
    """)


if __name__ == "__main__":
    main()
