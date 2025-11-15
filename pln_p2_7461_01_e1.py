from corpus import (
    CorpusReader,
    Corpus,
    PreprocessingPipeline,
    LinguisticAnalyzer,
    FeatureExtractor
)
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    print("="*80)
    print("EJERCICIO 1: Extracción de características lingüísticas de reseñas BGG")
    print("="*80)

    # =========================================================================
    # 1. CARGAR CORPUS (limitado a 10k documentos)
    # =========================================================================
    print("\n[1] Cargando corpus...")
    print("-" * 80)

    # Inicializar lector
    reader = CorpusReader('./data/raw_data')

    # Crear pipeline de preprocesamiento
    preprocessing = PreprocessingPipeline([
        'remove_html',
        'remove_urls',
        'lowercase',
        'remove_extra_whitespace'
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

    # print("Cargando hasta 10,000 documentos...")
    # corpus.load(reader, max_documents=10000, show_progress=True)
    # print("Cargando todos los documentos...")
    # corpus.load(reader, show_progress=True)
    print("Cargando hasta 1M documentos...")
    corpus.load(reader, max_documents=1000000, show_progress=True)

    print(f"\n✓ Corpus cargado: {len(corpus)} documentos")

    # Asignar etiquetas de sentimiento
    corpus.assign_labels()

    # Mostrar distribución de etiquetas
    print("\nDistribución de etiquetas:")
    for label in ['positive', 'negative', 'neutral']:
        docs = corpus.get_documents_by_label(label)
        print(f"  • {label:10s}: {len(docs):5d} documentos")

    # =========================================================================
    # 2. INICIALIZAR EXTRACTOR DE CARACTERÍSTICAS
    # =========================================================================
    print("\n[2] Inicializando extractor de características...")
    print("-" * 80)

    # Configurar vocabulario de dominio (términos específicos de juegos de mesa)
    domain_vocabulary = [
        # Mecánicas de juego
        'dice', 'roll', 'card', 'deck', 'strategy', 'tactics', 'mechanic',
        'worker placement', 'deck building', 'engine building', 'area control',

        # Componentes
        'component', 'miniature', 'token', 'board', 'piece', 'tile', 'meeple',
        'artwork', 'quality', 'production',

        # Reglas y jugabilidad
        'rule', 'rulebook', 'setup', 'turn', 'phase', 'round', 'player',
        'gameplay', 'playtime', 'downtime',

        # Experiencia
        'fun', 'replay', 'replayability', 'theme', 'immersion', 'complexity',
        'balance', 'interaction', 'luck', 'skill'
    ]

    # Crear extractor con lexicon VADER y vocabulario de dominio
    feature_extractor = FeatureExtractor(
        opinion_lexicon='vader',
        domain_vocabulary=domain_vocabulary
    )

    print("✓ Extractor configurado con:")
    print(f"  • Lexicon de opinión: VADER (Valence Aware Dictionary)")
    print(
        f"  • Vocabulario de dominio: {len(domain_vocabulary)} términos de juegos de mesa")

    # =========================================================================
    # 3. EXTRAER CARACTERÍSTICAS DE UN DOCUMENTO DE EJEMPLO
    # =========================================================================
    print("\n[3] Ejemplo: Características de un documento individual")
    print("-" * 80)

    # Tomar un documento de ejemplo
    all_docs = list(corpus.documents.values())
    sample_doc = all_docs[min(100, len(all_docs)-1)]  # Documento arbitrario

    print(f"\nDocumento ID: {sample_doc.doc_id}")
    print(f"Rating: {sample_doc.rating}")
    print(f"Label: {sample_doc.label}")
    print(f"\nTexto original (primeros 300 caracteres):")
    print(f"{sample_doc.raw_text[:300]}...")

    # Extraer características del documento
    features = feature_extractor.generate_feature_dict(sample_doc)

    print("\n" + "="*80)
    print("CARACTERÍSTICAS EXTRAÍDAS:")
    print("="*80)

    # Característica 1: Palabras de opinión/sentimiento
    print("\n[A] PALABRAS DE OPINIÓN/SENTIMIENTO (VADER):")
    print("-" * 80)
    print(f"  • Palabras positivas:     {features['positive_word_count']}")
    print(f"  • Palabras negativas:     {features['negative_word_count']}")
    print(
        f"  • Ratio opinión (pos-neg/total): {features['opinion_word_ratio']:.3f}")
    print(f"  • VADER compound score:   {features['vader_compound']:.3f}")
    print(f"  • VADER positive score:   {features['vader_positive']:.3f}")
    print(f"  • VADER negative score:   {features['vader_negative']:.3f}")
    print(f"  • VADER neutral score:    {features['vader_neutral']:.3f}")

    # Característica 2: Negaciones
    print("\n[B] NEGACIONES:")
    print("-" * 80)
    print(f"  • Recuento de negaciones: {features['negation_count']}")
    negation_ratio = features['negation_count'] / \
        max(features['token_count'], 1)
    print(f"  • Ratio negaciones/tokens: {negation_ratio:.3f}")
    print("  Ejemplos de palabras de negación: not, no, never, neither, hardly, barely")

    # Característica 3: Intensificadores/modificadores
    print("\n[C] INTENSIFICADORES Y MODIFICADORES:")
    print("-" * 80)
    print(f"  • Recuento intensificadores: {features['intensifier_count']}")
    print(f"  • Recuento atenuadores:      {features['mitigator_count']}")
    intensifier_ratio = features['intensifier_count'] / \
        max(features['token_count'], 1)
    print(f"  • Ratio intensif./tokens:    {intensifier_ratio:.3f}")
    print("  Ejemplos de intensificadores: very, extremely, really, totally, absolutely")
    print("  Ejemplos de atenuadores: slightly, somewhat, barely, hardly, sort of")

    # Característica 4: Vocabulario de dominio
    print("\n[D] VOCABULARIO DE DOMINIO (Juegos de Mesa):")
    print("-" * 80)
    print(f"  • Palabras de dominio:    {features['domain_word_count']}")
    print(f"  • Ratio dominio/tokens:   {features['domain_word_ratio']:.3f}")
    print(f"  Términos configurados: mecánicas, componentes, reglas, experiencia")

    # Características adicionales
    print("\n[E] CARACTERÍSTICAS ADICIONALES:")
    print("-" * 80)
    print(f"  • Longitud del texto:     {features['text_length']} caracteres")
    print(f"  • Número de tokens:       {features['token_count']}")
    print(f"  • Número de oraciones:    {features['sentence_count']}")
    print(
        f"  • Promedio palabras/oración: {features['avg_sentence_length']:.1f}")

    # =========================================================================
    # 4. EXTRAER CARACTERÍSTICAS DE TODOS LOS DOCUMENTOS
    # =========================================================================
    print("\n\n[4] Extrayendo características de todos los documentos...")
    print("-" * 80)

    # Extraer características en lote (con barra de progreso)
    all_features = feature_extractor.extract_features_batch(
        list(corpus.documents.values()),
        show_progress=True,
        n_jobs=12
    )

    print(f"\n✓ Características extraídas de {len(all_features)} documentos")

    # =========================================================================
    # 5. ESTADÍSTICAS AGREGADAS
    # =========================================================================
    print("\n[5] Estadísticas agregadas del corpus")
    print("-" * 80)

    # Calcular promedios
    avg_positive_words = sum(f['positive_word_count']
                             for f in all_features) / len(all_features)
    avg_negative_words = sum(f['negative_word_count']
                             for f in all_features) / len(all_features)
    avg_negations = sum(f['negation_count']
                        for f in all_features) / len(all_features)
    avg_intensifiers = sum(f['intensifier_count']
                           for f in all_features) / len(all_features)
    avg_domain_words = sum(f['domain_word_count']
                           for f in all_features) / len(all_features)
    avg_vader_compound = sum(f['vader_compound']
                             for f in all_features) / len(all_features)

    print("\nPROMEDIOS POR DOCUMENTO:")
    print(f"  • Palabras positivas:        {avg_positive_words:.2f}")
    print(f"  • Palabras negativas:        {avg_negative_words:.2f}")
    print(f"  • Negaciones:                {avg_negations:.2f}")
    print(f"  • Intensificadores:          {avg_intensifiers:.2f}")
    print(f"  • Palabras de dominio:       {avg_domain_words:.2f}")
    print(f"  • VADER compound score:      {avg_vader_compound:.3f}")

    # Estadísticas por etiqueta
    print("\n\nPROMEDIOS POR ETIQUETA DE SENTIMIENTO:")
    print("-" * 80)

    for label in ['positive', 'negative', 'neutral']:
        label_docs = corpus.get_documents_by_label(label)
        label_ids = {doc.doc_id for doc in label_docs}
        label_features = [f for f in all_features if f['doc_id'] in label_ids]

        if label_features:
            print(f"\n{label.upper()}:")
            avg_pos = sum(f['positive_word_count']
                          for f in label_features) / len(label_features)
            avg_neg = sum(f['negative_word_count']
                          for f in label_features) / len(label_features)
            avg_vader = sum(f['vader_compound']
                            for f in label_features) / len(label_features)
            avg_negation = sum(f['negation_count']
                               for f in label_features) / len(label_features)

            print(f"  • Palabras positivas:     {avg_pos:.2f}")
            print(f"  • Palabras negativas:     {avg_neg:.2f}")
            print(f"  • VADER compound:         {avg_vader:.3f}")
            print(f"  • Negaciones:             {avg_negation:.2f}")


if __name__ == "__main__":
    main()
