from corpus import (
    CorpusReader,
    Corpus,
    PreprocessingPipeline,
    LinguisticAnalyzer,
    PersistenceManager
)
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    print("="*80)
    print("EJERCICIO 3: Creación de ficheros de entrenamiento, validación y test")
    print("="*80)

    # =========================================================================
    # 1. ETIQUETADO DEL CORPUS
    # =========================================================================
    print("\n[1] ETIQUETADO DEL CORPUS")
    print("="*80)

    # Definir umbrales de conversión de ratings a etiquetas de polaridad
    print("\nUmbrales de etiquetado definidos:")
    print("-" * 80)
    label_map = {
        'positive': [7, 10],      # Ratings 7-10 → POSITIVA
        'neutral': [5, 6.99],     # Ratings 5-6.99 → NEUTRA
        'negative': [0, 4.99]     # Ratings 0-4.99 → NEGATIVA
    }

    for label, (min_rating, max_rating) in label_map.items():
        print(
            f"  • {label.upper():10s}: ratings en rango [{min_rating}, {max_rating}]")

    # Cargar corpus
    print("\n\nCargando corpus...")
    print("-" * 80)

    reader = CorpusReader('./data/raw_data')

    # Crear pipeline de preprocesamiento
    preprocessing = PreprocessingPipeline([
        'remove_html',
        'remove_urls',
        'lowercase',
        'remove_extra_whitespace',
        'remove_stopwords',
        'lemmatize'
    ])

    # Crear analizador lingüístico
    analyzer = LinguisticAnalyzer()

    # Crear corpus con label_map
    corpus = Corpus(
        label_map=label_map,
        preprocessing_pipeline=preprocessing,
        linguistic_analyzer=analyzer
    )

    # Cargar documentos (10k para el ejercicio)
    # print("Cargando hasta 10,000 documentos...")
    # corpus.load(reader, max_documents=10000, show_progress=True)
    # print("Cargando todos los documentos...")
    # corpus.load(reader, show_progress=True)
    print("Cargando hasta 1M documentos...")
    corpus.load(reader, max_documents=1000000, show_progress=True)

    print(f"\n✓ Corpus cargado: {len(corpus)} documentos")

    # Asignar etiquetas basándose en ratings
    print("\nAsignando etiquetas de polaridad basándose en ratings...")
    corpus.assign_labels()

    # Mostrar distribución antes de balancear
    print("\n✓ Etiquetas asignadas. Distribución:")
    print("-" * 80)
    stats = corpus.get_statistics()
    total = stats['labeled_documents']

    for label, count in stats['label_distribution'].items():
        percentage = (count / total) * 100
        print(
            f"  • {label.upper():10s}: {count:5d} documentos ({percentage:5.2f}%)")

    print(f"\n  TOTAL:        {total:5d} documentos")

    # =========================================================================
    # 2. PARTICIONADO DEL CONJUNTO DE DATOS
    # =========================================================================
    print("\n\n[2] PARTICIONADO DEL CONJUNTO DE DATOS")
    print("="*80)

    # Paso 1: Balancear dataset
    print("\n\nPaso 1: Balanceando dataset...")
    print("-" * 80)
    balanced_docs = corpus.balance_dataset(
        strategy='subsample', random_seed=42)

    print(f"✓ Dataset balanceado: {len(balanced_docs)} documentos")

    # Verificar balance
    balanced_stats = {}
    for label in ['positive', 'negative', 'neutral']:
        count = sum(1 for doc in balanced_docs if doc.label == label)
        balanced_stats[label] = count
        print(
            f"  • {label.upper():10s}: {count:5d} documentos ({count/len(balanced_docs)*100:.2f}%)")

    # Paso 2: Split Train/Test (80/20)
    print("\n\nPaso 2: Creando split Train/Test (80%/20%)...")
    print("-" * 80)
    train_val_docs, test_docs = corpus.create_stratified_split(
        test_size=0.2,
        random_seed=42,
        documents=balanced_docs
    )

    print(f"✓ Train+Val: {len(train_val_docs)} documentos (80%)")
    print(f"✓ Test:      {len(test_docs)} documentos (20%)")

    # Verificar estratificación en test
    print("\n  Distribución en conjunto TEST:")
    for label in ['positive', 'negative', 'neutral']:
        count = sum(1 for doc in test_docs if doc.label == label)
        print(
            f"    • {label.upper():10s}: {count:4d} ({count/len(test_docs)*100:.2f}%)")

    # Paso 3: Split Train/Validation (del 80% anterior → 64% train, 16% val del total)
    print("\n\nPaso 3: Creando split Train/Validation (80%/20% del Train+Val)...")
    print("-" * 80)

    # Crear un corpus temporal para hacer el split
    temp_corpus = Corpus(
        label_map=label_map,
        preprocessing_pipeline=preprocessing,
        linguistic_analyzer=analyzer
    )

    train_docs, val_docs = temp_corpus.create_stratified_split(
        test_size=0.2,
        random_seed=42,
        documents=train_val_docs
    )

    print(f"✓ Train:      {len(train_docs)} documentos (64% del total)")
    print(f"✓ Validation: {len(val_docs)} documentos (16% del total)")
    print(f"✓ Test:       {len(test_docs)} documentos (20% del total)")

    # Verificar estratificación en train y validation
    print("\n  Distribución en conjunto TRAIN:")
    for label in ['positive', 'negative', 'neutral']:
        count = sum(1 for doc in train_docs if doc.label == label)
        print(
            f"    • {label.upper():10s}: {count:4d} ({count/len(train_docs)*100:.2f}%)")

    print("\n  Distribución en conjunto VALIDATION:")
    for label in ['positive', 'negative', 'neutral']:
        count = sum(1 for doc in val_docs if doc.label == label)
        print(
            f"    • {label.upper():10s}: {count:4d} ({count/len(val_docs)*100:.2f}%)")

    # Resumen de splits
    print("\n\nRESUMEN DE PARTICIONES:")
    print("-" * 80)
    total_docs = len(train_docs) + len(val_docs) + len(test_docs)
    print(
        f"  • Train:      {len(train_docs):4d} documentos ({len(train_docs)/total_docs*100:.1f}%)")
    print(
        f"  • Validation: {len(val_docs):4d} documentos ({len(val_docs)/total_docs*100:.1f}%)")
    print(
        f"  • Test:       {len(test_docs):4d} documentos ({len(test_docs)/total_docs*100:.1f}%)")
    print(f"  • TOTAL:      {total_docs:4d} documentos")

    # =========================================================================
    # 3. ALMACENADO DE DATOS
    # =========================================================================
    print("\n\n[3] ALMACENADO DE DATOS")
    print("="*80)

    # Inicializar persistence manager
    persistence = PersistenceManager('./data')

    # Guardar corpus completo balanceado
    print("\n\nGuardando corpus completo balanceado...")
    print("-" * 80)
    persistence.save_documents_to_csv(balanced_docs, 'corpus_balanced.csv')
    print(f"✓ Guardado: corpus_balanced.csv ({len(balanced_docs)} documentos)")

    # Guardar conjunto de ENTRENAMIENTO
    print("\nGuardando conjunto de ENTRENAMIENTO...")
    print("-" * 80)
    persistence.save_documents_to_csv(train_docs, 'train_set.csv')
    print(f"✓ Guardado: train_set.csv ({len(train_docs)} documentos)")

    # Crear archivo con información adicional
    train_info = {
        'doc_id': [],
        'label': [],
        'rating': [],
        'text_length': [],
        'game_id': [],
        'user': []
    }
    for doc in train_docs:
        train_info['doc_id'].append(doc.doc_id)
        train_info['label'].append(doc.label)
        train_info['rating'].append(doc.rating)
        train_info['text_length'].append(len(doc.raw_text))
        train_info['game_id'].append(doc.game_id)
        train_info['user'].append(doc.user)

    # Guardar conjunto de VALIDACIÓN
    print("\nGuardando conjunto de VALIDACIÓN...")
    print("-" * 80)
    persistence.save_documents_to_csv(val_docs, 'validation_set.csv')
    print(f"✓ Guardado: validation_set.csv ({len(val_docs)} documentos)")

    # Guardar conjunto de TEST
    print("\nGuardando conjunto de TEST...")
    print("-" * 80)
    persistence.save_documents_to_csv(test_docs, 'test_set.csv')
    print(f"✓ Guardado: test_set.csv ({len(test_docs)} documentos)")

    # Guardar información de los splits
    print("\nGuardando información de splits...")
    print("-" * 80)
    persistence.save_data_split(
        train_docs, test_docs, val_docs, 'data_splits.csv')

    # Guardar estadísticas
    print("\nGenerando archivo de estadísticas...")
    print("-" * 80)

    stats_content = f"""# Estadísticas del Corpus - Ejercicio 3
# Generado automáticamente

## 1. ETIQUETADO

Umbrales de conversión rating → etiqueta:
- POSITIVA: ratings [7.0, 10.0]
- NEUTRAL:  ratings [5.0, 6.99]
- NEGATIVA: ratings [0.0, 4.99]

Corpus original (antes de balancear):
- Total documentos: {stats['labeled_documents']}
- Positivas: {stats['label_distribution'].get('positive', 0)} ({stats['label_distribution'].get('positive', 0)/stats['labeled_documents']*100:.2f}%)
- Neutral:   {stats['label_distribution'].get('neutral', 0)} ({stats['label_distribution'].get('neutral', 0)/stats['labeled_documents']*100:.2f}%)
- Negativas: {stats['label_distribution'].get('negative', 0)} ({stats['label_distribution'].get('negative', 0)/stats['labeled_documents']*100:.2f}%)

## 2. BALANCEO

Estrategia: Subsample (reducir clases mayoritarias)
Corpus balanceado:
- Total documentos: {len(balanced_docs)}
- Positivas: {balanced_stats['positive']} ({balanced_stats['positive']/len(balanced_docs)*100:.2f}%)
- Neutral:   {balanced_stats['neutral']} ({balanced_stats['neutral']/len(balanced_docs)*100:.2f}%)
- Negativas: {balanced_stats['negative']} ({balanced_stats['negative']/len(balanced_docs)*100:.2f}%)

## 3. PARTICIONADO

Split estratificado (mantiene proporciones de clases):

TRAIN: {len(train_docs)} documentos ({len(train_docs)/total_docs*100:.1f}%)
"""

    for label in ['positive', 'negative', 'neutral']:
        count = sum(1 for doc in train_docs if doc.label == label)
        stats_content += f"  - {label}: {count} ({count/len(train_docs)*100:.2f}%)\n"

    stats_content += f"\nVALIDATION: {len(val_docs)} documentos ({len(val_docs)/total_docs*100:.1f}%)\n"
    for label in ['positive', 'negative', 'neutral']:
        count = sum(1 for doc in val_docs if doc.label == label)
        stats_content += f"  - {label}: {count} ({count/len(val_docs)*100:.2f}%)\n"

    stats_content += f"\nTEST: {len(test_docs)} documentos ({len(test_docs)/total_docs*100:.1f}%)\n"
    for label in ['positive', 'negative', 'neutral']:
        count = sum(1 for doc in test_docs if doc.label == label)
        stats_content += f"  - {label}: {count} ({count/len(test_docs)*100:.2f}%)\n"

    stats_content += f"""
## 4. ARCHIVOS GENERADOS

Directorio: data/processed_data/

Conjuntos de documentos (CSV):
- corpus_balanced.csv:  Corpus completo balanceado ({len(balanced_docs)} docs)
- train_set.csv:        Conjunto de entrenamiento ({len(train_docs)} docs)
- validation_set.csv:   Conjunto de validación ({len(val_docs)} docs)
- test_set.csv:         Conjunto de test ({len(test_docs)} docs)
- data_splits.csv:      Mapeo doc_id → conjunto (train/validation/test)

Estos archivos están listos para ser utilizados en el Ejercicio 4 (clasificación).
"""

    stats_file = persistence.processed_dir / 'corpus_statistics.txt'
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(stats_content)

    print(f"✓ Guardado: corpus_statistics.txt")

    # =========================================================================
    # 4. RESUMEN FINAL
    # =========================================================================
    print("\n\n" + "="*80)
    print("RESUMEN - EJERCICIO 3 COMPLETADO")
    print("="*80)

    print("\n[✓] 1. ETIQUETADO DEL CORPUS")
    print("-" * 80)
    print(
        f"  • Umbrales definidos: Positiva [7-10], Neutral [5-6.99], Negativa [0-4.99]")
    print(
        f"  • {stats['labeled_documents']} documentos etiquetados automáticamente")
    print(f"  • Corpus original desbalanceado → aplicado balanceo por subsample")

    print("\n[✓] 2. PARTICIONADO DEL CONJUNTO DE DATOS")
    print("-" * 80)
    print(f"  • Dataset balanceado: {len(balanced_docs)} documentos")
    print(f"  • Split estratificado (mantiene proporción de clases):")
    print(
        f"      - TRAIN:      {len(train_docs):4d} docs ({len(train_docs)/total_docs*100:.1f}%)")
    print(
        f"      - VALIDATION: {len(val_docs):4d} docs ({len(val_docs)/total_docs*100:.1f}%)")
    print(
        f"      - TEST:       {len(test_docs):4d} docs ({len(test_docs)/total_docs*100:.1f}%)")
    print(f"  • Uso de validation set: Ajuste de hiperparámetros sin tocar test")


if __name__ == "__main__":
    main()
