import pandas as pd
from pathlib import Path

def filter_parquets(similarity_csv: str, corpus_path: str, train_path: str, output_dir: str):
    # Lire le CSV de similarité
    similarity_df = pd.read_csv(similarity_csv)
    
    # Créer un set de tous les docids à supprimer (docid + best_match)
    docids_to_remove = set(similarity_df['docid']).union(set(similarity_df['best_match']))
    
    # Charger les données existantes
    corpus_df = pd.read_parquet(corpus_path)
    train_df = pd.read_parquet(train_path)
    
    # Filtrer le corpus
    filtered_corpus = corpus_df[~corpus_df['docid'].isin(docids_to_remove)]
    
    # Filtrer le train (on garde seulement les queries dont la pos n'est pas dans la liste)
    filtered_train = train_df[~train_df['pos'].isin(docids_to_remove)]
    
    # Sauvegarder les nouveaux fichiers
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filtered_corpus.to_parquet(output_dir / 'corpus_filtered.parquet', index=False)
    filtered_train.to_parquet(output_dir / 'train_filtered.parquet', index=False)
    
    # Afficher les stats
    print("Filtrage terminé avec succès!")
    print(f"Documents supprimés du corpus: {len(corpus_df) - len(filtered_corpus)}")
    print(f"Requêtes supprimées du train: {len(train_df) - len(filtered_train)}")
    print(f"Nouveaux fichiers sauvegardés dans: {output_dir}")

if __name__ == "__main__":
    # Configurer les chemins
    SIMILARITY_CSV = "/Users/vuong/Desktop/vision-benchmark-maker/vect/filtres_similarity_results.csv"
    CORPUS_PATH = "/Users/vuong/Desktop/vision-benchmark-maker/parquet_files/corpus.parquet"
    TRAIN_PATH = "/Users/vuong/Desktop/vision-benchmark-maker/parquet_files/train.parquet"
    OUTPUT_DIR = "/Users/vuong/Desktop/vision-benchmark-maker/filtered_parquets"
    
    # Exécuter le filtrage
    filter_parquets(SIMILARITY_CSV, CORPUS_PATH, TRAIN_PATH, OUTPUT_DIR)