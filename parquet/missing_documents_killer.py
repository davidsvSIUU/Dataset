import pandas as pd
import sys
import base64
from pathlib import Path

# Ajoutez le chemin vers le dossier contenant pdf_utils.py
sys.path.append('/Users/vuong/Desktop/vision-benchmark-maker/src')  
from pdf_utils import capture_page_image_hd

def add_missing_documents_to_corpus(train_path, corpus_path, pdf_folder):
    try:
        # Charger les fichiers Parquet
        print("Chargement des fichiers parquet...")
        train_df = pd.read_parquet(train_path)
        corpus_df = pd.read_parquet(corpus_path)
        
        # Identifier les documents manquants dans le corpus
        train_docs = set(train_df['pos'].unique())
        corpus_docs = set(corpus_df['docid'])
        missing_docs = train_docs - corpus_docs
        
        print(f"Nombre de documents manquants à ajouter: {len(missing_docs)}")
        print("\nListe des documents manquants:")
        for doc in missing_docs:
            print(f"- {doc}")
        
        if missing_docs:
            new_entries = []
            
            for doc_id in missing_docs:
                print(f"\nTraitement de: {doc_id}")
                parts = doc_id.rsplit('_', 1)
                if len(parts) != 2:
                    print(f"Warning: Format de position invalide: {doc_id}")
                    continue
                    
                pdf_name, page_num = parts
                try:
                    page_num = int(page_num)
                    pdf_path = Path(pdf_folder) / pdf_name
                    
                    print(f"Recherche du PDF: {pdf_path}")
                    if not pdf_path.exists():
                        print(f"Warning: PDF non trouvé: {pdf_path}")
                        continue
                    
                    print(f"Capture de l'image pour {pdf_name} page {page_num}")
                    image_bytes = capture_page_image_hd(str(pdf_path), page_num)
                    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                    
                    new_entry = {
                        'docid': doc_id,
                        'image': image_b64
                    }
                    new_entries.append(new_entry)
                    print(f"Entrée créée avec succès pour {doc_id}")
                    
                except ValueError:
                    print(f"Warning: Numéro de page invalide dans la position: {doc_id}")
                    continue
                except Exception as e:
                    print(f"Erreur lors du traitement de {pdf_name} page {page_num}: {str(e)}")
                    continue
            
            if new_entries:
                print(f"\nCréation de {len(new_entries)} nouvelles entrées...")
                new_df = pd.DataFrame(new_entries)
                updated_corpus = pd.concat([corpus_df, new_df], ignore_index=True)
                
                print("Sauvegarde du fichier corpus.parquet mis à jour...")
                updated_corpus.to_parquet(corpus_path)
                
                print("Mise à jour terminée avec succès!")
                print(f"Documents ajoutés: {len(new_entries)}")
            else:
                print("\nAucune nouvelle entrée n'a pu être créée.")
                print("Vérifiez que les PDFs existent dans le dossier:")
                print(pdf_folder)
            
        else:
            print("Aucun document manquant à ajouter.")
            
    except Exception as e:
        print(f"Erreur lors de la mise à jour: {str(e)}")

# Chemins vers vos fichiers
train_path = "/Users/vuong/Desktop/vision-benchmark-maker/parquet/train.parquet"
corpus_path = "/Users/vuong/Desktop/vision-benchmark-maker/parquet/corpus.parquet"
pdf_folder = "/Users/vuong/Desktop/vision-benchmark-maker/input_pdfs"

add_missing_documents_to_corpus(train_path, corpus_path, pdf_folder)