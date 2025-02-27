import pandas as pd
import base64
from io import BytesIO
from PIL import Image
from datasets import load_dataset
from mcdse import MCDSEModel

# --------------------------------------------------------
# 1. Charger TOUT le corpus
# --------------------------------------------------------
corpus_df = pd.read_parquet("/Users/vuong/Desktop/geotechnie/parquet")

def base64_to_image(b64_str):
    image_data = base64.b64decode(b64_str)
    return Image.open(BytesIO(image_data))

# --------------------------------------------------------
# 2. Charger le benchmark une seule fois
# --------------------------------------------------------
print("Chargement du benchmark...")
benchmark_ds = load_dataset("paulml/testimages3-images", split="train")
benchmark_images = [img.convert("RGB") for img in benchmark_ds["image"]]

# --------------------------------------------------------
# 3. Initialiser le modèle
# --------------------------------------------------------
model = MCDSEModel(
    use_fake=True,
    dimension=768,
    batch_size=32,  # Augmenté pour meilleure performance
    device="cpu"
)

# Générer les embeddings du benchmark une seule fois
print("Encodage du benchmark...")
benchmark_embeddings = model.fake_encode_documents(benchmark_images)

# --------------------------------------------------------
# 4. Traiter le corpus par lots
# --------------------------------------------------------
BATCH_SIZE = 100  # Ajuster selon la mémoire disponible
results = []

for batch_start in range(0, len(corpus_df), BATCH_SIZE):
    batch_end = batch_start + BATCH_SIZE
    batch_df = corpus_df.iloc[batch_start:batch_end]
    
    print(f"\nTraitement des images {batch_start}-{batch_end}...")
    
    # Conversion et encodage par lot
    batch_images = [base64_to_image(img) for img in batch_df["image"]]
    batch_embeddings = model.fake_encode_documents(batch_images)
    
    # Calcul de similarité avec le benchmark
    batch_similarities = model.compute_similarity(batch_embeddings, benchmark_embeddings)
    
    # Stockage des résultats
    for i, (idx, row) in enumerate(batch_df.iterrows()):
        best_match_idx = batch_similarities[i].argmax().item()
        best_score = batch_similarities[i][best_match_idx].item()
        
        results.append({
            "docid": row["docid"],
            "best_match": benchmark_ds["image_id"][best_match_idx],
            "score": best_score
        })

# --------------------------------------------------------
# 5. Sauvegarder les résultats
# --------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("/Users/vuong/Desktop/geotechnie", index=False)

print("\nTraitement terminé !")
print(f"Résultats sauvegardés dans similarity_results.csv ({len(results_df)} lignes)")