import os
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv(override=True)

# Configuration des chemins de base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dossiers
PDF_FOLDER = os.path.join(BASE_DIR, "input_pdfs")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# Fichiers de sortie
OUTPUT_FILE = os.path.join(RESULTS_DIR, "technical_queries_results_all_folders.jsonl")
RETRIEVAL_RESULTS_FILE = os.path.join(RESULTS_DIR, "retrieval_results_fixed.json")
RANKED_RESULTS_FILE = os.path.join(RESULTS_DIR, "ranked_results.json")

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Vector API Hosts
VECTAPI_HOST_IMAGE = "https://lmspaul--mcdse-embeddings-image-embeddings.modal.run"
VECTAPI_HOST_TEXT = "https://lmspaul--mcdse-embeddings-text-embeddings.modal.run"

# Rate Limiter Settings
REQUESTS_PER_SECOND = 10

# Créer les dossiers nécessaires
for directory in [OUTPUT_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)