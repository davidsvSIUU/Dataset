import os
from dotenv import load_dotenv

load_dotenv(override=True)

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Paths
PDF_FOLDER = "Benchmark"
OUTPUT_FILE = "technical_queries_results_all_folders.jsonl"
RETRIEVAL_RESULTS_FILE = "retrieval_results_fixed.json"
RANKED_RESULTS_FILE = "ranked_results.json"

# Vector API Hosts
VECTAPI_HOST_IMAGE = "https://lmspaul--mcdse-embeddings-image-embeddings.modal.run"
VECTAPI_HOST_TEXT = "https://lmspaul--mcdse-embeddings-text-embeddings.modal.run"

# Rate Limiter Settings
REQUESTS_PER_SECOND = 10