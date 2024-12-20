import json
import random
import requests
import io
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from config import VECTAPI_HOST_IMAGE, VECTAPI_HOST_TEXT
from pdf_utils import capture_page_image

def get_recall_position(similarities: np.ndarray, correct_idx: int) -> int:
    sorted_indices = np.argsort(similarities)[::-1]
    return np.where(sorted_indices == correct_idx)[0][0]

def load_random_jsonl_entries(jsonl_path: str, n_samples: int = 500) -> List[Dict]:
    all_entries = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get('error') is None:
                        all_entries.append(entry)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error loading JSONL file: {str(e)}")
        return []
    return random.sample(all_entries, min(n_samples, len(all_entries)))

def embed_image(image_bytes: bytes, filename: str) -> Optional[np.ndarray]:
    try:
        file_obj = io.BytesIO(image_bytes)
        file_obj.seek(0)
        files = [('files', (filename, file_obj, 'image/png'))]
        response = requests.post(VECTAPI_HOST_IMAGE, files=files)
        response.raise_for_status()
        result = response.json()
        if not result.get('results'):
            print(f"Empty results in image embedding response for {filename}")
            return None
        embeddings = result['results'][0].get('embeddings')
        if not embeddings:
            print(f"No embeddings found in response for {filename}")
            return None
        return np.array(embeddings)
    except Exception as e:
        print(f"Error embedding image {filename}: {str(e)}")
        return None

def embed_text(text: str) -> Optional[np.ndarray]:
    try:
        payload = {"texts": [text]}
        response = requests.post(VECTAPI_HOST_TEXT, json=payload)
        response.raise_for_status()
        result = response.json()
        if not result.get('embeddings'):
            print("No embeddings found in text response")
            return None
        return np.array(result['embeddings'][0])
    except Exception as e:
        print(f"Error embedding text: {str(e)}")
        return None

def calculate_ndcg(relevance_scores: np.ndarray, similarities: np.ndarray, correct_idx: int, k: int = 5) -> float:
    if len(similarities) == 0:
        return 0.0
    top_k_indices = np.argsort(similarities)[::-1][:k]
    dcg = 0
    for i, idx in enumerate(top_k_indices):
        rel = 1 if idx == correct_idx else 0
        dcg += (2**rel - 1) / np.log2(i + 2)
    idcg = 1
    return dcg / idcg if idcg > 0 else 0.0

def calculate_recall_at_1(similarities: np.ndarray, correct_idx: int) -> float:
    if len(similarities) == 0:
        return 0.0
    predicted_idx = np.argmax(similarities)
    return 1.0 if predicted_idx == correct_idx else 0.0

def process_all_images(entries: List[Dict], pdf_folder: str) -> List[Optional[np.ndarray]]:
    image_embeddings = []
    for entry in entries:
        pdf_path = Path(pdf_folder) / entry['pdf_name']
        if not pdf_path.exists():
            print(f"PDF file not found: {pdf_path}")
            image_embeddings.append(None)
            continue
        image_bytes = capture_page_image(str(pdf_path), entry['page_number'])
        if image_bytes is None:
            print(f"Failed to capture page image for {pdf_path}")
            image_embeddings.append(None)
            continue
        image_embedding = embed_image(image_bytes, f"{entry['pdf_name']}_p{entry['page_number']}.png")
        image_embeddings.append(image_embedding)
    return image_embeddings

def process_single_query(
    query: str,
    image_embeddings: List[np.ndarray],
    query_idx: int,
    entries: List[Dict]
) -> Optional[Dict]:
    try:
        text_embedding = embed_text(query)
        if text_embedding is None:
            return None
        text_embedding = normalize(text_embedding.reshape(1, -1))
        similarities = []
        valid_indices = []
        for i, img_emb in enumerate(image_embeddings):
            if img_emb is not None:
                img_emb = normalize(img_emb.reshape(1, -1))
                similarity = cosine_similarity(text_embedding, img_emb)[0][0]
                similarities.append(similarity)
                valid_indices.append(i)
        if not similarities:
            return None
        similarities = np.array(similarities)
        recall_position = get_recall_position(similarities, valid_indices.index(query_idx))
        ndcg = calculate_ndcg(
            relevance_scores=None,
            similarities=similarities,
            correct_idx=valid_indices.index(query_idx),
            k=5
        )
        top_k = 15
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_matches = []
        for idx in top_indices:
            original_idx = valid_indices[idx]
            entry = entries[original_idx]
            top_matches.append({
                'pdf_name': entry['pdf_name'],
                'page_number': entry['page_number'],
                'similarity_score': float(similarities[idx])
            })
        return {
            'query': query,
            'pdf_name': entries[query_idx]['pdf_name'],
            'page_number': entries[query_idx]['page_number'],
            'recall_position': int(recall_position),
            'similarity_score': float(similarities[valid_indices.index(query_idx)]),
            'ndcg_score': float(ndcg),
            'top_15_matches': top_matches
        }
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return None

def process_and_evaluate_entries(entries: List[Dict], pdf_folder: str) -> Dict:
    all_results = []
    successful_entries = 0
    failed_entries = 0
    image_embeddings = process_all_images(entries, pdf_folder)
    for i, entry in enumerate(entries):
        query = entry.get('queries', {}).get('multimodal_query')
        if not query:
            failed_entries += 1
            continue
        result = process_single_query(query, image_embeddings, i, entries)
        if result:
            all_results.append(result)
            successful_entries += 1
        else:
            failed_entries += 1
    recall_positions = [r['recall_position'] for r in all_results]
    ndcg_scores = [r['ndcg_score'] for r in all_results]
    similarity_scores = [r['similarity_score'] for r in all_results]
    output = {
        'query_results': all_results,
        'summary': {
            'average_recall_position': float(np.mean(recall_positions)),
            'average_ndcg': float(np.mean(ndcg_scores)),
            'average_similarity': float(np.mean(similarity_scores)),
            'successful_entries': successful_entries,
            'failed_entries': failed_entries
        }
    }
    
    # Save results to the correct file
    with open('retrieval_results_fixed.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    return output