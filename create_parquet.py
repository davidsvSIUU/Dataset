import os
import json
import pandas as pd
from datasets import Dataset, Features, Value, Image
from typing import List, Dict
from config import RANKED_RESULTS_FILE, PDF_FOLDER
import logging
from pdf_utils import capture_page_image_hd
from PIL import Image as PILImage
import io
import fitz
import random

logger = logging.getLogger(__name__)

def load_ranked_results(file_path: str) -> List[Dict]:
    """Load ranked results from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading ranked results: {str(e)}")
        return []

def create_query_dataset(output_dir: str = "french_technical_dataset"):
    """Create query dataset with ranked documents."""
    try:
        # Create output directories
        queries_dir = os.path.join(output_dir)
        os.makedirs(queries_dir, exist_ok=True)
        
        # Load ranked results
        ranked_results = load_ranked_results(RANKED_RESULTS_FILE)
        
        # Process queries
        query_data = []
        for result in ranked_results[:500]:  # Limit to 500 queries
            query_data.append({
                'query': result.get('query', ''),
                'ranked_documents': json.dumps(result.get('ranked_documents', []))
            })
        
        # Create DataFrame
        df = pd.DataFrame(query_data)
        
        # Convert to Hugging Face Dataset
        dataset = Dataset.from_pandas(df)
        
        # Save directly as parquet
        output_file = os.path.join(queries_dir, "test-00000-of-00001.parquet")
        dataset.to_parquet(output_file)
        
        logger.info(f"Successfully created query dataset in: {output_file}")
        logger.info(f"Total queries: {len(query_data)}")
        
        return query_data
        
    except Exception as e:
        logger.error(f"Error creating query dataset: {str(e)}", exc_info=True)
        return None

def create_image_dataset(query_data, output_dir: str = "french_technical_dataset", total_pages: int = 5000):
    """Create image dataset with a total of 5000 pages including mandatory and random pages."""
    try:
        # Create output directories
        images_dir = os.path.join(output_dir)
        os.makedirs(images_dir, exist_ok=True)
        
        image_data = []
        processed_docs = set()  # To avoid duplicates
        
        # Define features for the dataset
        features = Features({
            'file_name': Value('string'),
            'page': Value('int64'),
            'image': Image(decode=True)
        })
        
        # Add mandatory pages from ranked documents
        for item in query_data:
            ranked_docs = json.loads(item['ranked_documents'])
            for doc in ranked_docs:
                doc_key = (doc['file_name'], doc['page'])
                if doc_key not in processed_docs:
                    pdf_path = os.path.join(PDF_FOLDER, doc['file_name'])
                    if os.path.exists(pdf_path):
                        try:
                            pil_image = capture_page_image_hd(pdf_path, doc['page'])
                            if pil_image:
                                image_data.append({
                                    'file_name': doc['file_name'],
                                    'page': doc['page'],
                                    'image': pil_image
                                })
                                processed_docs.add(doc_key)
                                if len(image_data) >= total_pages:
                                    break
                        except Exception as e:
                            logger.error(f"Error processing PDF page {doc['file_name']} page {doc['page']}: {str(e)}")
            if len(image_data) >= total_pages:
                break
        
        # Calculate remaining pages needed
        remaining = total_pages - len(image_data)
        
        if remaining > 0:
            # Collect all possible random pages
            random_pages = []
            pdf_files = os.listdir(PDF_FOLDER)
            for pdf_file in pdf_files:
                pdf_path = os.path.join(PDF_FOLDER, pdf_file)
                if os.path.isfile(pdf_path):
                    pdf_name = os.path.splitext(pdf_file)[0]
                    try:
                        pdf_document = fitz.open(pdf_path)
                        total_pages_in_pdf = pdf_document.page_count
                        pdf_document.close()
                        for page_num in range(1, total_pages_in_pdf + 1):
                            doc_key = (pdf_name, page_num)
                            if doc_key not in processed_docs:
                                random_pages.append(doc_key)
                    except Exception as e:
                        logger.error(f"Error opening PDF {pdf_path}: {str(e)}")
                        continue
            
            # Sample without replacement
            sampled_pages = random.sample(random_pages, min(remaining, len(random_pages)))
            
            for doc_key in sampled_pages:
                pdf_name, page_num = doc_key
                pdf_path = os.path.join(PDF_FOLDER, f"{pdf_name}.pdf")
                try:
                    pil_image = capture_page_image_hd(pdf_path, page_num)
                    if pil_image:
                        image_data.append({
                            'file_name': pdf_name,
                            'page': page_num,
                            'image': pil_image
                        })
                        processed_docs.add(doc_key)
                except Exception as e:
                    logger.error(f"Error processing PDF page {pdf_name} page {page_num}: {str(e)}")
        
        # Create DataFrame
        df = pd.DataFrame(image_data)
        
        # Convert to Hugging Face Dataset with specified features
        dataset = Dataset.from_pandas(df, features=features)
        
        # Save directly as parquet
        output_file = os.path.join(images_dir, "images-00000-of-00001.parquet")
        dataset.to_parquet(output_file)
        
        logger.info(f"Successfully created image dataset in: {output_file}")
        logger.info(f"Total images: {len(image_data)}")
        
    except Exception as e:
        logger.error(f"Error creating image dataset: {str(e)}", exc_info=True)
def main():
    output_dir = "./"
    
    # Create query dataset first
    query_data = create_query_dataset(output_dir)
    
    if query_data:
        # Create image dataset using the query data
        create_image_dataset(query_data, output_dir)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    main()