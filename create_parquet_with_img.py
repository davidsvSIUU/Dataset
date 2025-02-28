import fitz
import base64
import tempfile
from typing import List, Tuple, Dict
import os
import json
import pandas as pd
from datasets import Dataset, Features, Value, Image
import logging
from PIL import Image as PILImage
import io
import random
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Define the path directly in the code
RANKED_RESULTS_FILE = "/Users/vuong/Desktop/geotechnie/AirFranceKLM-query.jsonl"
PDF_FOLDER = "/Users/vuong/Desktop/dataset-compagnie-aerienneV2/AirFranceKLM"


def capture_page_image_hd(pdf_path: str, page_number: int) -> bytes:
    try:
        pdf_document = fitz.open(pdf_path)
        page = pdf_document[page_number]
        
        # Increase the resolution/DPI
        zoom = 2.0  # Increase this value for higher resolution
        mat = fitz.Matrix(zoom, zoom)  # Create transformation matrix
        
        # Get the pixmap with enhanced settings
        pix = page.get_pixmap(
            matrix=mat,            # Apply zoom transformation
            alpha=False,           # Remove alpha channel for cleaner output
            colorspace=fitz.csRGB  # Ensure RGB colorspace
        )
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            pix.save(tmp_file.name)
            with open(tmp_file.name, 'rb') as image_file:
                return image_file.read()
    finally:
        if 'pdf_document' in locals():
            pdf_document.close()
        if 'tmp_file' in locals():
            try:
                os.unlink(tmp_file.name)
            except:
                pass

def visualize_image(image_bytes: bytes):
    """Décode et affiche l'image à partir de bytes."""
    try:
        image = PILImage.open(io.BytesIO(image_bytes))
        plt.imshow(image)
        plt.axis('off')  # Désactiver les axes
        plt.show()
    except Exception as e:
      print(f"Erreur lors de l'affichage de l'image : {e}")

def load_and_transform_ranked_results(file_path: str) -> List[Dict]:
    transformed_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    print("Extracted entry:", entry)
                    pdf_name = entry.get("pdf_name")
                    page_number = entry.get("page_number")
                    queries = entry.get("queries")
                    
                    if queries is None:
                       print("Skipping entry, queries is None")
                       continue
                    
                    has_valid_query = False
                    for i in range(1, 4):
                        query = queries.get(f"query{i}", "NaN")
                        if query != "NaN":
                            
                            pdf_path = os.path.join(PDF_FOLDER, pdf_name)
                            try:
                                image_bytes = capture_page_image_hd(pdf_path, page_number)
                                if image_bytes:
                                  transformed_data.append({
                                        "query": query,
                                        "ranked_documents": json.dumps([{"file_name": pdf_name, "page": page_number}]),
                                        "image": image_bytes  # Add the image bytes to the dictionary
                                  })
                                  has_valid_query = True
                                else :
                                  print(f"Image not found for {pdf_name}, page {page_number}") # debug message if pil_image is None
                            except Exception as e:
                                logger.error(f"Error processing image: {e}")
                                 
            
                    if not has_valid_query:
                         print("Skipping entry, no valid query.")
                         continue
                    
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON: {e} in line: {line}")
                    continue
    except Exception as e:
        logger.error(f"Error loading ranked results: {str(e)}")
        return []
    return transformed_data


def create_query_dataset(output_dir: str = "french_technical_dataset"):
    """Create query dataset with ranked documents."""
    try:
        # Create output directories
        queries_dir = os.path.join(output_dir)
        os.makedirs(queries_dir, exist_ok=True)
        
        # Load and transform ranked results
        transformed_results = load_and_transform_ranked_results(RANKED_RESULTS_FILE)
        print(f"Length of transformed results: {len(transformed_results)}")
        
        # Process queries
        query_data = transformed_results[:500]
        
        # Create DataFrame
        df = pd.DataFrame(query_data)
        
        # Define features for the dataset
        features = Features({
              'query': Value('string'),
              'ranked_documents': Value('string'),
              'image': Image(decode=True)
          })
        
        # Convert to Hugging Face Dataset
        dataset = Dataset.from_pandas(df, features=features)
        
        # Save directly as parquet
        output_file = os.path.join(queries_dir, "test-comp-aer.parquet")
        dataset.to_parquet(output_file)
        
        logger.info(f"Successfully created query dataset in: {output_file}")
        logger.info(f"Total queries: {len(query_data)}")
        
        return query_data
        
    except Exception as e:
        logger.error(f"Error creating query dataset: {str(e)}", exc_info=True)
        return None

def create_image_dataset(query_data, output_dir: str = "french_technical_dataset"):
    """Create image dataset with pages only from ranked documents (no random pages)."""
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
            'query': Value('string'),
            'image': Image(decode=True)
        })
        
        # Add mandatory pages from ranked documents
        for item in query_data:
            ranked_docs = json.loads(item['ranked_documents'])
            query_text = item['query']  # Get the query for this image
            for doc in ranked_docs:
                doc_key = (doc['file_name'], doc['page'])
                if doc_key not in processed_docs:
                    pdf_path = os.path.join(PDF_FOLDER, doc['file_name'])
                    if os.path.exists(pdf_path):
                        try:
                            image_bytes = capture_page_image_hd(pdf_path, doc['page'])
                            if image_bytes:
                                image_data.append({
                                    'file_name': doc['file_name'],
                                    'page': doc['page'],
                                    'query': query_text,
                                    'image': image_bytes
                                })
                                processed_docs.add(doc_key)
                        except Exception as e:
                            logger.error(f"Error processing PDF page {doc['file_name']} page {doc['page']}: {str(e)}")
        
        # Create DataFrame
        df = pd.DataFrame(image_data)
        
        # Convert to Hugging Face Dataset with specified features
        dataset = Dataset.from_pandas(df, features=features)
        
        # Save directly as parquet
        output_file = os.path.join(images_dir, "train-comp-aer.parquet")
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