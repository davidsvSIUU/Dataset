#create_parquet.py
import pandas as pd
import json
from pathlib import Path
import base64
from pdf_utils import capture_page_image_hd
from tqdm import tqdm  # Pour avoir une barre de progression

def create_training_parquets(jsonl_paths: list, pdf_folder: str, output_folder: str):
    # Initialize lists for training data with language sorting
    questions_by_lang = {
        'EN': [], 'FR': [], 'ES': [], 'IT': [], 'DE': []
    }
    
    # Process all JSONL files
    processed_pdfs = set()
    total_lines_processed = 0
    
    for jsonl_path in jsonl_paths:
        if not Path(jsonl_path).exists():
            print(f"Warning: File not found: {jsonl_path}")
            continue
            
        print(f"\nProcessing {jsonl_path}...")
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    total_lines_processed += 1
                    try:
                        data = json.loads(line.strip())
                        if not isinstance(data, dict):
                            continue
                            
                        processed_pdfs.add(data.get('pdf_name', ''))
                        
                        queries_dict = data.get('queries', {})
                        if not isinstance(queries_dict, dict):
                            continue
                            
                        # Get declared language
                        lang = data.get('language', '')
                        if lang not in questions_by_lang:
                            continue
                            
                        # Create image ID
                        pdf_name = data.get('pdf_name', '')
                        page_number = data.get('page_number', '')
                        if not pdf_name or not isinstance(page_number, (int, str)):
                            continue
                            
                        image_id = f"{pdf_name}_{page_number}"
                        
                        # Add queries if they exist and aren't 'NaN' in any form
                        for query_key in ['query1', 'query2', 'query3']:
                            query = queries_dict.get(query_key, '')
                            if query and query != 'NaN' and '\\\"NaN\\\"' not in query and '"NaN"' not in query:
                                questions_by_lang[lang].append((query, image_id))
                                
                        if total_lines_processed % 1000 == 0:
                            print(f"Processed {total_lines_processed} lines...")
                            
                    except json.JSONDecodeError:
                        print(f"Warning: Invalid JSON in {jsonl_path}")
                        continue
                        
        except Exception as e:
            print(f"Error processing {jsonl_path}: {str(e)}")

    print(f"\nTotal JSONL lines processed: {total_lines_processed}")
    print(f"Found PDFs: {len(processed_pdfs)}")
    
    # Remove empty string from processed_pdfs if present
    processed_pdfs.discard('')
    print("PDFs to process:", sorted(list(processed_pdfs)))

    # Create sorted lists of queries and pos_ids
    queries = []
    pos_ids = []
    
    # Add questions in the specified order
    total_questions = 0
    for lang in ['EN', 'FR', 'ES', 'IT', 'DE']:
        lang_questions = len(questions_by_lang[lang])
        total_questions += lang_questions
        print(f"\nQuestions in {lang}: {lang_questions}")
        for query, pos in questions_by_lang[lang]:
            queries.append(query)
            pos_ids.append(pos)

    if total_questions == 0:
        print("Error: No valid questions found in the input files!")
        return

    print(f"\nTotal questions: {total_questions}")

    # Create and save train.parquet
    print("\nCreating train.parquet...")
    train_df = pd.DataFrame({
        'q': queries,
        'pos': pos_ids
    })
    
    # Initialize lists for corpus data
    docids = []
    images = []
    
    # Process all PDFs
    print("\nProcessing PDFs for corpus.parquet...")
    
    # Create a mapping of pages needed for each PDF
    pages_by_pdf = {}
    for pos in pos_ids:
        # Split on the last underscore to handle PDF names that contain underscores
        parts = pos.rsplit('_', 1)
        if len(parts) != 2:
            print(f"Warning: Invalid position format: {pos}")
            continue
        pdf_name, page_num = parts
        
        try:
            page_num = int(page_num)
            if pdf_name not in pages_by_pdf:
                pages_by_pdf[pdf_name] = set()
            pages_by_pdf[pdf_name].add(page_num)
        except ValueError:
            print(f"Warning: Invalid page number in position: {pos}")
            continue

    # Process each PDF
    total_pages = sum(len(pages) for pages in pages_by_pdf.values())
    print(f"\nTotal pages to process: {total_pages}")
    
    processed_pages = 0
    for pdf_name in sorted(pages_by_pdf.keys()):
        pdf_path = Path(pdf_folder) / pdf_name
        if not pdf_path.exists():
            print(f"Warning: PDF not found: {pdf_path}")
            continue
            
        print(f"\nProcessing {pdf_name} ({len(pages_by_pdf[pdf_name])} pages)")
        try:
            # Process pages for this PDF
            for page_num in sorted(pages_by_pdf[pdf_name]):
                processed_pages += 1
                image_id = f"{pdf_name}_{page_num}"
                print(f"Processing page {page_num} ({processed_pages}/{total_pages})")
                
                # Capture and encode image
                image_bytes = capture_page_image_hd(str(pdf_path), page_num)
                image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                
                docids.append(image_id)
                images.append(image_b64)
                
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            continue

    if not docids:
        print("Error: No images were processed!")
        return

    # Create and save corpus.parquet
    print("\nCreating corpus.parquet...")
    corpus_df = pd.DataFrame({
        'docid': docids,
        'image': images
    })

    # Create output directory if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Save parquet files
    train_df.to_parquet(Path(output_folder) / 'train.parquet', index=False)
    corpus_df.to_parquet(Path(output_folder) / 'corpus.parquet', index=False)

    print(f"\nFinal Stats:")
    print(f"train.parquet: {len(train_df)} queries")
    print(f"corpus.parquet: {len(corpus_df)} images")
    print(f"Files saved in {output_folder}")

if __name__ == "__main__":
    INPUT_DIR = "/Users/vuong/Desktop/dataset-compagnie-aerienneV2/FrenchBee"
    OUTPUT_FOLDER = "/Users/vuong/Desktop/geotechnie/FrenchBee_parquet_files"
    
    # Liste des fichiers JSONL à traiter
    jsonl_files = [
        "/Users/vuong/Desktop/query"
    ]
    
    create_training_parquets(jsonl_files, INPUT_DIR, OUTPUT_FOLDER)