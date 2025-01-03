import os
import asyncio
import json
import time
import random
from tqdm import tqdm
from typing import List, Tuple, Dict
from config import PDF_FOLDER, OUTPUT_FILE, RETRIEVAL_RESULTS_FILE, RANKED_RESULTS_FILE, GEMINI_API_KEY, REQUESTS_PER_SECOND
from utils import RateLimiter, process_with_retry, append_result_jsonl
from pdf_utils import pdf_to_images
from openai_utils import generate_technical_queries
from evaluation import load_random_jsonl_entries, process_and_evaluate_entries
from ranking import PDFRanker
import aiofiles
import litellm
litellm.set_verbose = True # Activation du mode verbose de litellm pour le débogage


class PDFProcessingResult:
    def __init__(self, pdf_name: str, queries: dict = None, processed_pages: List[int] = None, error: str = None):
        self.pdf_name = pdf_name
        self.queries = queries
        self.processed_pages = processed_pages if processed_pages is not None else []
        self.error = error


async def process_pdf_page(
    pdf_file: str,
    page_num: int,
    context_image: str,
    page_image: str,
    rate_limiter: RateLimiter,
    output_path: str
) -> Tuple[int, PDFProcessingResult]:
    start_time = time.time()
    try:
        queries = await process_with_retry(  # Appel asynchrone pour générer les requêtes
            generate_technical_queries,
            context_image,
            page_image,
            rate_limiter
        )
        result = (
            page_num,
            PDFProcessingResult(
                pdf_name=pdf_file,
                queries={
                    "main_query": queries.main_query,
                    "secondary_query": queries.secondary_query,
                    "visual_query": queries.visual_query,
                    "multimodal_query": queries.multimodal_query
                },
                processed_pages=[page_num],
                error=None
            )
        )
        processing_time = time.time() - start_time
    except Exception as e:
        print(f"Error processing page {page_num} of {pdf_file} after {time.time() - start_time:.2f} seconds: {str(e)}")
        result = (
            page_num,
            PDFProcessingResult(
                pdf_name=pdf_file,
                queries=None,
                processed_pages=[page_num],
                error=str(e)
            )
        )
    await append_result_jsonl(  # Ajout des résultats au fichier jsonl
        {
            "pdf_name": result[1].pdf_name,
            "page_number": result[0],
            "queries": result[1].queries,
            "error": result[1].error
        },
        output_path
    )
    return result


async def process_pdf(
    pdf_file: str,
    pdf_path: str,
    rate_limiter: RateLimiter,
    output_path: str,
    selected_pages: List[int]
) -> List[Tuple[int, PDFProcessingResult]]:
    try:
        page_images = pdf_to_images(pdf_path)
        context_image = page_images[0][1]
        results = []
        chunk_size = 5

        # Filter images based on selected pages (ne traite que les pages sélectionnées)
        selected_page_images = [img for img in page_images if img[0] in selected_pages]

        for i in range(0, len(selected_page_images), chunk_size):
            chunk = selected_page_images[i:i + chunk_size]
            chunk_tasks = []
            for page_num, page_image in chunk:
                task = asyncio.create_task(  # Création d'une tâche asynchrone pour chaque page
                    process_pdf_page(
                        pdf_file,
                        page_num,
                        context_image,
                        page_image,
                        rate_limiter,
                        output_path
                    )
                )
                chunk_tasks.append(task)
            chunk_results = await asyncio.gather(*chunk_tasks)  # Attend que toutes les tâches du chunk soient terminées
            results.extend(chunk_results)
        return results
    except Exception as e:
        print(f"Error processing {pdf_file}: {str(e)}")
        error_result = [(0, PDFProcessingResult(
            pdf_name=pdf_file,
            queries=None,
            processed_pages=[],
            error=str(e)
        ))]
        await append_result_jsonl(
            {
                "pdf_name": error_result[0][1].pdf_name,
                "page_number": error_result[0][0],
                "queries": error_result[0][1].queries,
                "error": error_result[0][1].error
            },
            output_path
        )
        return error_result


async def create_random_pages_json(folder_path: str, num_pages: int = 500, output_file: str = "random_pages.json") -> Dict[str, List[int]]:
    """
    Génère un fichier JSON contenant une liste de numéros de page aléatoires pour chaque PDF,
    totalisant environ `num_pages` sur tous les PDFs.

    Args:
        folder_path: Chemin vers le dossier contenant les fichiers PDF.
        num_pages: Nombre total de pages aléatoires à sélectionner sur tous les PDFs.
        output_file: Nom du fichier JSON de sortie.
    """
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    all_pages = [] # Liste pour stocker tous les tuples (pdf_file, page_number)

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        try:
            page_images = pdf_to_images(pdf_path) # Récupère toutes les pages d'un PDF
            for page_num, _ in page_images: # Ajoute chaque page dans la liste all_pages avec son nom de pdf
                all_pages.append((pdf_file, page_num))
        except Exception as e:
             print(f"Error processing {pdf_file}: {str(e)}")
    
    if len(all_pages) == 0:
      print("No pages found in any PDF")
      return {}
    
    num_pages_to_select = min(num_pages, len(all_pages)) # Sélectionne le nombre de pages à traiter (maximum num_pages)
    selected_pages = random.sample(all_pages, num_pages_to_select) # Choisi aléatoirement les pages à traiter

    random_pages = {} # Dictionnaire pour stocker les pages sélectionnées par pdf
    for pdf_file, page_num in selected_pages:
        if pdf_file not in random_pages:
            random_pages[pdf_file] = []
        random_pages[pdf_file].append(page_num)
    
    for pdf_file in random_pages:
      random_pages[pdf_file] = sorted(random_pages[pdf_file]) # Trie les numéros de page
    
    with open(output_file, 'w', encoding='utf-8') as f:  # Écrit les résultats dans un fichier JSON
        json.dump(random_pages, f, indent=2)
    
    print(f"Created random pages JSON file: {output_file}")
    return random_pages


async def process_pdf_folder(folder_path: str, output_path: str, random_pages: Dict[str, List[int]], num_query_pages: int = 100) -> Dict[str, List[Tuple[int, PDFProcessingResult]]]:
    """
    Traite les PDF du dossier en sélectionnant aléatoirement des pages pour les requêtes.
        folder_path: Chemin vers le dossier contenant les fichiers PDF.
        output_path: Chemin du fichier de sortie jsonl.
        random_pages: dictionnaire contenant les 500 pages sélectionnées pour le traitement.
        num_query_pages: le nombre de pages pour lesquelles les requêtes seront générées.
    """
    results = {}
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    async with aiofiles.open(output_path, 'w', encoding='utf-8') as f: # Ouvre le fichier de sortie en mode ecriture
        await f.write('') # vide le fichier si il existe
    rate_limiter = RateLimiter(requests_per_second=REQUESTS_PER_SECOND)
    tasks = [] # Liste pour stocker toutes les tâches asynchrones
    
    # Préparation des pages à traiter
    all_selected_pages = [] # liste temporaire des pages pour la sélection des 100 pages
    for pdf_file, pages in random_pages.items(): # Pour chaque pdf et pages dans le random_pages
        for page in pages:
            all_selected_pages.append((pdf_file, page)) # Ajout du tuple (pdf_file, page) à la liste
    
    num_query_pages = min(num_query_pages, len(all_selected_pages)) # Sélection du nombre max de pages pour le traitement des queries
    query_pages = random.sample(all_selected_pages, num_query_pages) # Selectionne aléatoirement les pages à traiter pour les queries
    query_pages_dict = {} # Dictionnaire pour stocker les pages à traiter pour les queries
    for pdf_file, page in query_pages: # Remplit query_pages_dict avec les pdfs comme clé, et les pages comme valeur
      if pdf_file not in query_pages_dict:
          query_pages_dict[pdf_file] = []
      query_pages_dict[pdf_file].append(page)
    
    for pdf_file in pdf_files: # Pour chaque fichier pdf
      pdf_path = os.path.join(folder_path, pdf_file)
      selected_pages = random_pages.get(pdf_file, []) # récupère la liste des pages sélectionnées pour ce pdf
      if selected_pages: # si la liste des pages n'est pas vide
            if pdf_file in query_pages_dict:  # si ce pdf contient des pages qui doivent être traitées pour les queries
              tasks.append(process_pdf( # Appel de la fonction process_pdf uniquement si la condition précédente est vraie
                pdf_file,
                pdf_path,
                rate_limiter,
                output_path,
                query_pages_dict[pdf_file]
            ))
    pdf_results = await asyncio.gather(*tasks) # Attend que toutes les tâches soient terminées
    for pdf_file, result in zip(pdf_files, pdf_results):
        if pdf_file in random_pages:  #  ne sauvegarde le résultat que si le pdf fait parti de la liste de ceux contenant les 500 pages
            results[pdf_file] = result
    return results


async def main():
    try:
        print("\n=== Starting Program ===")
        print("Creating random pages JSON...")
        random_pages = await create_random_pages_json(PDF_FOLDER) # Génère les 500 pages aléatoires
        
        print("Starting PDF processing...")
        results = await process_pdf_folder(PDF_FOLDER, OUTPUT_FILE, random_pages)  # Traite les PDFs avec la liste des 500 pages, et sélectionne 100 pages pour les queries.
        total_pdfs = len(results)
        total_pages = sum(len(page_results) for page_results in results.values()) # Compte le nombre total de pages traitées
        successful_pages = sum(
            sum(1 for _, result in page_results if not result.error) # Compte le nombre de pages traitées avec succès
            for page_results in results.values()
        )
        print(f"Processed {total_pdfs} PDFs with {total_pages} pages, {successful_pages} successful pages")
        print("Starting evaluation...")
        entries = load_random_jsonl_entries(OUTPUT_FILE) # Charge les données pour l'évaluation
        print(f"Loaded {len(entries)} random entries for evaluation")
        evaluation_results = process_and_evaluate_entries(entries, PDF_FOLDER) # Évalue les résultats
        print(f"Evaluation Results:")
        print(f"Average Recall Position: {evaluation_results['summary']['average_recall_position']:.2f}")
        print(f"Average NDCG: {evaluation_results['summary']['average_ndcg']:.4f}")
        print(f"Average Similarity: {evaluation_results['summary']['average_similarity']:.4f}")
        print(f"Successful entries: {evaluation_results['summary']['successful_entries']}")
        print(f"Failed entries: {evaluation_results['summary']['failed_entries']}")
        print(f"Full results saved to retrieval_results.json")
        print("Starting ranking...")
        ranker = PDFRanker(GEMINI_API_KEY) # Instance du classe de ranking
        with open(RETRIEVAL_RESULTS_FILE, 'r', encoding='utf-8') as f: # Charge les données du fichier retrieval_results.json
            retrieval_data = json.load(f)
        queries_to_process = retrieval_data['query_results'] # Récupère les données à traiter
        total_queries = len(queries_to_process)
        print(f"\n=== Processing {total_queries} queries for ranking ===")
        results_data = []
        for query_index, query_result in enumerate(tqdm(queries_to_process, desc="Ranking Queries"), 1): # Parcourt toutes les queries
            query_results = {
                "query": query_result.get('query', ''),
                "ranked_documents": []
            }
            pdfs_to_analyze = []
            for match in query_result.get('top_15_matches', []): # Parcourt les 15 top matchs
                pdf_name = match.get('pdf_name')
                page_number = match.get('page_number')
                if pdf_name and pdf_name.strip() and page_number is not None: # verifie que le pdf_name n'est pas vide et que page_number n'est pas null
                    pdf_path = os.path.join(PDF_FOLDER, pdf_name)
                    if os.path.exists(pdf_path):
                        pdfs_to_analyze.append((pdf_path, page_number))
            if not pdfs_to_analyze:
                print(f"No valid PDFs found for query {query_index}")
                continue
            try:
                pages_data = []
                for pdf_path, page_num in pdfs_to_analyze: # Parcourt la liste des pdf à analyser
                    if not pdf_path:
                        print(f"Empty PDF path received, skipping. Query {query_index}")
                        continue
                    try:
                        data = await ranker.analyze_specific_page(pdf_path, page_num) # Analyse la page specifique
                        if data[2]:
                            pages_data.append(data)
                    except Exception as e:
                        print(f"Error processing {os.path.basename(pdf_path)}: {str(e)}")
                if not pages_data:
                    print("No valid page data extracted")
                    continue
                initial_results = await ranker.process_batch(pages_data, query_result['query'])
                if not initial_results:
                    print("No results from Gemini processing")
                    continue
                ranked_results = await ranker.analyze_and_rank_documents(query_result['query'], initial_results) # Analyse et ordonne les résultats
                for i, doc in enumerate(ranked_results.top_documents, 1):
                    query_results["ranked_documents"].append({
                        "rank": i,
                        "file_name": doc.file_name,
                        "page": doc.page_number
                    })

                output_dir = os.path.dirname(RANKED_RESULTS_FILE)
                if output_dir and output_dir != '':
                    os.makedirs(output_dir, exist_ok=True)
                else:
                     print(f"Invalid directory: {output_dir}")

                print("Saving current results...")
                results_data.append(query_results)
                with open(RANKED_RESULTS_FILE, 'w', encoding='utf-8') as f: # Sauvegarde les resultats
                    json.dump(results_data, f, indent=2, ensure_ascii=False)
                print(f"Results saved to: {RANKED_RESULTS_FILE}")
            except Exception as e:
                print(f"Error processing query {query_index}: {str(e)}")
                continue
    except Exception as e:
        print(f"Fatal error: {str(e)}")
    finally:
        try:
            await asyncio.sleep(0)
            print("Program finished")
        except Exception as e:
            print(f"Cleanup error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())