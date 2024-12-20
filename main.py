import os
import asyncio
import json
import time
from tqdm import tqdm
from typing import List, Tuple, Dict
from config import PDF_FOLDER, OUTPUT_FILE, RETRIEVAL_RESULTS_FILE, RANKED_RESULTS_FILE, GEMINI_API_KEY, REQUESTS_PER_SECOND
from utils import RateLimiter, process_with_retry, append_result_jsonl
from pdf_utils import pdf_to_images
from openai_utils import generate_technical_queries
from evaluation import load_random_jsonl_entries, process_and_evaluate_entries
from ranking import PDFRanker
import aiofiles

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
        queries = await process_with_retry(
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
    await append_result_jsonl(
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
    output_path: str
) -> List[Tuple[int, PDFProcessingResult]]:
    try:
        page_images = pdf_to_images(pdf_path)
        context_image = page_images[0][1]
        results = []
        chunk_size = 5
        for i in range(1, len(page_images), chunk_size):
            chunk = page_images[i:i + chunk_size]
            chunk_tasks = []
            for page_num, page_image in chunk:
                task = asyncio.create_task(
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
            chunk_results = await asyncio.gather(*chunk_tasks)
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

async def process_pdf_folder(folder_path: str, output_path: str) -> Dict[str, List[Tuple[int, PDFProcessingResult]]]:
    results = {}
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
        await f.write('')
    rate_limiter = RateLimiter(requests_per_second=REQUESTS_PER_SECOND)
    tasks = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        tasks.append(process_pdf(
            pdf_file,
            pdf_path,
            rate_limiter,
            output_path
        ))
    pdf_results = await asyncio.gather(*tasks)
    for pdf_file, result in zip(pdf_files, pdf_results):
        results[pdf_file] = result
    return results

async def main():
    try:
        print("\n=== Starting Program ===")
        print("Starting PDF processing...")
        results = await process_pdf_folder(PDF_FOLDER, OUTPUT_FILE)
        total_pdfs = len(results)
        total_pages = sum(len(page_results) for page_results in results.values())
        successful_pages = sum(
            sum(1 for _, result in page_results if not result.error)
            for page_results in results.values()
        )
        print(f"Processed {total_pdfs} PDFs with {total_pages} pages, {successful_pages} successful pages")
        print("Starting evaluation...")
        entries = load_random_jsonl_entries(OUTPUT_FILE)
        print(f"Loaded {len(entries)} random entries for evaluation")
        evaluation_results = process_and_evaluate_entries(entries, PDF_FOLDER)
        print(f"Evaluation Results:")
        print(f"Average Recall Position: {evaluation_results['summary']['average_recall_position']:.2f}")
        print(f"Average NDCG: {evaluation_results['summary']['average_ndcg']:.4f}")
        print(f"Average Similarity: {evaluation_results['summary']['average_similarity']:.4f}")
        print(f"Successful entries: {evaluation_results['summary']['successful_entries']}")
        print(f"Failed entries: {evaluation_results['summary']['failed_entries']}")
        print(f"Full results saved to retrieval_results.json")
        print("Starting ranking...")
        ranker = PDFRanker(GEMINI_API_KEY)
        with open(RETRIEVAL_RESULTS_FILE, 'r', encoding='utf-8') as f:
            retrieval_data = json.load(f)
        queries_to_process = retrieval_data['query_results']
        total_queries = len(queries_to_process)
        print(f"\n=== Processing {total_queries} queries for ranking ===")
        results_data = []
        for query_index, query_result in enumerate(tqdm(queries_to_process, desc="Ranking Queries"), 1):
            query_results = {
                "query": query_result.get('query', ''),
                "ranked_documents": []
            }
            pdfs_to_analyze = []
            for match in query_result.get('top_15_matches', []):
                pdf_name = match.get('pdf_name')
                page_number = match.get('page_number')
                if pdf_name and pdf_name.strip() and page_number is not None:
                    pdf_path = os.path.join(PDF_FOLDER, pdf_name)
                    if os.path.exists(pdf_path):
                        pdfs_to_analyze.append((pdf_path, page_number))
            if not pdfs_to_analyze:
                print(f"No valid PDFs found for query {query_index}")
                continue
            try:
                pages_data = []
                for pdf_path, page_num in pdfs_to_analyze:
                    if not pdf_path:
                        print(f"Empty PDF path received, skipping. Query {query_index}")
                        continue
                    try:
                        data = await ranker.analyze_specific_page(pdf_path, page_num)
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
                ranked_results = await ranker.analyze_and_rank_documents(query_result['query'], initial_results)
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
                with open(RANKED_RESULTS_FILE, 'w', encoding='utf-8') as f:
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