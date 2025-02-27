import os 
import asyncio
import aiofiles
import json
import base64
import fitz
from datetime import datetime
from pydantic import BaseModel
import instructor
from litellm import Field, acompletion
from pdf_utils import capture_page_image_hd
import random

class TechnicalQueries(BaseModel):
    relevant: bool
    reference_query: str = Field(..., 
        description="Query technique en français avec formulation originale et variée. Doit éviter les débuts répétitifs.")
    en_query: str
    es_query: str
    de_query: str
    it_query: str

SYSTEM_PROMPT = """You are a geotechnical engineering expert specialized in analyzing technical documents.
Your task is to:
1. Analyze the provided technical content from geotechnical documents.
2. Generate one complex technical query in English that demonstrates deep understanding. Use varied formulations:
   - Technical questions (why, how, compare...)
   - Varied openings (imperative, verbal noun, complex sentences)
   - Alternative formulations to "Quelle/Quel"
3. Translate this query accurately into French, Spanish, German, and Italian while maintaining technical precision.
4. Determine if the image contains maps, cartography, or photos that support the technical topic.

The query should:
- Use varied structures:
  * "Analyze the factors influencing..."
  * "Compare the methods of..."
  * "Evaluate the impact of..."
  * "Determine the key parameters for..."
  * "Explain the correlation between..."
  * "Propose a methodology for..."
- Avoid repetitive patterns in query openings
- Mix different grammatical structures
- Maintain technical depth

New examples of valid French queries:
"Analyser les critères de choix des paramètres de résistance au cisaillement dans ce contexte géologique particulier.",
"Comparer les approches de calcul de stabilité des pentes selon les différentes normes internationales présentes dans le document.",
"Évaluer l'influence de la granulométrie des sols sur les techniques de compactage recommandées.",
"Déterminer les indicateurs clés de risque de liquéfaction des sols dans le cas étudié.",
"Pourquoi les essais pressiométriques sont-ils privilégiés pour ce type de formation rocheuse selon les données présentées ?",
"Expliquer la relation entre les caractéristiques minéralogiques des argiles et leur comportement mécanique observé."

Note: Return 'NaN' for non-technical/generic content
"""
async def generate_queries(context_image_b64: str, page_image_b64: str) -> TechnicalQueries:
    try:
        client = instructor.from_litellm(acompletion)
        response = await client.chat.completions.create(
            model="gemini/gemini-1.5-flash-002",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Generate a technical query based on these pages:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{context_image_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{page_image_b64}"}}
                    ]
                }
            ],
            response_model=TechnicalQueries
        )
        return response
    except Exception as e:
        print(f"Error generating queries: {str(e)}")
        raise

async def get_total_pages_info(pdf_folder: str) -> list:
    """Collecte les informations sur toutes les pages disponibles dans tous les PDFs"""
    pages_info = []
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        try:
            pdf_document = fitz.open(pdf_path)
            for page_num in range(len(pdf_document)):
                pages_info.append({
                    'pdf_file': pdf_file,
                    'pdf_path': pdf_path,
                    'page_num': page_num
                })
            pdf_document.close()
        except Exception as e:
            print(f"Error reading {pdf_file}: {str(e)}")
    
    return pages_info

async def process_pdf_page(page_info: dict, context_image_b64: str, output_path: str):
    try:
        page_image = capture_page_image_hd(page_info['pdf_path'], page_info['page_num'])
        page_image_b64 = base64.b64encode(page_image).decode('utf-8')
        
        queries = await generate_queries(context_image_b64, page_image_b64)
        result = {
            "pdf_name": page_info['pdf_file'],
            "page_number": page_info['page_num'],
            "timestamp": datetime.now().isoformat(),
            "queries": {    
                "relevant": queries.relevant,
                "reference": queries.reference_query,
                "en": queries.en_query,
                "es": queries.es_query,
                "de": queries.de_query,
                "it": queries.it_query
            }
        }
        
        # Écrire uniquement si le traitement est réussi
        async with aiofiles.open(output_path, 'a', encoding='utf-8') as f:
            await f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f"Processed and saved page {page_info['page_num']} of {page_info['pdf_file']}")
            
    except Exception as e:
        # Simplement logger l'erreur sans l'écrire dans le fichier
        print(f"Error processing page {page_info['page_num']} of {page_info['pdf_file']}: {str(e)}")
        
async def main():
    PDF_FOLDER = "/Users/vuong/Desktop/geotechnie/dataset-benchmark-v2"
    OUTPUT_FILE = "/Users/vuong/Desktop/geotechnie/benchmark-query.jsonl"
    PAGES_TO_PROCESS = 1000
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Créer/vider le fichier de sortie
    async with aiofiles.open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        await f.write('')
    
    # Collecter toutes les pages disponibles
    all_pages = await get_total_pages_info(PDF_FOLDER)
    
    if len(all_pages) < PAGES_TO_PROCESS:
        print(f"Warning: Only {len(all_pages)} pages available, processing all of them")
        selected_pages = all_pages
    else:
        selected_pages = random.sample(all_pages, PAGES_TO_PROCESS)
    
    # Grouper les pages par PDF pour optimiser la lecture du contexte
    pages_by_pdf = {}
    for page in selected_pages:
        if page['pdf_file'] not in pages_by_pdf:
            pages_by_pdf[page['pdf_file']] = []
        pages_by_pdf[page['pdf_file']].append(page)
    
    # Traiter les pages sélectionnées
    for pdf_file, pages in pages_by_pdf.items():
        try:
            # Capturer l'image de contexte une seule fois par PDF
            context_image = capture_page_image_hd(pages[0]['pdf_path'], 0)
            context_image_b64 = base64.b64encode(context_image).decode('utf-8')
            
            # Traiter toutes les pages sélectionnées pour ce PDF
            for page_info in pages:
                await process_pdf_page(page_info, context_image_b64, OUTPUT_FILE)
                
        except Exception as e:
            print(f"Error processing PDF {pdf_file}: {str(e)}")
    
    print(f"Completed processing {PAGES_TO_PROCESS} random pages")

if __name__ == "__main__":
    asyncio.run(main())
