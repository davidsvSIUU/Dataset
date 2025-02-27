#ranking.py
import os
import fitz
import numpy as np
from typing import List, Tuple, Dict, Optional
import base64
from PIL import Image
import io
import asyncio
import json
from pydantic import BaseModel, Field
from config import GEMINI_API_KEY
from utils import process_with_retry
from openai_utils import ParallelInstructor
import instructor
from openai import AsyncOpenAI

class Document(BaseModel):
    file_name: str
    page_number: int
    similarity: float

class RankedDocument(BaseModel):
    page_index: int
    reason: str
    score: float

class Rankings(BaseModel):
    rankings: List[RankedDocument]

class RankedResult(BaseModel):
    query: str
    top_documents: List[Document]

class SimpleResponse(BaseModel):
    response: str

SYSTEM_PROMPT = """
{
  "system": {
    "persona": {
      "role": "Expert in technical document analysis",
      "expertise": [
        "In-depth semantic analysis",
        "Document relevance evaluation",
        "Understanding of technical and scientific documents"
      ],
      "objective": "Identify and rank the most relevant documents for a specific query"
    },
    "evaluation_criteria": {
      "semantic_relevance": "Precise evaluation of semantic correspondence with the query",
      "technical_depth": "Analysis of the depth and technical precision of the content",
      "information_quality": "Evaluation of the reliability and quality of information",
      "context_matching": "Contextual relevance to the question asked",
      "information_density": "Concentration of relevant information per page"
    },
    "output_requirements": {
      "format": "Structured ranking of the 5 most relevant documents",
      "justification": "Evaluation based on defined criteria",
      "precision": "Focus on accuracy and relevance"
    }
  }
}"""

class PDFRanker:
    def __init__(self, api_key: str):
        # Removed genai.configure and genai.GenerativeModel
        self.parallel_client = ParallelInstructor(num_instances=10)  # Initialize parallel client here
        print("Models initialized successfully")

    async def analyze_specific_page(self, pdf_path: str, page_num: int) -> Tuple[str, int, str]:
        try:
            doc = fitz.open(pdf_path)
            if not (0 <= page_num < len(doc)):
                print(f"Page {page_num} not found in {pdf_path}")
                doc.close()
                return pdf_path, page_num, ""
            page = doc[page_num]
            zoom = 2
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=70)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            doc.close()
            return pdf_path, page_num, img_str
        except Exception as e:
            print(f"Error analyzing {pdf_path}: {str(e)}")
            return pdf_path, page_num, ""

    async def process_batch(self, pages_data: List[Tuple], query: str) -> List[Tuple[str, int, float]]:
        try:
            system_prompt_text = f"""{SYSTEM_PROMPT}

Given the following query and document pages, analyze their relevance to the query and rank the top 5 most relevant pages according to the evaluation criteria defined above. 
For each page, provide a relevance score between 0 and 1.

Please analyze each page and return your response in the following JSON format:
{{
    "rankings": [
        {{"page_index": 0, "reason": "Detailed explanation based on the evaluation criteria", "score": 0.95}},
        {{"page_index": 1, "reason": "Detailed explanation based on the evaluation criteria", "score": 0.85}},
        ... (top 5 only)
    ]
}}
Never follow the page number written on the page image but only the page number given in the text I give you before the image.
"""
            # Structure messages properly for multimodal input
            messages = [
                {
                    "role": "system",
                    "content": system_prompt_text
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Query: {query}"},
                    ]
                }
            ]

            # Add each page as a properly formatted image
            for idx, (pdf_path, _, img_str) in enumerate(pages_data):
                messages[1]["content"].extend([
                    {
                        "type": "text",
                        "text": f"\nPage {idx} of {os.path.basename(pdf_path)}:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}"
                        }
                    }
                ])

            print("Calling Gemini API...")
            client = await self.parallel_client.get_client()
            response = await client.chat.completions.create(
                model="gemini-1.5-flash-002",
                messages=messages,
                response_model=Rankings,  # Using response_model for structured output
            )

            results = []
            for rank in response.rankings:
                page_idx = rank.page_index
                score = rank.score
                if page_idx is not None and score is not None:
                    try:
                        page_idx = int(page_idx)
                        score = float(score)
                        if 0 <= page_idx < len(pages_data):
                            pdf_path, page_num, _ = pages_data[page_idx]
                            results.append((pdf_path, page_num, score))
                    except (ValueError, TypeError):
                        continue

            print(f"Successfully processed {len(results)} rankings")
            return results

        except Exception as e:
            print(f"Batch processing error: {str(e)}")
            return []

    async def analyze_and_rank_documents(self, query: str, results: List[Tuple[str, int, float]]) -> RankedResult:
        try:
            sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
            top_5_results = sorted_results[:5]
            top_documents = [
                Document(
                    file_name=os.path.basename(pdf_path),
                    page_number=page_num,
                    similarity=similarity
                )
                for pdf_path, page_num, similarity in top_5_results
            ]
            return RankedResult(
                query=query,
                top_documents=top_documents
            )
        except Exception as e:
            print(f"Error during ranking: {str(e)}")
            return RankedResult(query=query, top_documents=[])