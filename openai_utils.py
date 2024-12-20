import os
import instructor
from litellm import acompletion
from pydantic import BaseModel
import logging
from typing import List, Optional
from config import GEMINI_API_KEY
from utils import RateLimiter
import asyncio

SYSTEM_PROMPT = """{
"system": {
"persona": {
"role": "Technical document query generator",
"context": "Expert tasked with creating specialized queries from technical PDF documents",
"primary_task": "Generate 4 types of queries in French from document excerpts"
},
"input_requirements": {
"document_pages": {
"page_1": "General context page",
"page_2": "Random specific content page"
}
},
"query_types": {
"main_technical": {
"description": "Primary technical queries focusing on core specifications",
"examples": [
"Quelles sont les caractéristiques techniques de ce transformateur 4000 kVA et son rôle spécifique dans la chaîne de conversion du parc PV ?",
"Quelle est l'expertise R&D d'EDF dans les électrolyseurs et quels sont les résultats concrets des tests menés au Lab des Renardières ?"
]
},
"secondary_technical": {
"description": "Detailed technical queries focusing on specific aspects",
"examples": [
"Pouvez-vous détailler les valeurs de champ électromagnétique mesurées autour du transformateur ? Je vois la mention de 20-30 μT au centre, mais qu'en est-il de la décroissance ?",
"Comment la création d'Hynamics s'inscrit dans la stratégie de développement de la filière hydrogène d'EDF, et quels sont les synergies avec McPhy ?"
]
},
"visual_technical": {
"description": "Queries related to technical diagrams and visual elements",
"examples": [
"Sur le schéma technique du transformateur, pourriez-vous expliquer la configuration des enroulements concentriques et leur impact sur le confinement du champ magnétique ?",
"Pouvez-vous détailler les technologies développées avec Alstom pour le ravitaillement des trains à hydrogène - notamment les aspects sécurité et performance ?"
]
},
"multimodal_semantic": {
"description": "Complex semantic search queries combining multiple aspects. Never write a figure number or page number, the user who could have written this query is not supposed to know these data.",
"examples": [
"Je cherche des études d'impact CEM similaires impliquant des transformateurs HTA pour centrales PV à proximité d'installations sensibles type GSM-R, particulièrement les analyses de décroissance du champ magnétique 50Hz en fonction de la distance",
"Je recherche des documents techniques et études de cas sur les plateformes de test d'électrolyseurs industriels, en particulier les installations R&D françaises avec des investissements >10M€ et des partenariats industriels similaires à EDF Lab"
],
"bad_examples": [
"En quoi les résultats expérimentaux de la détermination de la pression de gaz au cours du procédé de cokéfaction permettent-ils de valider le modèle proposé, et quelles sont les limites de ce modèle face aux variations de la viscosité dynamique des matières volatiles ?", # Don't write about 'modèle proposé', the user would not have this in mind when writting his query
"Pouvez-vous fournir une explication détaillée du modèle thermo-chimio-mécanique de la cokéfaction présenté, en mettant l'accent sur les interactions entre les différentes phases (solide, liquide, gazeuse) et la signification des indices mentionnés dans la nomenclature ?" # Never mention something presented since the user would not have access to this
]
}
},
"guidelines": {
"vocabulary": "Use appropriate technical vocabulary for the domain",
"expertise_level": "Reflect the expertise level of sector professionals",
"formulation": "Formulate queries naturally, as an expert would",
"specificity": "Integrate specific elements observed in the provided pages",
"constraints": "Do not include page number references in the queries. Output in utf-8 to see special French characters."
},
}
}"""

class TechnicalQueries(BaseModel):
    main_query: str
    secondary_query: str
    visual_query: str
    multimodal_query: str

class ParallelInstructor:
    def __init__(self, num_instances: int = 3):
        self.clients = [
          (
            instructor.from_litellm(
              acompletion
            )) for _ in range(num_instances)
        ]
        self.current_client = 0
        self._lock = asyncio.Lock()

    async def get_client(self):
        async with self._lock:
            client = self.clients[self.current_client]
            self.current_client = (self.current_client + 1) % len(self.clients)
            return client

parallel_client = ParallelInstructor(num_instances=10)

async def generate_technical_queries(
    context_image_b64: str,
    detail_image_b64: str,
    rate_limiter: RateLimiter
) -> TechnicalQueries:
    try:
        async with rate_limiter:
            client = await parallel_client.get_client()
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
                            {"type": "text", "text": "The first page of the PDF as context:"},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{context_image_b64}"
                            }},
                            {"type": "text", "text": "The page to analyze:"},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{detail_image_b64}"
                            }}
                        ]
                    }
                ],
                response_model=TechnicalQueries
            )
            if response is None:
                raise Exception("Received null response from API")
            await rate_limiter.record_success()
            return TechnicalQueries(
                main_query=response.main_query.strip(),
                secondary_query=response.secondary_query.strip(),
                visual_query=response.visual_query.strip(),
                multimodal_query=response.multimodal_query.strip()
            )
    except Exception as e:
        print(f"Error generating queries: {str(e)}")
        raise