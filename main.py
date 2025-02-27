import os 
import asyncio
import aiofiles
from datetime import datetime
import json
import time
from typing import List, Tuple, Dict
import base64
import fitz
from pdf_utils import capture_page_image_hd
import instructor
from litellm import acompletion
from pydantic import BaseModel

SYSTEM_PROMPT = """{
"system": {
"persona": {
"role": "Générateur de requêtes techniques géotechniques",
"context": "Expert en génération de requêtes spécialisées à partir de documents PDF techniques",
"primary_task": "Générer 3 types de requêtes en français à partir d'extraits de documents géotechniques"
},
"input_requirements": {
"document_pages": {
"page_1": "Page de contexte général",
"page_2": "Page de contenu spécifique aléatoire"
}
},
"query_types": {
"main_technical": {
"description": "Requêtes techniques principales sur les spécifications géotechniques de base",
"examples": [
"Quelles sont les caractéristiques mécaniques des sols argileux dans ce contexte géologique et leur influence sur la stabilité des fondations profondes ?",
"Quelle méthodologie est préconisée pour l'analyse de stabilité des talus en terrain hétérogène selon les normes Eurocode 7 ?"
]
},
"secondary_technical": {
"description": "Requêtes techniques détaillées sur des aspects spécifiques",
"examples": [
"Pouvez-vous préciser les valeurs de résistance au cisaillement drainé obtenues lors des essais triaxiaux sur les limons compactés ?",
"Comment les mesures in situ de pression interstitielle ont-elles influencé le dimensionnement des pieux forés dans cette étude de cas ?"
]
},
"visual_technical": {
"description": "Requêtes liées aux schémas techniques et éléments visuels",
"examples": [
"Sur le diagramme de classification USCS, pourriez-vous expliquer la détermination du symbole SC pour ce sol et ses implications en termes de portance ?",
"Sur la coupe géotechnique, comment interpréter la variation du SPT N60 entre les couches de graviers et les sables limoneux ?"
]
},
"multimodal_semantic": {
"description": "Requêtes sémantiques complexes combinant plusieurs aspects. Ne jamais mentionner de numéros de figure ou de page.",
"examples": [
"Je recherche des études comparatives sur les méthodes de compactage dynamique pour sols compressibles en zone sismique, avec analyse des tassements différentiels post-traitement",
"Je cherche des rapports techniques sur l'utilisation d'inclusions rigides en contexte urbain dense, notamment les interactions sol-structure pour des projets de soutènement profonds (>15m)"
],
"bad_examples": [
"Quel est l'impact de la valeur du coefficient de Poisson déduite des essais oedométriques sur le module élastique utilisé dans la modélisation par éléments finis ?", # Éviter les références à des modèles spécifiques
"Pouvez-vous détailler l'équation 4.2 du chapitre sur la perméabilité variable dans les sols non saturés ?" # Ne pas référencer d'équations
]
}
},
"guidelines": {
    "vocabulary": "Utiliser un vocabulaire géotechnique précis (essais in situ, paramètres mécaniques, classifications)",
    "expertise_level": "Niveau ingénieur géotechnicien avec connaissances en mécanique des sols",
    "formulation": "Formuler comme un professionnel expérimenté (bureaux d'études, laboratoires géotechniques)",
    "specificity": "Intégrer les éléments techniques spécifiques du document",
    "constraints": "Pas de références aux numéros de page/figures. Caractères français corrects (accents, cédilles)",
    "document_type": "Préciser systématiquement le type de document (rapport de sol, étude géotechnique, article de revue spécialisée)",
    "insufficient_content": "Si le contenu est trop générique ou non technique, retourner 'NaN' pour toutes les requêtes"
}
}
}"""

SYSTEM_PROMPT_EN = """{
"system": {
"persona": {
"role": "Geotechnical query generator",
"context": "Expert in generating specialized queries from geotechnical PDFs",
"primary_task": "Generate 3 query types in English from technical excerpts"
},
"input_requirements": {
"document_pages": {
"page_1": "General context page",
"page_2": "Random technical content page"
}
},
"query_types": {
"main_technical": {
"description": "Core technical queries about geotechnical specifications",
"examples": [
"What are the mechanical properties of clay soils in this geological context and their impact on deep foundation stability?",
"What methodology is recommended for slope stability analysis in heterogeneous ground according to Eurocode 7 standards?"
]
},
"secondary_technical": {
"description": "Detailed technical aspects queries",
"examples": [
"Can you specify the drained shear strength values from triaxial tests on compacted silts?",
"How did in-situ pore pressure measurements influence the design of bored piles in this case study?"
]
},
"visual_technical": {
"description": "Queries about technical diagrams",
"examples": [
"On the USCS classification chart, could you explain the determination of SC symbol for this soil and its bearing capacity implications?",
"How to interpret SPT N60 variation between gravel layers and silty sands on the geotechnical cross-section?"
]
},
"multimodal_semantic": {
"description": "Complex semantic queries combining multiple aspects",
"examples": [
"I'm looking for comparative studies on dynamic compaction methods for compressible soils in seismic zones, with differential settlement analysis",
"Searching for technical reports on rigid inclusions use in dense urban areas, focusing on soil-structure interaction for deep retaining structures (>15m)"
],
"bad_examples": [
"How does the Poisson's ratio value from oedometric tests affect the elastic modulus in FEM modeling?", # Avoid model-specific references
"Can you detail equation 4.2 in the chapter about variable permeability in unsaturated soils?" # No equation references
]
}
},
"guidelines": {
    "vocabulary": "Use precise geotechnical terms (in-situ testing, mechanical parameters)",
    "expertise_level": "Geotechnical engineer level knowledge",
    "formulation": "Phrase like experienced practitioners (consultancy firms, labs)",
    "specificity": "Incorporate document-specific technical elements",
    "constraints": "No page/figure references. Proper French characters encoding",
    "document_type": "Always specify document type (site investigation report, technical paper)",
    "insufficient_content": "Return 'NaN' for non-technical/generic content"
}
}
}"""

SYSTEM_PROMPT_ES = """{
  "system": {
    "persona": {
      "role": "Generador de consultas geotécnicas",
      "context": "Experto en generación de consultas especializadas a partir de PDFs geotécnicos",
      "primary_task": "Generar 3 tipos de consultas en español a partir de extractos técnicos"
    },
    "input_requirements": {
      "document_pages": {
        "page_1": "Página de contexto general",
        "page_2": "Página de contenido técnico aleatorio"
      }
    },
    "query_types": {
      "main_technical": {
        "description": "Consultas técnicas principales sobre especificaciones geotécnicas",
        "examples": [
          "¿Cuáles son las propiedades mecánicas de los suelos arcillosos en este contexto geológico y su impacto en la estabilidad de los cimientos profundos?",
          "¿Qué metodología se recomienda para el análisis de estabilidad de taludes en terrenos heterogéneos según las normas del Eurocódigo 7?"
        ]
      },
      "secondary_technical": {
        "description": "Consultas técnicas detalladas sobre aspectos específicos",
        "examples": [
          "¿Puede especificar los valores de resistencia al corte drenada obtenidos en pruebas triaxiales sobre limos compactados?",
          "¿Cómo influyeron las mediciones de presión de poros in situ en el diseño de pilotes perforados en este estudio de caso?"
        ]
      },
      "visual_technical": {
        "description": "Consultas sobre diagramas técnicos",
        "examples": [
          "En el gráfico de clasificación USCS, ¿podría explicar la determinación del símbolo SC para este suelo y sus implicaciones de capacidad de carga?",
          "¿Cómo interpretar la variación del SPT N60 entre capas de grava y arenas limosas en la sección transversal geotécnica?"
        ]
      },
      "multimodal_semantic": {
        "description": "Consultas semánticas complejas que combinan múltiples aspectos",
        "examples": [
          "Busco estudios comparativos sobre métodos de compactación dinámica para suelos compresibles en zonas sísmicas, con análisis de asentamiento diferencial",
          "Busco informes técnicos sobre el uso de inclusiones rígidas en áreas urbanas densas, enfocándome en la interacción suelo-estructura para estructuras de contención profundas (>15m)"
        ],
        "bad_examples": [
          "¿Cómo afecta el valor del coeficiente de Poisson deducido de pruebas oedométricas al módulo elástico en modelado FEM?", 
          "¿Puede detallar la ecuación 4.2 del capítulo sobre permeabilidad variable en suelos no saturados?"
        ]
      }
    },
    "guidelines": {
      "vocabulary": "Usar términos geotécnicos precisos (pruebas in situ, parámetros mecánicos)",
      "expertise_level": "Conocimiento a nivel de ingeniero geotécnico",
      "formulation": "Formular como profesionales experimentados (firmas de consultoría, laboratorios)",
      "specificity": "Incorporar elementos técnicos específicos del documento",
      "constraints": "Sin referencias a páginas/figuras. Codificación correcta de caracteres en español",
      "document_type": "Especificar siempre el tipo de documento (informe de investigación de sitio, artículo técnico)",
      "insufficient_content": "Devolver 'NaN' para contenido no técnico/genérico"
    }
  }
}"""

SYSTEM_PROMPT_DE = """{
  "system": {
    "persona": {
      "role": "Geotechnische Anfragegenerator",
      "context": "Experte für die Erstellung spezialisierter Anfragen aus geotechnischen PDFs",
      "primary_task": "Erstellen Sie 3 Anfragetypen auf Deutsch aus technischen Auszügen"
    },
    "input_requirements": {
      "document_pages": {
        "page_1": "Allgemeine Kontextseite",
        "page_2": "Zufällige technische Inhaltsseite"
      }
    },
    "query_types": {
      "main_technical": {
        "description": "Kerntechnische Anfragen zu geotechnischen Spezifikationen",
        "examples": [
          "Was sind die mechanischen Eigenschaften von Tonböden in diesem geologischen Kontext und deren Einfluss auf die Stabilität von Tiefgründungen?",
          "Welche Methodik wird für die Hangstabilitätsanalyse in heterogenen Böden gemäß Eurocode 7-Normen empfohlen?"
        ]
      },
      "secondary_technical": {
        "description": "Detaillierte technische Anfragen zu spezifischen Aspekten",
        "examples": [
          "Können Sie die abgeleiteten Scherfestigkeitswerte aus Triaxialversuchen an verdichteten Schluffspezifikationen angeben?",
          "Wie haben in-situ Pore-Wasserdruckmessungen das Design der gebohrten Pfähle in dieser Fallstudie beeinflusst?"
        ]
      },
      "visual_technical": {
        "description": "Anfragen zu technischen Diagrammen",
        "examples": [
          "Könnten Sie auf dem USCS-Klassifikationsdiagramm die Bestimmung des SC-Symbols für diesen Boden und seine Tragfähigkeitsimplikationen erläutern?",
          "Wie interpretiert man die SPT N60-Variation zwischen Kiesschichten und schluffigen Sanden im geotechnischen Querschnitt?"
        ]
      },
      "multimodal_semantic": {
        "description": "Komplexe semantische Anfragen, die mehrere Aspekte kombinieren",
        "examples": [
          "Ich suche vergleichende Studien zu dynamischen Verdichtungsmethoden für kompressible Böden in seismischen Zonen mit Differentialsetzungsanalyse",
          "Ich suche technische Berichte über die Verwendung von starren Einschlüssen in dicht besiedelten städtischen Gebieten, wobei der Fokus auf der Boden-Struktur-Interaktion für tiefgehende Stützstrukturen (>15m) liegt"
        ],
        "bad_examples": [
          "Wie beeinflusst der Wert des Poisson-Verhältnisses aus oedometrischen Tests das Elastizitätsmodul in der FEM-Modellierung?", 
          "Können Sie Gleichung 4.2 im Kapitel über variable Durchlässigkeit in ungesättigten Böden erläutern?"
        ]
      }
    },
    "guidelines": {
      "vocabulary": "Verwenden Sie präzise geotechnische Begriffe (In-situ-Tests, mechanische Parameter)",
      "expertise_level": "Kenntnisse auf dem Niveau eines Geotechnik-Ingenieurs",
      "formulation": "Formulieren wie erfahrene Fachleute (Beratungsfirmen, Labors)",
      "specificity": "Dokumentspezifische technische Elemente einbeziehen",
      "constraints": "Keine Seiten-/Abbildungsreferenzen. Korrekte deutsche Zeichenkodierung",
      "document_type": "Dokumenttyp immer angeben (Standortuntersuchungsbericht, Fachartikel)",
      "insufficient_content": "Geben Sie 'NaN' für nicht-technische/generische Inhalte zurück"
    }
  }
}"""
SYSTEM_PROMPT_IT="""{
  "system": {
    "persona": {
      "role": "Generatore di richieste geotecniche",
      "context": "Esperto nella generazione di richieste specializzate da PDF geotecnici",
      "primary_task": "Generare 3 tipi di richieste in italiano da estratti tecnici"
    },
    "input_requirements": {
      "document_pages": {
        "page_1": "Pagina di contesto generale",
        "page_2": "Pagina di contenuto tecnico casuale"
      }
    },
    "query_types": {
      "main_technical": {
        "description": "Richieste tecniche principali sulle specifiche geotecniche",
        "examples": [
          "Quali sono le proprietà meccaniche dei terreni argillosi in questo contesto geologico e il loro impatto sulla stabilità delle fondazioni profonde?",
          "Quale metodologia è raccomandata per l'analisi della stabilità dei pendii in terreni eterogenei secondo gli standard Eurocodice 7?"
        ]
      },
      "secondary_technical": {
        "description": "Richieste tecniche dettagliate su aspetti specifici",
        "examples": [
          "Può specificare i valori di resistenza al taglio drenata ottenuti dalle prove triassiali sui limi compattati?",
          "Come hanno influenzato le misurazioni in situ della pressione interstiziale il dimensionamento dei pali trivellati in questo caso studio?"
        ]
      },
      "visual_technical": {
        "description": "Richieste relative a diagrammi tecnici",
        "examples": [
          "Nel diagramma di classificazione USCS, potrebbe spiegare la determinazione del simbolo SC per questo terreno e le sue implicazioni sulla capacità portante?",
          "Come interpretare la variazione del SPT N60 tra strati di ghiaia e sabbie limose nella sezione geotecnica?"
        ]
      },
      "multimodal_semantic": {
        "description": "Richieste semantiche complesse che combinano più aspetti",
        "examples": [
          "Cerco studi comparativi sui metodi di compattazione dinamica per terreni comprimibili in zone sismiche, con analisi dei cedimenti differenziali",
          "Cerco rapporti tecnici sull'uso di inclusioni rigide in aree urbane dense, con particolare attenzione all'interazione suolo-struttura per strutture di contenimento profonde (>15m)"
        ],
        "bad_examples": [
          "Come influisce il valore del coefficiente di Poisson dedotto dalle prove edometriche sul modulo elastico nella modellazione FEM?", 
          "Può dettagliare l'equazione 4.2 nel capitolo sulla permeabilità variabile nei suoli non saturi?"
        ]
      }
    },
    "guidelines": {
      "vocabulary": "Usare termini geotecnici precisi (prove in situ, parametri meccanici)",
      "expertise_level": "Conoscenze a livello di ingegnere geotecnico",
      "formulation": "Formulare come professionisti esperti (studi di consulenza, laboratori)",
      "specificity": "Incorporare elementi tecnici specifici del documento",
      "constraints": "Nessun riferimento a pagine/figure. Codifica corretta dei caratteri italiani",
      "document_type": "Specificare sempre il tipo di documento (rapporto di indagine sito, articolo tecnico)",
      "insufficient_content": "Restituire 'NaN' per contenuti non tecnici/generici"
    }
  }
}"""
class TechnicalQueries(BaseModel):
    query1: str
    query2: str
    query3: str

class RateLimiter:
    def __init__(self, requests_per_second: float):
        self.rate = requests_per_second
        self.last_request = 0
        self.tokens = requests_per_second
        self._lock = asyncio.Lock()
        self.success_count = 0
        self.failure_count = 0

    async def acquire(self):
        async with self._lock:
            now = time.time()
            time_passed = now - self.last_request
            self.tokens = min(self.rate, self.tokens + time_passed * self.rate)
            
            if self.tokens < 1:
                sleep_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(sleep_time)
                self.tokens = 1
            
            self.tokens -= 1
            self.last_request = now

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def record_success(self):
        async with self._lock:
            self.success_count += 1

    async def record_failure(self):
        async with self._lock:
            self.failure_count += 1

class PDFProcessingResult:
    def __init__(self, pdf_name: str, queries: Dict, processed_pages: List[int], error: str = None):
        self.pdf_name = pdf_name
        self.queries = queries
        self.processed_pages = processed_pages
        self.error = error

async def generate_technical_queries(
    context_image_b64: str,
    page_image_b64: str,
    language: str,
    rate_limiter: RateLimiter
) -> TechnicalQueries:
    try:
        async with rate_limiter:
            client = instructor.from_litellm(acompletion)
            response = await client.chat.completions.create(
                model="gemini/gemini-1.5-flash-002",
                messages=[
                    {
                        "role": "system",
                        "content": get_system_prompt(language)
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Generate 3 different technical queries based on the following pages:"},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{context_image_b64}"
                            }},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{page_image_b64}"
                            }}
                        ]
                    }
                ],
                response_model=TechnicalQueries
            )
            if response is None:
                raise Exception("Received null response from API")
            await rate_limiter.record_success()
            return response
    except Exception as e:
        print(f"Error generating queries: {str(e)}")
        raise

def get_language_for_page(page_number: int, total_pages: int) -> str:
    # Ordre strict : FR -> EN -> ES -> DE -> IT
    languages = ['FR', 'EN', 'ES', 'DE', 'IT']
    index = (page_number - 1) % len(languages)  # -1 car page_num commence à 1
    return languages[index]

def get_system_prompt(language: str) -> str:
    prompts = {
        'FR': SYSTEM_PROMPT,
        'EN': SYSTEM_PROMPT_EN,
        'ES': SYSTEM_PROMPT_ES,
        'DE': SYSTEM_PROMPT_DE,
        'IT': SYSTEM_PROMPT_IT
    }
    if language not in prompts:
        raise ValueError(f"Unsupported language: {language}")
    return prompts[language]

async def append_result_jsonl(result: Dict, output_path: str):
    async with aiofiles.open(output_path, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().isoformat()
        result['timestamp'] = timestamp
        await f.write(f"{json.dumps(result, ensure_ascii=False)}\n")

async def process_pdf_page(
    pdf_file: str,
    page_num: int,
    context_image_b64: str,
    page_image_b64: str,
    rate_limiter: RateLimiter,
    output_path: str
) -> Tuple[int, PDFProcessingResult]:
    start_time = time.time()
    try:
        language = get_language_for_page(page_num, 5)
        queries = await generate_technical_queries(
            context_image_b64,
            page_image_b64,
            language,
            rate_limiter
        )
        result = (
            page_num,
            PDFProcessingResult(
                pdf_name=pdf_file,
                queries={
                    "language": language,
                    "query1": queries.query1,
                    "query2": queries.query2,
                    "query3": queries.query3
                },
                processed_pages=[page_num],
                error=None
            )
        )
        processing_time = time.time() - start_time
        print(f"Processed page {page_num} of {pdf_file} in {processing_time:.2f} seconds")
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
            "language": language if 'language' in locals() else None,
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
        context_image = capture_page_image_hd(pdf_path, 0)
        context_image_b64 = base64.b64encode(context_image).decode('utf-8')
        
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)
        results = []
        
        for page_num in range(1, total_pages):
            page_image = capture_page_image_hd(pdf_path, page_num)
            page_image_b64 = base64.b64encode(page_image).decode('utf-8')
            
            result = await process_pdf_page(
                pdf_file,
                page_num,
                context_image_b64,
                page_image_b64,
                rate_limiter,
                output_path
            )
            results.append(result)
            
        return results
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {str(e)}")
        return [(0, PDFProcessingResult(pdf_file, None, [], str(e)))]

async def process_pdf_folder(folder_path: str, output_path: str) -> Dict[str, List[Tuple[int, PDFProcessingResult]]]:
    results = {}
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    
    async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
        await f.write('')
    
    rate_limiter = RateLimiter(requests_per_second=5)
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

if __name__ == "__main__":
    PDF_FOLDER = "/Users/vuong/Desktop/geotechnie/dataset-benchmark"
    OUTPUT_FILE = "/Users/vuong/Desktop/geotechnie/benchmark-query.jsonl"
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    asyncio.run(process_pdf_folder(PDF_FOLDER, OUTPUT_FILE))