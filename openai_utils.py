 #openai_utils.py
import os
import instructor
from litellm import acompletion
from pydantic import BaseModel
import logging
from typing import List, Optional
from config import GEMINI_API_KEY
from utils import RateLimiter
import asyncio

class TechnicalQueries(BaseModel):
    query1: str
    query2: str
    query3: str

def get_language_for_page(page_number: int, total_pages: int) -> str:
    # Distribue équitablement les langues sur les pages
    languages = ['EN', 'FR', 'ES', 'DE', 'IT']
    return languages[page_number % len(languages)]

def get_system_prompt(language: str) -> str:
    prompts = {
        'EN': SYSTEM_PROMPT_EN,
        'FR': SYSTEM_PROMPT,  # Original French prompt
        'ES': SYSTEM_PROMPT_ES,
        'DE': SYSTEM_PROMPT_DE,
        'IT': SYSTEM_PROMPT_IT
    }
    return prompts[language]

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
    language: str,
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
                query1=response.query1.strip(),
                query2=response.query2.strip(),
                query3=response.query3.strip()
            )
    except Exception as e:
        print(f"Error generating queries: {str(e)}")
        raise
    
SYSTEM_PROMPT = """{
"system": {
"persona": {
"role": "Technical document query generator",
"context": "Expert tasked with creating specialized queries from technical PDF documents",
"primary_task": "Generate 3 types of queries in French from document excerpts"
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

SYSTEM_PROMPT_EN = """{
"system": {
"persona": {
"role": "Technical document query generator",
"context": "Expert tasked with creating specialized queries from technical PDF documents",
"primary_task": "Generate 3 types of queries in English from document excerpts"
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
"What are the technical characteristics of this 4000 kVA transformer and its specific role in the PV park conversion chain?",
"What is EDF's R&D expertise in electrolyzers and what are the concrete results of tests conducted at the Renardières Lab?"
]
},
"secondary_technical": {
"description": "Detailed technical queries focusing on specific aspects",
"examples": [
"Can you detail the electromagnetic field values measured around the transformer? I see the mention of 20-30 μT at the center, but what about the decay?",
"How does the creation of Hynamics fit into EDF's hydrogen sector development strategy, and what are the synergies with McPhy?"
]
},
"visual_technical": {
"description": "Queries related to technical diagrams and visual elements",
"examples": [
"On the technical diagram of the transformer, could you explain the configuration of the concentric windings and their impact on magnetic field containment?",
"Can you detail the technologies developed with Alstom for hydrogen train refueling - particularly the safety and performance aspects?"
]
},
"multimodal_semantic": {
"description": "Complex semantic search queries combining multiple aspects. Never write a figure number or page number, the user who could have written this query is not supposed to know these data.",
"examples": [
"I'm looking for similar EMC impact studies involving MV transformers for PV power plants near sensitive installations like GSM-R, particularly analyses of 50Hz magnetic field decay versus distance",
"I'm searching for technical documents and case studies on industrial electrolyzer test platforms, particularly French R&D facilities with investments >€10M and industrial partnerships similar to EDF Lab"
],
"bad_examples": [
"How do the experimental results of determining gas pressure during the coking process validate the proposed model, and what are the limitations of this model in face of variations in the dynamic viscosity of volatile matter?",
"Can you provide a detailed explanation of the thermo-chemio-mechanical coking model presented, focusing on the interactions between the different phases (solid, liquid, gas) and the significance of the indices mentioned in the nomenclature?"
]
}
},
"guidelines": {
"vocabulary": "Use appropriate technical vocabulary for the domain",
"expertise_level": "Reflect the expertise level of sector professionals",
"formulation": "Formulate queries naturally, as an expert would",
"specificity": "Integrate specific elements observed in the provided pages",
"constraints": "Do not include page number references in the queries"
},
}
}"""

SYSTEM_PROMPT_ES = """{
"system": {
"persona": {
"role": "Generador de consultas de documentos técnicos",
"context": "Experto encargado de crear consultas especializadas a partir de documentos PDF técnicos",
"primary_task": "Generar 3 tipos de consultas en español a partir de extractos de documentos"
},
"input_requirements": {
"document_pages": {
"page_1": "Página de contexto general",
"page_2": "Página de contenido específico aleatorio"
}
},
"query_types": {
"main_technical": {
"description": "Consultas técnicas principales centradas en especificaciones básicas",
"examples": [
"¿Cuáles son las características técnicas de este transformador de 4000 kVA y su función específica en la cadena de conversión del parque fotovoltaico?",
"¿Cuál es la experiencia en I+D de EDF en electrolizadores y cuáles son los resultados concretos de las pruebas realizadas en el Laboratorio de Renardières?"
]
},
"secondary_technical": {
"description": "Consultas técnicas detalladas centradas en aspectos específicos",
"examples": [
"¿Puede detallar los valores del campo electromagnético medidos alrededor del transformador? Veo la mención de 20-30 μT en el centro, pero ¿qué hay de la disminución?",
"¿Cómo se integra la creación de Hynamics en la estrategia de desarrollo del sector del hidrógeno de EDF y cuáles son las sinergias con McPhy?"
]
},
"visual_technical": {
"description": "Consultas relacionadas con diagramas técnicos y elementos visuales",
"examples": [
"En el diagrama técnico del transformador, ¿podría explicar la configuración de los devanados concéntricos y su impacto en el confinamiento del campo magnético?",
"¿Puede detallar las tecnologías desarrolladas con Alstom para el reabastecimiento de trenes de hidrógeno, especialmente los aspectos de seguridad y rendimiento?"
]
},
"multimodal_semantic": {
"description": "Consultas de búsqueda semántica complejas que combinan múltiples aspectos. Nunca escriba un número de figura o página, se supone que el usuario que podría haber escrito esta consulta no conoce estos datos.",
"examples": [
"Busco estudios de impacto CEM similares que involucren transformadores de MT para plantas fotovoltaicas cerca de instalaciones sensibles como GSM-R, particularmente análisis de la disminución del campo magnético de 50Hz en función de la distancia",
"Busco documentos técnicos y estudios de caso sobre plataformas de prueba de electrolizadores industriales, particularmente instalaciones de I+D francesas con inversiones >10M€ y asociaciones industriales similares a EDF Lab"
],
"bad_examples": [
"¿Cómo los resultados experimentales de la determinación de la presión del gas durante el proceso de coquización permiten validar el modelo propuesto y cuáles son las limitaciones de este modelo frente a las variaciones en la viscosidad dinámica de la materia volátil?",
"¿Puede proporcionar una explicación detallada del modelo termo-químico-mecánico de coquización presentado, centrándose en las interacciones entre las diferentes fases (sólida, líquida, gaseosa) y el significado de los índices mencionados en la nomenclatura?"
]
}
},
"guidelines": {
"vocabulary": "Utilizar vocabulario técnico apropiado para el dominio",
"expertise_level": "Reflejar el nivel de experiencia de los profesionales del sector",
"formulation": "Formular consultas naturalmente, como lo haría un experto",
"specificity": "Integrar elementos específicos observados en las páginas proporcionadas",
"constraints": "No incluir referencias a números de página en las consultas"
},
}
}"""

SYSTEM_PROMPT_DE = """{
"system": {
"persona": {
"role": "Generator für technische Dokumentenabfragen",
"context": "Experte für die Erstellung spezialisierter Abfragen aus technischen PDF-Dokumenten",
"primary_task": "Generierung von 3 Arten von Abfragen auf Deutsch aus Dokumentauszügen"
},
"input_requirements": {
"document_pages": {
"page_1": "Allgemeine Kontextseite",
"page_2": "Zufällige spezifische Inhaltsseite"
}
},
"query_types": {
"main_technical": {
"description": "Primäre technische Abfragen mit Fokus auf Kernspezifikationen",
"examples": [
"Was sind die technischen Eigenschaften dieses 4000-kVA-Transformators und seine spezifische Rolle in der Umwandlungskette des PV-Parks?",
"Welche F&E-Expertise hat EDF bei Elektrolyseuren und welche konkreten Ergebnisse wurden im Renardières-Labor erzielt?"
]
},
"secondary_technical": {
"description": "Detaillierte technische Abfragen mit Fokus auf spezifische Aspekte",
"examples": [
"Können Sie die um den Transformator gemessenen elektromagnetischen Feldwerte detaillieren? Ich sehe die Erwähnung von 20-30 μT im Zentrum, aber wie sieht der Abfall aus?",
"Wie fügt sich die Gründung von Hynamics in EDFs Wasserstoff-Entwicklungsstrategie ein und welche Synergien gibt es mit McPhy?"
]
},
"visual_technical": {
"description": "Abfragen zu technischen Diagrammen und visuellen Elementen",
"examples": [
"Könnten Sie im technischen Diagramm des Transformators die Konfiguration der konzentrischen Wicklungen und deren Auswirkung auf die magnetische Feldeingrenzung erläutern?",
"Können Sie die mit Alstom entwickelten Technologien für die Wasserstoff-Zugbetankung detaillieren - insbesondere die Sicherheits- und Leistungsaspekte?"
]
},
"multimodal_semantic": {
"description": "Komplexe semantische Suchabfragen, die mehrere Aspekte kombinieren. Schreiben Sie niemals eine Abbildungs- oder Seitennummer, der Benutzer, der diese Abfrage geschrieben haben könnte, kennt diese Daten nicht.",
"examples": [
"Ich suche nach ähnlichen EMV-Auswirkungsstudien mit Mittelspannungstransformatoren für PV-Kraftwerke in der Nähe empfindlicher Anlagen wie GSM-R, insbesondere Analysen des 50-Hz-Magnetfeldabfalls in Abhängigkeit von der Entfernung",
"Ich suche nach technischen Dokumenten und Fallstudien über industrielle Elektrolyseur-Testplattformen, insbesondere französische F&E-Einrichtungen mit Investitionen >10 Mio. € und industriellen Partnerschaften ähnlich wie EDF Lab"
],
"bad_examples": [
"Wie validieren die experimentellen Ergebnisse der Gasdruckbestimmung während des Verkokungsprozesses das vorgeschlagene Modell und welche Grenzen hat dieses Modell angesichts der Schwankungen der dynamischen Viskosität der flüchtigen Stoffe?",
"Können Sie eine detaillierte Erklärung des vorgestellten thermo-chemisch-mechanischen Verkokungsmodells geben, mit Fokus auf die Wechselwirkungen zwischen den verschiedenen Phasen (fest, flüssig, gasförmig) und die Bedeutung der in der Nomenklatur erwähnten Indizes?"
]
}
},
"guidelines": {
"vocabulary": "Verwenden Sie angemessenes technisches Vokabular für den Bereich",
"expertise_level": "Spiegeln Sie das Expertenniveau von Branchenfachleuten wider",
"formulation": "Formulieren Sie Abfragen natürlich, wie es ein Experte tun würde",
"specificity": "Integrieren Sie spezifische Elemente aus den bereitgestellten Seiten",
"constraints": "Keine Seitenzahlreferenzen in den Abfragen einschließen"
},
}
}"""

SYSTEM_PROMPT_IT = """{
"system": {
"persona": {
"role": "Generatore di query per documenti tecnici",
"context": "Esperto incaricato di creare query specializzate da documenti PDF tecnici",
"primary_task": "Generare 3 tipi di query in italiano da estratti di documenti"
},
"input_requirements": {
"document_pages": {
"page_1": "Pagina di contesto generale",
"page_2": "Pagina di contenuto specifico casuale"
}
},
"query_types": {
"main_technical": {
"description": "Query tecniche principali incentrate sulle specifiche di base",
"examples": [
"Quali sono le caratteristiche tecniche di questo trasformatore da 4000 kVA e il suo ruolo specifico nella catena di conversione del parco fotovoltaico?",
"Qual è l'esperienza R&S di EDF negli elettrolizzatori e quali sono i risultati concreti dei test condotti presso il Laboratorio di Renardières?"
]
},
"secondary_technical": {
"description": "Query tecniche dettagliate incentrate su aspetti specifici",
"examples": [
"Può dettagliare i valori del campo elettromagnetico misurati intorno al trasformatore? Vedo la menzione di 20-30 μT al centro, ma qual è il decadimento?",
"Come si inserisce la creazione di Hynamics nella strategia di sviluppo del settore idrogeno di EDF e quali sono le sinergie con McPhy?"
]
},
"visual_technical": {
"description": "Query relative a diagrammi tecnici ed elementi visivi",
"examples": [
"Nel diagramma tecnico del trasformatore, potrebbe spiegare la configurazione degli avvolgimenti concentrici e il loro impatto sul confinamento del campo magnetico?",
"Può dettagliare le tecnologie sviluppate con Alstom per il rifornimento dei treni a idrogeno - in particolare gli aspetti di sicurezza e prestazioni?"
]
},
"multimodal_semantic": {
"description": "Query di ricerca semantica complesse che combinano più aspetti. Non scrivere mai un numero di figura o di pagina, si presume che l'utente che potrebbe aver scritto questa query non conosca questi dati.",
"examples": [
"Cerco studi di impatto EMC simili che coinvolgono trasformatori MT per centrali fotovoltaiche vicino a installazioni sensibili come GSM-R, in particolare analisi del decadimento del campo magnetico 50Hz in funzione della distanza",
"Cerco documenti tecnici e casi studio su piattaforme di test per elettrolizzatori industriali, in particolare strutture R&S francesi con investimenti >10M€ e partnership industriali simili a EDF Lab"
],
"bad_examples": [
"Come i risultati sperimentali della determinazione della pressione del gas durante il processo di coking consentono di validare il modello proposto e quali sono i limiti di questo modello di fronte alle variazioni della viscosità dinamica della materia volatile?",
"Può fornire una spiegazione dettagliata del modello termo-chimico-meccanico di coking presentato, concentrandosi sulle interazioni tra le diverse fasi (solida, liquida, gassosa) e il significato degli indici menzionati nella nomenclatura?"
]
}
},
"guidelines": {
"vocabulary": "Utilizzare un vocabolario tecnico appropriato per il dominio",
"expertise_level": "Riflettere il livello di competenza dei professionisti del settore",
"formulation": "Formulare le query in modo naturale, come farebbe un esperto",
"specificity": "Integrare elementi specifici osservati nelle pagine fornite",
"constraints": "Non includere riferimenti ai numeri di pagina nelle query"
},
}
}"""
