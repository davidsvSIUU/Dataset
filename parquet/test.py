import pandas as pd

def search_multiple_pdfs_in_corpus(corpus_path, pdf_list):
    try:
        # Lecture du fichier corpus.parquet
        df = pd.read_parquet(corpus_path)
        
        # Extraction des noms de fichiers uniques de votre liste
        # On enlève les numéros après le .pdf_ pour avoir juste les noms de fichiers
        pdf_names = set(pdf.split('.pdf_')[0] + '.pdf' for pdf in pdf_list)
        
        print(f"\nRecherche de {len(pdf_names)} fichiers PDF uniques:")
        for pdf_name in sorted(pdf_names):
            print(f"\nRecherche de '{pdf_name}':")
            found = False
            for column in df.columns:
                if df[column].astype(str).str.contains(pdf_name, case=False).any():
                    print(f"✓ Trouvé dans la colonne '{column}'")
                    found = True
            
            if not found:
                print("✗ Non trouvé dans le corpus")
            
    except Exception as e:
        print(f"Erreur lors de la recherche: {str(e)}")

# Chemin vers votre fichier corpus.parquet
corpus_path = "/Users/vuong/Desktop/vision-benchmark-maker/parquet/corpus.parquet"

# Liste de tous les PDFs à rechercher
pdfs_to_search = [
    'annualreport22-23rev.pdf_96',
    'NYSE_PBR_2019.pdf_77',
    'Sustainability Report 2023.pdf_224',
    'annualreport2022_New.pdf_77',
    'Sustainability Report 2023.pdf_92',
    '85e626beab274137ac1e8068de282d72.pdf_287',
    'tc-investorday-2023-presentation.pdf_16',
    'NetZeroemission.pdf_62',
    'Res_Exec_Ing_Rel_Sust_V03.pdf_10',
    'tce-corporate-profile.pdf_19',
    'annualreport2022_New.pdf_255',
    '47b58253c2504b4cb45f3c170394dfe7.pdf_19',
    'NYSE_PBR_2019.pdf_352',
    'climate-change-resilience-report.pdf_84',
    'investor-presentation_july2023.pdf_44',
    'Sempra-Sustainability-Report-Full-2021.pdf_48',
    'Climate Change Supplement.pdf_79',
    '20234Q_Investors Meeting.pdf_37',
    'ecopetrol-rigs-2021-eng.pdf_187',
    'chiffres-cles-de-lenergie-2022-signets.pdf'
]

search_multiple_pdfs_in_corpus(corpus_path, pdfs_to_search)