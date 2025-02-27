import os
from PyPDF2 import PdfReader

def compter_pages_pdfs(chemin_dossier):
    # Vérifier si le chemin existe
    if not os.path.exists(chemin_dossier):
        print(f"Le dossier {chemin_dossier} n'existe pas.")
        return
    
    # Initialiser le compteur total de pages
    total_pages = 0
    
    # Parcourir tous les fichiers du dossier
    for fichier in os.listdir(chemin_dossier):
        # Vérifier si le fichier est un PDF
        if fichier.lower().endswith('.pdf'):
            chemin_complet = os.path.join(chemin_dossier, fichier)
            try:
                # Ouvrir le PDF
                with open(chemin_complet, 'rb') as f:
                    pdf = PdfReader(f)
                    nb_pages = len(pdf.pages)
                    total_pages += nb_pages
                    print(f"{fichier}: {nb_pages} pages")
            except Exception as e:
                print(f"Erreur lors de la lecture de {fichier}: {str(e)}")
    
    print(f"\nNombre total de pages: {total_pages}")

# Utilisation avec votre chemin Mac
if __name__ == "__main__":
    chemin_dossier = "/Users/vuong/Desktop/geotechnie/dataset-benchmark-v2"
    compter_pages_pdfs(chemin_dossier)