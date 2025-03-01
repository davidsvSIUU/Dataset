import json
import os

def clean_jsonl(input_file: str, output_file: str = None):
    """
    Supprime les lignes contenant 'NaN' dans les queries d'un fichier JSONL.
    
    Args:
        input_file (str): Chemin du fichier JSONL d'entrée
        output_file (str): Chemin du fichier JSONL de sortie. Si None, écrase le fichier d'entrée
    """
    if output_file is None:
        output_file = input_file + '.temp'
    
    # Lire et filtrer les lignes
    valid_lines = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # Vérifie si l'une des valeurs dans queries est 'NaN'
                if 'queries' in data and not any(v == "NaN" for v in data['queries'].values()):
                    valid_lines.append(line)
            except json.JSONDecodeError:
                print(f"Erreur de décodage JSON pour la ligne: {line}")
                continue

    # Écrire les lignes valides
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(valid_lines)
    
    # Si on écrase le fichier d'entrée
    if output_file == input_file + '.temp':
        os.replace(output_file, input_file)
        print(f"Fichier nettoyé: {input_file}")
    else:
        print(f"Nouveau fichier créé: {output_file}")
    
    print(f"Nombre de lignes conservées: {len(valid_lines)}")
    
# Exemple d'utilisation
if __name__ == "__main__":
    # Remplacez par le chemin de votre fichier
    input_file = "C:\\Users\\david\\Desktop\\dataset\\EasyJet-query.jsonl"
    
    # Option 1: Écraser le fichier d'origine
    clean_jsonl(input_file)
    
    # Option 2: Créer un nouveau fichier
    # clean_jsonl(input_file, "/Users/vuong/Desktop/geotechnie/benchmark-query-clean.jsonl")