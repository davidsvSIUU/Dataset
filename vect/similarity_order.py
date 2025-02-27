import csv

# Fichiers d'entrée et de sortie
input_file = '/Users/vuong/Desktop/vision-benchmark-maker/vect/similarity_results.csv'
output_file = '/Users/vuong/Desktop/vision-benchmark-maker/vect/filtres_similarity_results.csv'

with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
     open(output_file, mode='w', newline='', encoding='utf-8') as outfile:

    # Créer les lecteur et écrivain CSV
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # Lire et écrire l'en-tête
    header = next(reader)
    writer.writerow(header)
    
    # Filtrer les lignes avec score >= 0.9
    for row in reader:
        try:
            score = float(row[2])
            if score >= 0.76:
                writer.writerow(row)
        except ValueError:
            # Gérer les erreurs de conversion numérique
            print(f"Erreur de format pour le score dans la ligne : {row}")

print("Filtrage terminé. Les résultats sont dans", output_file)