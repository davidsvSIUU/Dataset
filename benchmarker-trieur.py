import json

def sort_queries(input_file, output_file):
    # Lire le fichier JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Parser chaque ligne en JSON et créer une liste d'entrées
    entries = []
    for line in lines:
        try:
            entry = json.loads(line.strip())
            entries.append(entry)
        except json.JSONDecodeError:
            continue

    # Fonction pour déterminer la priorité de tri
    def sort_key(entry):
        reference = entry.get('queries', {}).get('reference', '')
        
        # Vérifie si l'entrée est vide ou NaN
        if reference in ['""', 'NaN', '']:
            return (2, '')  # Priorité basse pour les entrées vides
        
        # Vérifie si l'entrée commence par "Relevant"
        if reference.strip().startswith('Relevant'):
            return (0, reference)  # Priorité haute pour les entrées "Relevant"
        
        return (1, reference)  # Priorité moyenne pour les autres entrées

    # Trier les entrées
    sorted_entries = sorted(entries, key=sort_key)

    # Écrire le résultat dans un nouveau fichier
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in sorted_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

# Utilisation
input_file = '/Users/vuong/Desktop/geotechnie/benchmark-query.jsonl'  # Remplacer par le nom de votre fichier d'entrée
output_file = '/Users/vuong/Desktop/geotechnie/benchmark-query-trie.jsonl'  # Remplacer par le nom de fichier souhaité pour la sortie

sort_queries(input_file, output_file)