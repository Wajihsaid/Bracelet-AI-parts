import pandas as pd
import glob
import os
from pathlib import Path

# Configuration
input_folder = Path("UMAFALL_Dataset")  # Dossier source
output_folder = Path("new")  # Dossier de sortie

# Noms et largeurs des colonnes
COLUMNS = {
    "TimeStamp": 10,
    "Sample No": 10, 
    "X-Axis": 12,
    "Y-Axis": 12,
    "Z-Axis": 12,
    "Sensor Type": 12,
    "Sensor ID": 10
}

def format_value(value, width, is_float=False):
    """Formate une valeur avec l'espacement approprié"""
    if is_float:
        return f"{float(value):<{width}.7f}".rstrip('0').rstrip('.')
    return f"{value:<{width}}"

def process_file(input_path, output_path):
    try:
        # Lire le fichier
        df = pd.read_csv(input_path, sep=';', comment='%', header=None)
        
        # Vérifier le nombre de colonnes
        if len(df.columns) != len(COLUMNS):
            print(f"Erreur: {input_path.name} a {len(df.columns)} colonnes")
            return
            
        # Appliquer les noms de colonnes
        df.columns = COLUMNS.keys()
        
        # Créer le contenu formaté
        content = []
        
        # Ligne d'en-tête
        header = "| " + " | ".join(
            f"{name:^{width}}" for name, width in COLUMNS.items()
        ) + " |"
        content.append(header)
        
        # Ligne de séparation
        separator = "|-" + "-|-".join(
            "-" * width for width in COLUMNS.values()
        ) + "-|"
        content.append(separator)
        
        # Lignes de données
        for _, row in df.iterrows():
            line = "| " + " | ".join([
                format_value(row['TimeStamp'], COLUMNS['TimeStamp']),
                format_value(row['Sample No'], COLUMNS['Sample No']),
                format_value(row['X-Axis'], COLUMNS['X-Axis'], is_float=True),
                format_value(row['Y-Axis'], COLUMNS['Y-Axis'], is_float=True),
                format_value(row['Z-Axis'], COLUMNS['Z-Axis'], is_float=True),
                format_value(row['Sensor Type'], COLUMNS['Sensor Type']),
                format_value(row['Sensor ID'], COLUMNS['Sensor ID'])
            ]) + " |"
            content.append(line)
        
        # Sauvegarder
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
            
        print(f"✓ {input_path.name} traité")
        
    except Exception as e:
        print(f"✗ Erreur sur {input_path.name}: {str(e)}")

# Traiter tous les fichiers
output_folder.mkdir(parents=True, exist_ok=True)
for input_path in input_folder.glob("*.csv"):
    output_path = output_folder / input_path.name
    process_file(input_path, output_path)

print(f"\nTraitement terminé. Fichiers dans : {output_folder}")