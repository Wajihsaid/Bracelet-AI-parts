import pandas as pd
import os

# Le chemin du dossier contenant tes fichiers
folder_path = "filtered-dataset - Copy"

# Liste pour stocker les DataFrames
dataframes = []

# Noms des colonnes
columns = [
    "TimeStamp", "Sample No", "X-Axis", "Y-Axis", "Z-Axis", "Sensor Type", "Sensor ID"
]

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        
        # Lire le CSV
        df = pd.read_csv(file_path, header=None, sep=None, engine="python")
        
        # Nettoyer les lignes invalides
        df = df.dropna()
        if df.shape[1] > 7:
            df = df.iloc[:, :7]
        df.columns = columns
        
        # 🔍 Extraire le label (par exemple ici c’est le 4ème élément après split par "_")
        parts = filename.split("_")
        label = parts[5] if len(parts) > 4 else "Unknown"
        
        # Ajouter une colonne 'Label'
        df["Label"] = label
        
        dataframes.append(df)

# Fusionner tous les fichiers en un seul dataset
final_df = pd.concat(dataframes, ignore_index=True)

# 💾 Enregistrer le DataFrame fusionné dans un fichier CSV
final_df.to_csv("merged_dataset-Copy.csv", index=False)

print("✅ Dataset construit avec forme :", final_df.shape)
print(final_df["Label"].value_counts())
print("📁 Fichier exporté sous : merged_dataset.csv")
