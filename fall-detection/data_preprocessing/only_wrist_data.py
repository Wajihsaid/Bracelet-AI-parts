import pandas as pd

# === Configuration ===
# Remplace par le nom réel de ton fichier
input_file = 'merged_dataset-Copy.csv'     # <-- fichier d'origine
output_file = 'wrist_data-Copy.csv'      # <-- fichier filtré (poignet seulement)
sensor_id_wrist = 3

# === Chargement du dataset ===
df = pd.read_csv(input_file)
classes_names = df['Label'].unique()
print(classes_names)
# Nettoyer la colonne "Sensor ID"
df = df[df["Sensor ID"] != "Sensor ID"]  # retirer les lignes incorrectes

# Convertir toutes les valeurs en chaîne après les avoir mises en int
df["Sensor ID"] = df["Sensor ID"].astype(float).astype(int).astype(str)

# Filtrer les lignes avec le capteur du poignet (ID = "3")
df_wrist = df[df["Sensor ID"] == "3"]

df_wrist.to_csv(output_file, index=False)

print(f"✅ Fichier filtré sauvegardé sous : {output_file}")
print(f"Nombre de lignes conservées : {len(df_wrist)}")
