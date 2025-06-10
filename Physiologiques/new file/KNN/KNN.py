import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, log_loss
import joblib

# ========== CONFIGURATION ==========
current_dir = os.path.dirname(__file__)
DATABASE_FOLDER = os.path.join(current_dir, "patient_data(BDD)")
FEATURES = ['Fréquence Cardiaque (bpm)', 'Saturation en Oxygène (%)', 'Température (°C)']

# ========== LOAD DATA ==========
if not os.path.exists(DATABASE_FOLDER):
    print(f"\n❌ Erreur : Le dossier '{DATABASE_FOLDER}' est introuvable.")
    exit()

print(f"\n⚡ Chargement depuis : {DATABASE_FOLDER}")
try:
    # Exclude any temporary Excel files (e.g., those starting with ~$
    fichiers = [f for f in os.listdir(DATABASE_FOLDER) if f.endswith('.xlsx') and not f.startswith('~$')]
    if not fichiers:
        print("❌ Aucun fichier .xlsx trouvé dans le dossier")
        exit()

    # Load and concatenate data
    data = pd.concat([pd.read_excel(os.path.join(DATABASE_FOLDER, f)) for f in fichiers],
                     ignore_index=True)
    print(f"✅ {len(fichiers)} fichiers chargés | {len(data)} lignes")
except Exception as e:
    print(f"❌ Erreur lors du chargement : {str(e)}")
    exit()

# ========== CLEAN DATA ==========
print("\n🧹 Nettoyage des données...")
print("Valeurs manquantes:\n", data.isnull().sum())
data = data.dropna().drop_duplicates()

# ========== VISUALISATION DE CLASSES ==========
print("\n🔎 Vérification de l'équilibre des classes:")
class_balance = pd.crosstab(index=data['État'], columns='% de patients', normalize=True) * 100
print(class_balance.round(2))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
class_balance.plot(kind='bar', legend=False, color=['#1f77b4', '#ff7f0e'])
plt.title('Répartition des Classes (%)')
plt.xticks([0, 1], ['Anormal', 'Normal'], rotation=0)
plt.ylabel('Pourcentage')

plt.subplot(1, 2, 2)
data[FEATURES].plot(kind='box', patch_artist=True)
plt.title('Distribution des Caractéristiques')
plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'class_balance.png'))
plt.close()

# ========== ENCODAGE ==========
encoder = LabelEncoder()
data['État'] = encoder.fit_transform(data['État'])  # Normal = 1, Anormal = 0

# ========== SCALING & SPLIT ==========
X = data[FEATURES]
y = data['État']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ========== KNN MODEL ==========
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict the test set labels
y_pred = knn.predict(X_test)

# Calculate accuracy
final_accuracy = accuracy_score(y_test, y_pred)

# Calculate log loss
y_proba = knn.predict_proba(X_test)
final_log_loss = log_loss(y_test, y_proba)

# Output results
print(f"✅ Précision finale (k=3) : {final_accuracy:.4f}")
print(f"📉 Log Loss final (k=3) : {final_log_loss:.4f}")

# ========== SAVE MODEL ==========
save_dir = os.path.join(current_dir, "model_output")
os.makedirs(save_dir, exist_ok=True)

joblib.dump(knn, os.path.join(save_dir, "knn_model.pkl"))
joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
joblib.dump(encoder, os.path.join(save_dir, "label_encoder.pkl"))

print("\n✅ Modèle, scaler, et encoder enregistrés dans 'model_output/'")

# ========== VISUALISATION DU MODÈLE ==========
train_sizes, train_scores, test_scores = learning_curve(
    knn, X_scaled, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='accuracy'
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(train_sizes, train_scores_mean, 'o-', color='green', label='Train Accuracy')
plt.plot(train_sizes, test_scores_mean, 'o-', color='blue', label='Validation Accuracy')
plt.title('Courbe de Précision - KNN')
plt.xlabel('Taille de l\'échantillon')
plt.ylabel('Accuracy')
plt.grid()
plt.legend()

# Log Loss for different k
k_values = list(range(1, 11))
losses = []
for k in k_values:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train, y_train)
    y_proba_k = knn_k.predict_proba(X_test)
    losses.append(log_loss(y_test, y_proba_k))

plt.subplot(1, 2, 2)
plt.plot(k_values, losses, 'o-', color='red')
plt.title('Log Loss vs Valeur de K')
plt.xlabel('K')
plt.ylabel('Log Loss')
plt.grid()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "knn_accuracy_loss.png"))
plt.close()
