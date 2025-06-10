import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
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
    fichiers = [f for f in os.listdir(DATABASE_FOLDER) if f.endswith('.xlsx') and not f.startswith('~$')]
    if not fichiers:
        print("❌ Aucun fichier .xlsx trouvé dans le dossier")
        exit()

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

# ========== RANDOM FOREST MODEL ==========
print("\n🌲 Entraînement du modèle Random Forest...")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)

rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_log_loss = log_loss(y_test, y_proba_rf)

print(f"✅ Précision Random Forest : {rf_accuracy:.4f}")
print(f"📉 Log Loss Random Forest : {rf_log_loss:.4f}")

# ========== SAVE MODEL ==========
save_dir = os.path.join(current_dir, "model_output")
os.makedirs(save_dir, exist_ok=True)

joblib.dump(rf, os.path.join(save_dir, "random_forest_model.pkl"))
joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
joblib.dump(encoder, os.path.join(save_dir, "label_encoder.pkl"))

print("\n✅ Modèle Random Forest, scaler, et encoder enregistrés dans 'model_output/'")

# ========== COURBE D'APPRENTISSAGE ==========
train_sizes, train_scores, test_scores = learning_curve(
    rf, X_scaled, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='accuracy'
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_sizes, train_scores_mean, 'o-', color='green', label='Train Accuracy')
plt.plot(train_sizes, test_scores_mean, 'o-', color='blue', label='Validation Accuracy')
plt.title('Courbe de Précision - Random Forest')
plt.xlabel('Taille de l\'échantillon')
plt.ylabel('Accuracy')
plt.grid()
plt.legend()

# ========== LOG LOSS POUR DIFFERENTS ESTIMATORS ==========
estimators_range = list(range(10, 201, 10))
losses = []

for n in estimators_range:
    rf_tmp = RandomForestClassifier(n_estimators=n, random_state=42)
    rf_tmp.fit(X_train, y_train)
    y_proba_tmp = rf_tmp.predict_proba(X_test)
    losses.append(log_loss(y_test, y_proba_tmp))

plt.subplot(1, 2, 2)
plt.plot(estimators_range, losses, 'o-', color='red')
plt.title('Log Loss vs n_estimators (Random Forest)')
plt.xlabel('n_estimators')
plt.ylabel('Log Loss')
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "random_forest_accuracy_loss.png"))
plt.close()

# ========== COURBE ROC ==========
fpr, tpr, thresholds = roc_curve(y_test, y_proba_rf[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Courbe ROC - Random Forest')
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "roc_curve_random_forest.png"))
plt.close()
print("📈 Courbe ROC enregistrée.")

# ========== MATRICE DE CONFUSION ==========
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)

plt.figure(figsize=(5, 5))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Matrice de Confusion - Random Forest')
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "confusion_matrix_random_forest.png"))
plt.close()
print("🧩 Matrice de confusion enregistrée.")
