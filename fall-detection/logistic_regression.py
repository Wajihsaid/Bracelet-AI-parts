import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# 🔹 1. Charger le dataset
data = pd.read_csv("binaire.csv")
data = data.dropna()
data = data.drop(columns=['TimeStamp', 'Sample No', 'Sensor ID'])

# 🔹 2. Équilibrage des classes
data = data.groupby("Label").apply(lambda x: x.sample(n=117564, random_state=42)).reset_index(drop=True)

# 🔹 3. Séparer features et labels
X = data.drop("Label", axis=1).values
y = data["Label"].values

# 🔹 4. Encodage des labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 🔹 5. Normalisation des features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 6. Diviser en données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# 🔹 7. Créer et entraîner le modèle
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 🔹 8. Évaluer le modèle
y_pred = model.predict(X_test)

# 🔸 Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🔹 Accuracy: {accuracy:.2f}")

# 🔸 Classification report
print("\n🔹 Classification Report :")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 🔸 Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Matrice de Confusion")
plt.xlabel("Prédiction")
plt.ylabel("Réel")
plt.show()

# 🔸 Courbe ROC
y_scores = model.predict_proba(X_test)[:, 1]  # Probabilité pour la classe positive
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
auc = roc_auc_score(y_test, y_scores)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random guess")
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.title("Courbe ROC - Régression Logistique")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
