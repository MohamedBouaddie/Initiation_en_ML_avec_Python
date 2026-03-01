"""
TP Général — Partie 2 : Classification supervisée (Wine)
Comparer KNN, RandomForest, SVM puis évaluer le meilleur modèle.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

wine = load_wine()
X = wine.data
y = wine.target

print("Features :", wine.feature_names)
print("Classes  :", wine.target_names)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Modèle A : KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"\nAccuracy KNN : {acc_knn:.3f}")

# Modèle B : Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy RandomForest : {acc_rf:.3f}")

# Bonus : importance des variables
importances = rf.feature_importances_
best_idx = int(np.argmax(importances))
print("Feature la plus importante (RF) :", wine.feature_names[best_idx], "| importance =", importances[best_idx])

# Modèle C : SVM RBF
svm = SVC(kernel="rbf", random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"Accuracy SVM (rbf) : {acc_svm:.3f}")

# Choisir le meilleur
scores = {"KNN": acc_knn, "RandomForest": acc_rf, "SVM": acc_svm}
best_name = max(scores, key=scores.get)
print("\n✅ Meilleur modèle :", best_name, "| score =", scores[best_name])

best_model = {"KNN": knn, "RandomForest": rf, "SVM": svm}[best_name]
best_pred  = {"KNN": y_pred_knn, "RandomForest": y_pred_rf, "SVM": y_pred_svm}[best_name]

# Évaluation avancée (matrice + report + cross-val)
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=wine.target_names, yticklabels=wine.target_names)
plt.title(f"Matrice de confusion — {best_name}")
plt.ylabel("Vraie classe")
plt.xlabel("Classe prédite")
plt.show()

print("\n--- Classification report ---")
print(classification_report(y_test, best_pred, target_names=wine.target_names))

cv_scores = cross_val_score(best_model, X, y, cv=5)
print("\nCross-validation (cv=5) :", cv_scores)
print("Moyenne =", cv_scores.mean(), "| Écart-type =", cv_scores.std())
