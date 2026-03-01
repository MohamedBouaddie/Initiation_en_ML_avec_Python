"""
TP2 — Apprentissage Supervisé
- Dataset Iris : comparaison KNN / Arbre / SVM + évaluation SVM
- Régression linéaire : Salary_Data.csv (URL) + prédiction 12 ans

Référence : TP2_Ennoncé_251219_235525.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

# -----------------------
# Partie 1 : dataset Iris
# -----------------------
iris = datasets.load_iris()
X = iris.data
y = iris.target

print(f"Dimensions de X : {X.shape}")
print(f"Noms des classes : {iris.target_names}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Modèle A : KNN (K=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print(f"\nAccuracy KNN : {accuracy_score(y_test, y_pred_knn):.2f}")

# Modèle B : Arbre (max_depth=3)
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
print(f"Accuracy Arbre : {accuracy_score(y_test, y_pred_tree):.2f}")

# Modèle C : SVM linéaire
svm = SVC(kernel="linear", random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print(f"Accuracy SVM : {accuracy_score(y_test, y_pred_svm):.2f}")

# -----------------------
# Partie 3 : Évaluation SVM
# -----------------------
cm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.ylabel("Vraie classe")
plt.xlabel("Classe prédite")
plt.title("Matrice de Confusion (SVM)")
plt.show()

print("\n--- Rapport de Classification (SVM) ---")
print(classification_report(y_test, y_pred_svm, target_names=iris.target_names))

scores = cross_val_score(svm, X, y, cv=5)
print("\nScores CV=5 :", scores)
print(f"Accuracy moyenne : {scores.mean():.2f}")
print(f"Écart-type       : {scores.std():.2f}")

# -----------------------
# Partie 4 : Régression (Salary)
# -----------------------
url = "https://raw.githubusercontent.com/sudarshan-koirala/Salary-Prediciton-based-on-Years-of-Experience/master/Salary_Data.csv"
df_salary = pd.read_csv(url)

X_sal = df_salary[["YearsExperience"]]
y_sal = df_salary["Salary"]

reg = LinearRegression()
reg.fit(X_sal, y_sal)

plt.figure()
plt.scatter(X_sal, y_sal, label="Données réelles")
plt.plot(X_sal, reg.predict(X_sal), label="Droite de régression")
plt.xlabel("Années d'expérience")
plt.ylabel("Salaire")
plt.legend()
plt.title("Régression linéaire — Salary vs YearsExperience")
plt.show()

pred_12 = reg.predict([[12]])[0]
print("\nSalaire prédit pour 12 ans d'expérience :", pred_12)
