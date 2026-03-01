"""
TP Général — Partie 1 : Pre-processing (Nettoyage et Transformation)
Dataset : loan_prediction.csv (Risque de crédit)

Fichier attendu :
TP_General/data/loan_prediction.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "loan_prediction.csv")

# 1) Chargement
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Fichier introuvable : {DATA_PATH}\n"
        "➡️ Place 'loan_prediction.csv' dans TP_General/data/"
    )

df = pd.read_csv(DATA_PATH)

print("=== 1) Inspection initiale ===")
print(df.head())
print("\n--- info() ---")
print(df.info())
print("\n--- describe() ---")
print(df.describe())

# 2) Doublons
print("\n=== 2) Doublons ===")
nb_dup = df.duplicated().sum()
print("Nombre de doublons :", nb_dup)
df = df.drop_duplicates()

# 3) Visualisations
print("\n=== 3) Visualisations ===")
plt.figure()
sns.histplot(df["ApplicantIncome"], bins=30, kde=True)
plt.title("Histogramme - ApplicantIncome")
plt.xlabel("ApplicantIncome")
plt.ylabel("Fréquence")
plt.show()

num_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(8, 6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="Blues", fmt=".2f")
plt.title("Heatmap de corrélation (variables numériques)")
plt.show()

# 4) Nettoyage : valeurs manquantes
print("\n=== 4) Valeurs manquantes ===")
na_counts = df.isna().sum().sort_values(ascending=False)
print(na_counts[na_counts > 0])

# Stratégie 1 : Numérique (LoanAmount -> moyenne)
if "LoanAmount" in df.columns:
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].mean())

# Stratégie 2 : Catégorielle (mode) Gender, Self_Employed
for col in ["Gender", "Self_Employed"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode(dropna=True)[0])

# Stratégie 3 : Critique (Credit_History) -> supprimer lignes
if "Credit_History" in df.columns:
    df = df.dropna(subset=["Credit_History"])

print("\nAprès imputation/suppression :")
print(df.isna().sum()[df.isna().sum() > 0])

# 5) Outliers : ApplicantIncome (IQR)
print("\n=== 5) Outliers ApplicantIncome (IQR) ===")
plt.figure()
sns.boxplot(x=df["ApplicantIncome"])
plt.title("Boxplot - ApplicantIncome (avant filtrage)")
plt.show()

Q1 = df["ApplicantIncome"].quantile(0.25)
Q3 = df["ApplicantIncome"].quantile(0.75)
IQR = Q3 - Q1
seuil_sup = Q3 + 1.5 * IQR

df = df[df["ApplicantIncome"] <= seuil_sup]

plt.figure()
sns.boxplot(x=df["ApplicantIncome"])
plt.title("Boxplot - ApplicantIncome (après filtrage)")
plt.show()

print("Taille après suppression outliers :", df.shape)

# 6) Transformation
print("\n=== 6) Transformation ===")

# 6.1 Supprimer Loan_ID
if "Loan_ID" in df.columns:
    df = df.drop(columns=["Loan_ID"])

# 6.2 One-hot encoding Property_Area
if "Property_Area" in df.columns:
    df = pd.get_dummies(df, columns=["Property_Area"], drop_first=False)

# 6.3 Cible Loan_Status Y/N -> 1/0
if "Loan_Status" in df.columns:
    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# 6.4 Standardisation ApplicantIncome et LoanAmount
scaler = StandardScaler()
for col in ["ApplicantIncome", "LoanAmount"]:
    if col in df.columns:
        df[col] = scaler.fit_transform(df[[col]])

print("\n✅ Dataset prêt (aperçu) :")
print(df.head())
print("\nColonnes finales :", df.columns.tolist())
