"""
TP Général — Partie non supervisée (Tech-Industries)
1) KMeans (blobs) + elbow
2) Cercles : KMeans vs DBSCAN
3) PCA sur Breast Cancer (30 features)
4) Isolation Forest (anomalies)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs, make_circles, load_breast_cancer
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# -----------------------------
# 1) KMeans sur pièces (blobs)
# -----------------------------
X_pieces, _ = make_blobs(n_samples=400, centers=3, cluster_std=0.8, random_state=42)

plt.figure(figsize=(7,5))
plt.scatter(X_pieces[:,0], X_pieces[:,1], s=30, alpha=0.7)
plt.title("Pièces (non étiquetées)")
plt.xlabel("Poids (std)")
plt.ylabel("Résistance (std)")
plt.show()

# Elbow 1..10
inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(X_pieces)
    inertias.append(km.inertia_)

plt.figure()
plt.plot(list(K_range), inertias, marker="o")
plt.title("Elbow method (inertie vs K)")
plt.xlabel("K")
plt.ylabel("Inertie")
plt.show()

# KMeans K=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
labels = kmeans.fit_predict(X_pieces)
centers = kmeans.cluster_centers_

plt.figure(figsize=(7,5))
plt.scatter(X_pieces[:,0], X_pieces[:,1], c=labels, cmap="viridis", s=30)
plt.scatter(centers[:,0], centers[:,1], c="red", s=200, marker="X")
plt.title("KMeans (K=3) + centroïdes")
plt.show()

# --------------------------------------
# 2) Cercles : KMeans échoue / DBSCAN OK
# --------------------------------------
X_circles, _ = make_circles(n_samples=400, factor=0.5, noise=0.05, random_state=42)

km2 = KMeans(n_clusters=2, random_state=42, n_init="auto")
y_km = km2.fit_predict(X_circles)

plt.figure()
plt.scatter(X_circles[:,0], X_circles[:,1], c=y_km, cmap="viridis")
plt.title("KMeans sur cercles (souvent incohérent)")
plt.show()

db = DBSCAN(eps=0.15, min_samples=5)
y_db = db.fit_predict(X_circles)

plt.figure()
plt.scatter(X_circles[:,0], X_circles[:,1], c=y_db, cmap="viridis")
plt.title("DBSCAN sur cercles (séparation par densité)")
plt.show()

# -----------------------------
# 3) PCA sur Breast Cancer (30)
# -----------------------------
data = load_breast_cancer()
X_sensors = data.data
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sensors)

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)

plt.figure(figsize=(7,5))
plt.scatter(X_2d[:,0], X_2d[:,1], c=y, cmap="coolwarm", s=20, alpha=0.8)
plt.title("PCA 2D (Breast Cancer)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

print("Explained variance ratio :", pca.explained_variance_ratio_)
print("Variance totale conservée :", pca.explained_variance_ratio_.sum())

# -----------------------------
# 4) Isolation Forest (anomalies)
# -----------------------------
X_regular, _ = make_blobs(n_samples=300, centers=1, cluster_std=1.0, random_state=42)
X_outliers, _ = make_blobs(n_samples=15, centers=1, cluster_std=10.0, center_box=(-10, 10), random_state=42)
X_system = np.vstack([X_regular, X_outliers])

iso = IsolationForest(contamination=0.05, random_state=42)
pred = iso.fit_predict(X_system)  # -1 anomalies, 1 normal

plt.figure(figsize=(7,5))
plt.scatter(X_system[pred==1,0], X_system[pred==1,1], s=25, alpha=0.7, label="Normal")
plt.scatter(X_system[pred==-1,0], X_system[pred==-1,1], s=40, alpha=0.9, label="Anomalie")
plt.title("Isolation Forest — Détection d'anomalies")
plt.legend()
plt.show()

print("Anomalies détectées :", np.sum(pred==-1))
