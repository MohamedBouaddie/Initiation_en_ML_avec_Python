"""
TP3 — Apprentissage Non Supervisé (Atlas Bank)
- KMeans + elbow sur clients (make_blobs)
- KMeans vs DBSCAN sur données complexes (make_moons)
- PCA (Wine dataset)
- Isolation Forest (fraude/anomalies)

Référence : TP3_Ennoncé.PDF
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs, make_moons, load_wine
from sklearn.preprocessing import StandardScaler

plt.style.use("ggplot")
np.random.seed(42)

print("Environnement prêt.")

# ---------------------------
# Partie 1 : KMeans (clients)
# ---------------------------
X_clients, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

plt.figure()
plt.scatter(X_clients[:,0], X_clients[:,1], s=50)
plt.title("Données Clients (Non étiquetées)")
plt.show()

# Elbow k=1..10
inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(X_clients)
    inertias.append(km.inertia_)

plt.figure()
plt.plot(list(K_range), inertias, marker="o")
plt.title("Elbow method (clients)")
plt.xlabel("k")
plt.ylabel("Inertie")
plt.show()

# KMeans avec k=4 (souvent le coude)
k_opt = 4
kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init="auto")
labels = kmeans.fit_predict(X_clients)
centers = kmeans.cluster_centers_

plt.figure()
plt.scatter(X_clients[:,0], X_clients[:,1], c=labels, cmap="viridis", s=40)
plt.scatter(centers[:,0], centers[:,1], c="red", s=200, marker="X")
plt.title(f"KMeans (k={k_opt}) + centroïdes")
plt.show()

# ---------------------------
# Partie 2 : KMeans vs DBSCAN
# ---------------------------
X_complex, _ = make_moons(n_samples=300, noise=0.05, random_state=0)

km2 = KMeans(n_clusters=2, random_state=42, n_init="auto")
y_km = km2.fit_predict(X_complex)

plt.figure()
plt.scatter(X_complex[:,0], X_complex[:,1], c=y_km, cmap="viridis", s=30)
plt.title("KMeans sur make_moons (séparation souvent mauvaise)")
plt.show()

db = DBSCAN(eps=0.3, min_samples=5)
y_db = db.fit_predict(X_complex)

plt.figure()
plt.scatter(X_complex[:,0], X_complex[:,1], c=y_db, cmap="viridis", s=30)
plt.title("DBSCAN sur make_moons (clusters + bruit=-1)")
plt.show()

print("Points considérés comme bruit (label=-1) :", np.sum(y_db == -1))

# ---------------------------
# Partie 3 : PCA sur Wine (13 → 2)
# ---------------------------
data_wine = load_wine()
X_wine = data_wine.data
y_wine = data_wine.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_wine)

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)

plt.figure(figsize=(7,5))
plt.scatter(X_2d[:,0], X_2d[:,1], c=y_wine, cmap="viridis", s=30)
plt.title("PCA 2D — Wine")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

print("Variance expliquée :", pca.explained_variance_ratio_)
print("Total conservé :", pca.explained_variance_ratio_.sum())

# ---------------------------
# Partie 4 : Isolation Forest (fraude)
# ---------------------------
X_normal, _ = make_blobs(n_samples=300, centers=1, cluster_std=1.0, random_state=42)
X_anomalies, _ = make_blobs(n_samples=20, centers=1, cluster_std=10.0, random_state=42)
X_transactions = np.vstack([X_normal, X_anomalies])

iso = IsolationForest(contamination=0.05, random_state=42)
pred = iso.fit_predict(X_transactions)  # -1 anomalie, 1 normal

plt.figure(figsize=(7,5))
plt.scatter(X_transactions[pred==1,0], X_transactions[pred==1,1], s=25, label="Normal")
plt.scatter(X_transactions[pred==-1,0], X_transactions[pred==-1,1], s=50, label="Anomalie")
plt.title("Isolation Forest — Fraude détectée")
plt.legend()
plt.show()

print("Anomalies détectées :", np.sum(pred==-1))
