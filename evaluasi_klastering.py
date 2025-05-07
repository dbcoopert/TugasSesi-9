#evaluasi klastering dari file kode sesi 7
# Import library yang diperlukan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Memuat dataset Iris
iris = load_iris()
X = iris.data

# Menjalankan algoritma K-Means dengan jumlah cluster yang ditentukan
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Evaluasi clustering
sil_score = silhouette_score(X, y_kmeans)
db_score = davies_bouldin_score(X, y_kmeans)
ch_score = calinski_harabasz_score(X, y_kmeans)

print(f"Silhouette Score: {sil_score:.3f}")
print(f"Davies-Bouldin Index: {db_score:.3f}")
print(f"Calinski-Harabasz Score: {ch_score:.3f}")

# Visualisasi hasil clustering
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('K-Means Clustering on Iris Dataset')
plt.show(