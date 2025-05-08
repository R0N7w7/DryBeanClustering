import pandas as pd
from scipy.io import arff

from cleaning import load_clean_dataset, load_normalized_dataset

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Cargar datos
df_original = load_clean_dataset()
df_normalized, scaler = load_normalized_dataset()

# Lista con los números de clusters deseados
cantidades_clusters = [6, 7, 8]

print("=== Resultados del Clustering Jerárquico Aglomerativo ===\n")

for n_clusters in cantidades_clusters:
    model = AgglomerativeClustering(n_clusters=n_clusters)
    etiquetas = model.fit_predict(df_normalized)

    # Contar elementos en cada cluster
    conteos = pd.Series(etiquetas).value_counts().sort_index()
    
    print(f"Partición {cantidades_clusters.index(n_clusters)+1}: ({n_clusters} clusters)")
    for cluster_id, cantidad in conteos.items():
        print(f"Cluster {cluster_id + 1}: {cantidad} elementos")
    
    # Calcular índices de validez
    davies_bouldin = davies_bouldin_score(df_normalized, etiquetas)
    calinski_harabasz = calinski_harabasz_score(df_normalized, etiquetas)

    print("\nÍndices de validez:")
    print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
    print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")
    print()

    # Aplicar PCA para reducción a 2D
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_normalized)

    # Graficar los clusters
    plt.figure(figsize=(8, 5))
    for i in range(n_clusters):
        plt.scatter(df_pca[etiquetas == i, 0], df_pca[etiquetas == i, 1], label=f'Cluster {i+1}', alpha=0.6)
    
    plt.title(f'Clustering Jerárquico con {n_clusters} clusters (PCA)')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
