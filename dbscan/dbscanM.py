from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from collections import Counter
from cleaning import load_normalized_dataset
import numpy as np
import matplotlib.pyplot as plt

# Cargar datos normalizados
X, _ = load_normalized_dataset()

def aplicar_dbscan(eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)

    # Contar elementos por grupo (-1 es ruido)
    conteo = Counter(labels)
    num_clusters = len([label for label in conteo if label != -1])
    
    print(f"\nResultado para eps={eps}, min_samples={min_samples}")
    print(f"Número de grupos encontrados (sin contar ruido): {num_clusters}")
    print("Elementos por grupo (incluyendo ruido):")
    for grupo, cantidad in sorted(conteo.items()):
        if grupo == -1:
            print(f"Ruido (-1): {cantidad}")
        else:
            print(f"Grupo {grupo}: {cantidad}")
    
    # Solo calculamos índices si hay al menos 2 clusters
    if num_clusters > 1:
        X_no_noise = X[labels != -1]
        labels_no_noise = labels[labels != -1]

        davies_bouldin = davies_bouldin_score(X_no_noise, labels_no_noise)
        calinski_harabasz = calinski_harabasz_score(X_no_noise, labels_no_noise)

        print("\nÍndices de validez del clustering:")
        print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
        print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")
    else:
        print("\nNo hay suficientes clusters para calcular índices de validez.")

    # Gráfica con PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 5))
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            # Ruido
            plt.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1],
                        c='k', marker='x', label='Ruido (-1)', alpha=0.5)
        else:
            plt.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1],
                        label=f'Cluster {label}', alpha=0.6)

    plt.title(f'DBSCAN con eps={eps}, min_samples={min_samples} (PCA)')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return labels

# Probar
aplicar_dbscan(eps=0.97, min_samples=25)
