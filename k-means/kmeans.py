# Importamos las librerías necesarias
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

# Función para graficar clusters
def graficar_clusters_pca(X, resultados):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    for k, resultado in resultados.items():
        etiquetas = resultado["etiquetas"]

        plt.figure(figsize=(6, 4))
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=etiquetas, cmap="tab10", s=15)
        plt.title(f"K-Means desde cero con k = {k}")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.grid(True)
        plt.show()

# Función para cargar y limpiar el dataset
def load_clean_dataset(ruta="Dry_Bean_Dataset.arff"):
    data, meta = arff.loadarff(ruta)
    df = pd.DataFrame(data)

    if "Class" in df.columns:
        df = df.drop(columns=["Class"])
    
    return df

# Función para cargar y normalizar el dataset
def load_normalized_dataset(ruta="Dry_Bean_Dataset.arff"):
    df = load_clean_dataset(ruta)
    
    scaler = StandardScaler()
    df_normalized = scaler.fit_transform(df)
    
    return df_normalized, scaler

# Implementación de K-Means manual
def kmeans_fun(X, k, max_iter=100):
    n_samples, n_features = X.shape

    rng = np.random.default_rng(seed=42)
    indices_iniciales = rng.choice(n_samples, size=k, replace=False)
    centroides = X[indices_iniciales]

    for iteracion in range(max_iter):
        distancias = np.linalg.norm(X[:, np.newaxis] - centroides, axis=2)
        etiquetas = np.argmin(distancias, axis=1)

        nuevos_centroides = np.array([
            X[etiquetas == i].mean(axis=0) if len(X[etiquetas == i]) > 0 else centroides[i]
            for i in range(k)
        ])

        if np.allclose(centroides, nuevos_centroides):
            print(f"Convergió en la iteración {iteracion}")
            break
        
        centroides = nuevos_centroides

    inercia = np.sum([
        np.linalg.norm(X[i] - centroides[etiquetas[i]])**2
        for i in range(n_samples)
    ])

    return etiquetas, centroides, inercia

# Bloque principal
if __name__ == "__main__":
    df = load_clean_dataset()
    print("Dataset limpio:")
    print(df.head())

    df_normalized, scaler = load_normalized_dataset()
    print("\nDataset normalizado:")
    print(df_normalized[:5])

    print("Ejecución de K means")

    ks = [6, 7, 8]
    resultados_kmeans = {}

    for k in ks:
        print("_____________________________________")
        print("Ejecución de kmeans con k =", k)
        etiquetas, centroides, inercia = kmeans_fun(df_normalized, k)
        print(f"\nK = {k}")
        print("Inercia:", inercia)

        db_index = davies_bouldin_score(df_normalized, etiquetas)
        ch_score = calinski_harabasz_score(df_normalized, etiquetas)

        print("Davies-Bouldin Index:", db_index)
        print("Calinski-Harabasz Score:", ch_score)

        print(f"Partición con k = {k}:")
        unique_labels, counts = np.unique(etiquetas, return_counts=True)
        for i, count in zip(unique_labels, counts):
            print(f"Cluster {i + 1}: {count} elementos")
        
        resultados_kmeans[k] = {
            "etiquetas": etiquetas,
            "centroides": centroides,
            "inercia": inercia,
            "davies_bouldin": db_index,
            "calinski_harabasz": ch_score
        }

    graficar_clusters_pca(df_normalized, resultados_kmeans)
