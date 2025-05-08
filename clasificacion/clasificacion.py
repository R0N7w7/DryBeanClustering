import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from scipy.io import arff

# === Cargar dataset con etiquetas ===
def load_dataset_with_labels(ruta="Dry_Bean_Dataset.arff"):
    data, meta = arff.loadarff(ruta)
    df = pd.DataFrame(data)
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(str)  # Etiquetas como strings
    return X, y

# === Graficar con PCA ===
def plot_pca(X, labels, title):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=10)
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.show()

# === Evaluar clasificador ===
def evaluar_clasificador(nombre, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"{nombre} => Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")
    return acc, f1

# === Entrenamiento general ===
def entrenar_y_evaluar(X, y, nombre_conjunto):
    # Codificar etiquetas si son strings
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Dividir en entrenamiento/prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    print(f"\n----- Resultados para {nombre_conjunto} -----")

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    acc_nb, f1_nb = evaluar_clasificador("Naive Bayes", y_test, y_pred_nb)
    plot_pca(X_test, y_pred_nb, f"Naive Bayes - {nombre_conjunto}")

    # Red Neuronal
    nn = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
    nn.fit(X_train, y_train)
    y_pred_nn = nn.predict(X_test)
    acc_nn, f1_nn = evaluar_clasificador("Red Neuronal", y_test, y_pred_nn)
    plot_pca(X_test, y_pred_nn, f"Red Neuronal - {nombre_conjunto}")

    return {
        'Naive Bayes': {'accuracy': acc_nb, 'f1': f1_nb},
        'Red Neuronal': {'accuracy': acc_nn, 'f1': f1_nn}
    }
    
# === Graficar comparativa de métricas ===
def graficar_comparativa_metricas(resultados_reales, resultados_kmeans):
    modelos = list(resultados_reales.keys())
    metricas = ['accuracy', 'f1']
    x = np.arange(len(modelos))  # posiciones en el eje x

    width = 0.35  # ancho de barras

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    for i, metrica in enumerate(metricas):
        reales = [resultados_reales[m][metrica] for m in modelos]
        kmeans = [resultados_kmeans[m][metrica] for m in modelos]

        ax[i].bar(x - width/2, reales, width, label='Reales', color='skyblue')
        ax[i].bar(x + width/2, kmeans, width, label='KMeans', color='salmon')
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(modelos)
        ax[i].set_ylim(0, 1.1)
        ax[i].set_ylabel(metrica.capitalize())
        ax[i].set_title(f'Comparativa de {metrica.upper()}')
        ax[i].legend()

    plt.tight_layout()
    plt.show()


# === MAIN ===
if __name__ == "__main__":
    # Cargar datos reales
    X_raw, y_real = load_dataset_with_labels()

    # Normalizar
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # KMeans para etiquetas sintéticas
    kmeans = KMeans(n_clusters=6, random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    # Evaluar con etiquetas reales
    resultados_reales = entrenar_y_evaluar(X, y_real, "Etiquetas Reales")

    # Evaluar con etiquetas KMeans
    resultados_kmeans = entrenar_y_evaluar(X, y_kmeans, "Etiquetas por KMeans")

    # Mostrar resumen
    print("\n==== Resumen de Resultados ====")
    print("\nEtiquetas Reales:")
    for modelo, res in resultados_reales.items():
        print(f"{modelo}: Accuracy={res['accuracy']:.4f} | F1 Score={res['f1']:.4f}")

    print("\nEtiquetas por KMeans:")
    for modelo, res in resultados_kmeans.items():
        print(f"{modelo}: Accuracy={res['accuracy']:.4f} | F1 Score={res['f1']:.4f}")
        
        # Graficar comparativa
    graficar_comparativa_metricas(resultados_reales, resultados_kmeans)

