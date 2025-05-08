from sklearn.cluster import DBSCAN
from collections import Counter
from cleaning import load_normalized_dataset
import numpy as np

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
    
    return labels

# probar con valores como estos para empezar
aplicar_dbscan(eps=1.83, min_samples=18)

