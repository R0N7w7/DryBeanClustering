import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler

# Función para cargar el dataset limpio (sin la columna 'Class')
def load_clean_dataset(ruta="Dry_Bean_Dataset.arff"):
    data, meta = arff.loadarff(ruta)
    df = pd.DataFrame(data)

    if "Class" in df.columns:
        df = df.drop(columns=["Class"])
    
    return df

# Función para cargar el dataset y normalizarlo
def load_normalized_dataset(ruta="Dry_Bean_Dataset.arff"):
    df = load_clean_dataset(ruta)
    
    # Normalizar los datos
    scaler = StandardScaler()
    df_normalized = scaler.fit_transform(df)
    
    return df_normalized, scaler

# Solo se ejecuta si este script es llamado directamente
if __name__ == "__main__":
    df = load_clean_dataset()
    print("Dataset limpio:")
    print(df.head())

    # Llamamos la función de normalización
    df_normalized, scaler = load_normalized_dataset()
    print("\nDataset normalizado:")
    print(df_normalized[:5])  # Muestra las primeras 5 filas normalizadas