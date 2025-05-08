from sklearn.cluster import DBSCAN
from cleaning import load_clean_dataset, load_normalized_dataset
import numpy as np
import pandas as pd

def find_valid_dbscan_configs(min_clusters=4, max_noise_ratio=0.2):
    # 1. Cargar datos
    print("Cargando datos...")
    df_original = load_clean_dataset()
    df_normalized, _ = load_normalized_dataset()
    
    # 2. Rangos de parámetros para explorar
    eps_values = np.linspace(1., 1.84, 50)
    min_samples_values = range(4, 40, 1)
    
    # 3. Lista para almacenar resultados válidos
    valid_configs = []
    
    print("\nBuscando configuraciones válidas...")
    print(f"Criterios: ≥{min_clusters} clusters, ≤{max_noise_ratio:.0%} ruido")
    print("="*50)
    
    # 4. Búsqueda exhaustiva de parámetros
    for eps in eps_values:
        for min_samples in min_samples_values:
            # Ejecutar DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(df_normalized)
            
            # Calcular métricas
            n_noise = (clusters == -1).sum()
            noise_ratio = n_noise / len(clusters)
            n_clusters = len(set(clusters) - {-1})
            # Mostrar progreso
            print(f"✓ Config {len(valid_configs)}: eps={eps:.2f}, min_samples={min_samples} | "
                    f"Clusters: {n_clusters} | Ruido: {noise_ratio:.1%}")
    
    # 5. Procesar resultados
    if not valid_configs:
        print("\n⚠ No se encontraron configuraciones válidas")
        print(f"Requisitos: ≥{min_clusters} clusters, ≤{max_noise_ratio:.0%} ruido")
        return None
    
    # 6. Crear DataFrame con resultados
    results_df = pd.DataFrame([{
        'eps': config['eps'],
        'min_samples': config['min_samples'],
        'n_clusters': config['n_clusters'],
        'noise_ratio': config['noise_ratio'],
        'cluster_sizes': ', '.join(map(str, sorted(config['cluster_sizes'], reverse=True))),
        'config_id': idx
    } for idx, config in enumerate(valid_configs, 1)])
    
    # 7. Mostrar resumen
    print("\n" + "="*50)
    print(f"RESUMEN: Se encontraron {len(valid_configs)} configuraciones válidas")
    print("="*50)
    print(results_df.sort_values(by=['n_clusters', 'noise_ratio'], ascending=[False, True]))
    print("\nPara usar una configuración específica:")
    print("1. Elige un config_id del listado")
    print("2. Usa: df_original['Cluster'] = valid_configs[config_id-1]['labels']")
    
    return valid_configs, results_df

# Ejemplo de uso:
all_configs, configs_summary = find_valid_dbscan_configs(
    min_clusters=3,      # Mínimo 4 clusters
    max_noise_ratio=0.45  # Máximo 20% de ruido
)

# Para acceder a una configuración específica:
if all_configs is not None:
    # Ejemplo: usar la primera configuración encontrada
    config_id = 1  # Cambiar por el ID deseado
    selected_config = all_configs[config_id-1]
    
    # Cargar datos nuevamente para asignar clusters
    df_original = load_clean_dataset()
    df_original['Cluster'] = selected_config['labels']
    
    print(f"\nAsignada configuración {config_id} al DataFrame:")
    print(f"• Parámetros: eps={selected_config['eps']:.2f}, min_samples={selected_config['min_samples']}")
    print(f"• Clusters formados: {selected_config['n_clusters']}")
    print(f"• Porcentaje de ruido: {selected_config['noise_ratio']:.1%}")