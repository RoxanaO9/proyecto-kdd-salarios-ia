# FASE 4.4: CLUSTERING

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')

# 1. Carga de Datos Procesados 
print("\n Cargando datos procesados: ")
try:
    df_encoded = pd.read_csv('salarios_procesados.csv')
    print(f" Datos procesados cargados. Dimensiones: {df_encoded.shape}")
except FileNotFoundError:
    print(" Error: No se encontró 'salarios_procesados.csv'. Ejecute '01_preprocesamiento.py' primero.")
    exit()

# 2. Preparación de Datos para Clustering 
print("\n Preparando datos para clustering: ")

# Seleccionar variables relevantes para clustering
cluster_vars = ['remote_ratio', 'work_year']
cluster_vars += [col for col in df_encoded.columns if col.startswith('experience_level_')]
cluster_vars += [col for col in df_encoded.columns if col.startswith('company_size_')]
# Filtrar solo las que existen en el DataFrame
cluster_vars = [col for col in cluster_vars if col in df_encoded.columns]

cluster_data = df_encoded[cluster_vars].copy()

# Escalar los datos de clustering
cluster_scaler = StandardScaler()
cluster_data_scaled = cluster_scaler.fit_transform(cluster_data)

print(f" Datos escalados para clustering: {cluster_data_scaled.shape}")

# 3. Encontrar el Número Óptimo de Clusters (Método del Codo)
print("\n Encontrando número óptimo de clusters (Método del Codo): ")

inertias = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(cluster_data_scaled)
    inertias.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, marker='o', linestyle='--', color='navy', linewidth=2)
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.title('Método del Codo para Clustering')
plt.grid(True, alpha=0.3)
plt.savefig('metodo_codo_mejorado.png', dpi=300, bbox_inches='tight')
plt.show()
print(" Gráfico del codo guardado como 'metodo_codo_mejorado.png'")

# 4. Aplicar K-Means (con 4 clusters) 
print("\n Aplicando K-Means con 4 clusters: ")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_encoded['cluster'] = kmeans.fit_predict(cluster_data_scaled)

# 5. Análisis de Clusters 
print("\n Análisis de Clusters: ")
cluster_analysis = df_encoded.groupby('cluster').agg({
    'salary_in_usd': ['mean', 'min', 'max', 'count'],
    'remote_ratio': 'mean',
    'work_year': 'mean'
}).round(2)
print(cluster_analysis)

# 6. Visualización de Clusters (PCA) 
print("\n Visualizando Clusters con PCA: ")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(cluster_data_scaled)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1],
    c=df_encoded['cluster'], cmap='viridis', alpha=0.6, s=30)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Visualización de Clusters (PCA)')
plt.colorbar(scatter, label='Cluster')
plt.savefig('clustering_mejorado.png', dpi=300, bbox_inches='tight')
plt.show()
print(" Visualización de clusters guardada como 'clustering_mejorado.png'")

# 7. Guardar Resultados
print("\n Guardando resultados con cluster: ")
df_encoded.to_csv('salarios_con_cluster.csv', index=False)
print(" Archivo 'salarios_con_cluster.csv' generado")
