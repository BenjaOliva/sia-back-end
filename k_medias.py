import numpy as np
from algoritmo import exec_algorithm
from load_data import cargar_datos_desde_arff
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Dataset de examen final
archivo_arff = 'k_medias_data/winequality.arff'

def distancia_euclidiana(a, b):
    """Calcula la distancia euclidiana entre dos puntos."""
    return np.linalg.norm(a - b)

def asignar_clusters(data, centros):
    """Asigna cada punto de datos al cluster más cercano."""
    clusters = []
    for punto in data:
        distancias = [distancia_euclidiana(punto, centro) for centro in centros]
        cluster_asignado = np.argmin(distancias)
        clusters.append(cluster_asignado)
    return np.array(clusters)

def actualizar_centros(data, clusters, num_clusters):
    """Actualiza la posición de los centros según los puntos asignados a cada cluster."""
    nuevos_centros = np.zeros((num_clusters, data.shape[1]))
    for i in range(num_clusters):
        puntos_en_cluster = data[clusters == i]
        if len(puntos_en_cluster) > 0:
            nuevos_centros[i] = np.mean(puntos_en_cluster, axis=0)
    return nuevos_centros

def k_medias_no_vectorizado(data, num_clusters, max_iter=100, tol=1e-5):
    """Implementación no vectorizada del algoritmo K-Medias."""
    # Inicializar centros de forma aleatoria
    centros = data[np.random.choice(len(data), num_clusters, replace=False)]
    print("Prev: ", centros)
    for _ in range(max_iter):
        # Asignar puntos a clusters
        clusters = asignar_clusters(data, centros)
        
        # Actualizar centros
        nuevos_centros = actualizar_centros(data, clusters, num_clusters)
        
       # Normalizar centros
        denominador = nuevos_centros.max(axis=0) - nuevos_centros.min(axis=0)
        denominador[denominador == 0] = 1  # Evitar divisiones por cero
        nuevos_centros = (nuevos_centros - nuevos_centros.min(axis=0)) / denominador

        # Verificar convergencia
        if np.all(np.abs(centros - nuevos_centros) < tol):
            break
        
        centros = nuevos_centros

    print("Normalizados: ", centros)

    return clusters, centros

def k_medias_vectorizado(data, num_clusters, max_iter=100, tol=1e-5):
    """Implementación vectorizada del algoritmo K-Medias."""
    centros = data[np.random.choice(len(data), num_clusters, replace=False)]
    print("Prev: ", centros)

    for _ in range(max_iter):
        # Asignar puntos a clusters
        distancias = np.linalg.norm(data[:, np.newaxis, :] - centros, axis=2)
        clusters = np.argmin(distancias, axis=1)

        # Actualizar centros
        nuevos_centros = np.array([np.mean(data[clusters == i], axis=0) if np.sum(clusters == i) > 0 else centros[i] for i in range(num_clusters)])

        # Normalizar centros
        denominador = nuevos_centros.max(axis=0) - nuevos_centros.min(axis=0)
        denominador[denominador == 0] = 1  # Evitar divisiones por cero
        nuevos_centros = (nuevos_centros - nuevos_centros.min(axis=0)) / denominador

        # Verificar convergencia
        if np.all(np.abs(centros - nuevos_centros) < tol):
            break

        centros = nuevos_centros

    print("Normalizados: ", centros)
    return clusters, centros

# Ejemplo de uso
data = cargar_datos_desde_arff(archivo_arff)
num_clusters = 2

# # No vectorizado
# clusters_no_vectorizado, centros_no_vectorizado = k_medias_no_vectorizado(data, num_clusters)
# print("Clusters (no vectorizado):", clusters_no_vectorizado)

# # Vectorizado
clusters_vectorizado, centros_vectorizado = k_medias_vectorizado(data, num_clusters)
# print("Clusters (vectorizado):", clusters_vectorizado)

def predecir_cluster_old(nuevo_elemento, centros):
    """Predice a qué cluster pertenecería un nuevo elemento."""
    distancias = [distancia_euclidiana(nuevo_elemento, centro) for centro in centros]
    cluster_predicho = np.argmin(distancias)
    return cluster_predicho

# Luego de ejecutar tu código de K-Medias...

# Ejemplo de predicción para un nuevo elemento
# nuevo_elemento1 = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
# cluster_predicho1 = predecir_cluster(nuevo_elemento1, centros_vectorizado)
# print("Cluster predicho para el nuevo elemento 1:", cluster_predicho1)

# nuevo_elemento2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.5, 0.6])
# cluster_predicho2 = predecir_cluster(nuevo_elemento2, centros_vectorizado)
# print("Cluster predicho para el nuevo elemento 2:", cluster_predicho2)

# # Ejemplo de ejecución con diferentes números de clusters
# num_clusters = [2, 3, 4, 5, 6]
# resultados_propios = []

# for k in num_clusters:
#     clusters, centros = k_medias_vectorizado(data, k)
#     # Guardar los resultados o métricas relevantes para cada ejecución
#     resultados_propios.append((k, clusters, centros))

# resultados_terceros = []

# for k in num_clusters:
#     kmeans = KMeans(n_clusters=k)
#     clusters = kmeans.fit_predict(data)
#     centros = kmeans.cluster_centers_
#     # Guardar los resultados o métricas relevantes para cada ejecución
#     resultados_terceros.append((k, clusters, centros))

#     # Visualizar los resultados
# for propios, terceros in zip(resultados_propios, resultados_terceros):
#     k_propios, clusters_propios, _ = propios
#     k_terceros, clusters_terceros, _ = terceros


# exec_algorithm()