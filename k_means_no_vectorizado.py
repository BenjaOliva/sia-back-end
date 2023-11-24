import math
import random

def distancia_euclidiana(punto, centroide):
    """
    Calcula la distancia euclidiana entre un punto y un centroide (no vectorizada).
    """
    distancia = 0.0
    for i in range(len(punto)):
        distancia += (punto[i] - centroide[i]) ** 2
    return math.sqrt(distancia)

def asignar_clusters(datos, centroides):
    """
    Asigna cada punto al clúster del centroide más cercano (no vectorizada).
    """
    asignaciones = []
    for punto in datos:
        distancias = [distancia_euclidiana(punto, centroide) for centroide in centroides]
        asignacion = distancias.index(min(distancias))
        asignaciones.append(asignacion)
    return asignaciones

def k_means_no_vectorizado(datos, k, features=None, max_iter=500, seed=100):
    """
    Aplica el algoritmo k-means a los datos (no vectorizado).
    """
    if features is not None:
        # Seleccionar solo las características especificadas
        datos_seleccionados = datos[:, features]
    else:
        # Utilizar todos los atributos si no se especifican características seleccionadas
        datos_seleccionados = datos
    random.seed(seed)

    # Inicializar centroides de forma aleatoria
    centroides = [datos_seleccionados[i] for i in random.sample(range(len(datos_seleccionados)), k)]

    # Contador para el número de puntos asignados a cada clúster
    contador_puntos = [0] * k
    
    for _ in range(max_iter):
        # Asignar cada punto al clúster del centroide más cercano
        asignaciones = asignar_clusters(datos_seleccionados, centroides)
        
        # Actualizar centroides
        for i in range(k):
            puntos_cluster = [datos_seleccionados[j] for j in range(len(datos_seleccionados)) if asignaciones[j] == i]
            contador_puntos[i] = len(puntos_cluster)
            if len(puntos_cluster) > 0:
                centroides[i] = [sum(p) / len(puntos_cluster) for p in zip(*puntos_cluster)]
    
    # Calcular porcentajes redondeando al entero más cercano
    porcentajes_redondeados = [round((contador_puntos[i] / len(datos_seleccionados)) * 100) for i in range(k)]
    
    return asignaciones, centroides, contador_puntos, porcentajes_redondeados

def predecir_cluster_nv(punto, centroides):
    """
    Predice el clúster al que pertenece un nuevo punto (no vectorizado).
    """
    distancias = [distancia_euclidiana(punto, centroide) for centroide in centroides]
    return distancias.index(min(distancias))
