import numpy as np

def distancia_euclidiana(punto, centroides):
    """
    Calcula las distancias euclidianas entre un punto y los centroides.
    """
    return np.linalg.norm(punto - centroides)

def asignar_clusters(datos, centroides):
    """
    Asigna cada punto al clúster del centroide más cercano.
    """
    distancias = np.linalg.norm(datos - centroides[:, np.newaxis], axis=2)
    return np.argmin(distancias, axis=0)

def k_means(datos, k, features=None, max_iter=500, seed=100):

    if features is not None:
        # Seleccionar solo las características especificadas
        datos_seleccionados = datos[:, features]
    else:
        # Utilizar todos los atributos si no se especifican características seleccionadas
        datos_seleccionados = datos

    np.random.seed(seed)  # O cualquier otra semilla que desees utilizar

    # Inicializar centroides de forma aleatoria
    centroides = datos_seleccionados[np.random.choice(len(datos_seleccionados), k, replace=False)]
    
    # Contador para el número de puntos asignados a cada clúster
    contador_puntos = np.zeros(k)
    
    for _ in range(max_iter):
        # Asignar cada punto al clúster del centroide más cercano
        asignaciones = asignar_clusters(datos_seleccionados, centroides)
        
        # Actualizar centroides
        for i in range(k):
            puntos_cluster = datos_seleccionados[asignaciones == i]
            centroides[i] = np.mean(puntos_cluster, axis=0)
            contador_puntos[i] = len(puntos_cluster)
    
    # Calcular porcentajes redondeando al entero más cercano
    porcentajes_redondeados = np.round((contador_puntos / len(datos_seleccionados)) * 100).astype(int)
    
    return asignaciones, centroides, contador_puntos, porcentajes_redondeados

def predecir_cluster(punto, centroides):
    """
    Predice el clúster al que pertenece un nuevo punto.
    """
    distancias = [distancia_euclidiana(punto, centro) for centro in centroides]
    return np.argmin(distancias)

