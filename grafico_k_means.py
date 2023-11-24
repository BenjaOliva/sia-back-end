
import matplotlib.pyplot as plt
from k_means import k_means, mm_normalize, predecir_cluster
import numpy as np
from k_medias import predecir_cluster_old
from sklearn import preprocessing

from load_data import cargar_datos_desde_arff
def plot_k_means(datos, asignaciones, centroides, contador_puntos, porcentajes_redondeados, caracteristicas_seleccionadas=None):
    # Visualizar resultados
    colores = ['r', 'g', 'b']
    k = len(centroides)
    
    for i in range(k):
        puntos_cluster = datos[asignaciones == i][:, caracteristicas_seleccionadas]
        label = f'Cluster {i + 1} ({contador_puntos[i]} puntos, {porcentajes_redondeados[i]}%)'
        plt.scatter(puntos_cluster[:, 0], puntos_cluster[:, 1], c=colores[i], label=label)

    # Visualizar centroides
    plt.scatter(centroides[:, 0], centroides[:, 1], c='k', marker='x', label='Centroides')
    plt.xlabel('Característica 1' if caracteristicas_seleccionadas is None else f'Característica {caracteristicas_seleccionadas[0]+1}')
    plt.ylabel('Característica 2' if caracteristicas_seleccionadas is None else f'Característica {caracteristicas_seleccionadas[1]+1}')

    plt.legend()
    plt.show()

# Cargar datos desde el archivo ARFF
archivo_arff = 'k_medias_data/winequality.arff'
datos = cargar_datos_desde_arff(archivo_arff)

# Elegir qué características considerar (puedes ajustar esto según tus necesidades)
caracteristicas_seleccionadas = [0, 1, 2]  # Por ejemplo, selecciona las dos primeras características

# Aplicar k-means con k=2 y las características seleccionadas
k = 2
asignaciones, centroides, contador_puntos, porcentajes_redondeados = k_means(datos, k, caracteristicas_seleccionadas)

print("Centroides: ", centroides)
print("Porcentajes redondeados:", porcentajes_redondeados)

# Predecir cluster de un nuevo punto (puedes ajustar esto según tus necesidades)
nuevo_elemento = np.array([0.30018597, 0.17102719, 0.21146347])
cluster_predicho = predecir_cluster(nuevo_elemento, centroides)
print("Cluster predicho para el nuevo elemento:", cluster_predicho)


plot_k_means(datos, asignaciones, centroides, contador_puntos, porcentajes_redondeados, caracteristicas_seleccionadas)