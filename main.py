import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from k_means import k_means, predecir_cluster
from k_means_no_vectorizado import k_means_no_vectorizado, predecir_cluster_nv
from load_data import cargar_datos_desde_arff

app = FastAPI()

archivo_arff = 'k_medias_data/winequality.arff'

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"], 
)

@app.post('/predecir-cluster')
def predecir_cluster_endpoint(data: dict):

    datos = cargar_datos_desde_arff(archivo_arff)

    nuevo_elemento = np.array(data['nuevo_elemento'], dtype=float)
    features = np.array(data['atributos'], dtype=int).tolist()
    clusters = data['clusters']

    print(nuevo_elemento)
    print(features)
    print(clusters)

    asignaciones, centroides, contador_puntos, porcentajes_redondeados = k_means(datos, clusters, features)
    print("Centroides: ", centroides)

    cluster_predicho = predecir_cluster(nuevo_elemento, centroides)

    return { 
        'cluster_predicho': cluster_predicho.tolist()
     }

@app.post('/predecir-cluster-nv')
def predecir_cluster_endpoint_nv(data: dict):

    datos = cargar_datos_desde_arff(archivo_arff)

    nuevo_elemento = np.array(data['nuevo_elemento'], dtype=float)
    features = np.array(data['atributos'], dtype=int).tolist()
    clusters = data['clusters']

    print(nuevo_elemento)
    print(features)
    print(clusters)

    asignaciones, centroides, contador_puntos, porcentajes_redondeados = k_means_no_vectorizado(datos, clusters, features)
    print("Centroides: ", centroides)

    cluster_predicho = predecir_cluster_nv(nuevo_elemento, centroides)

    return { 
        'cluster_predicho': cluster_predicho
     }

# endpoint POST que devuelva un mensaje con los datos de entrada
@app.post('/k-means-vectorizado')
def return_post_data(data: dict):
    # access posted data 
    print(data)
    datos = cargar_datos_desde_arff(archivo_arff)

    asignaciones, centroides, contador_puntos, porcentajes_redondeados = k_means(datos, data['clusters'], data['atributos'], data['iteraciones'], data['seed_inicial'])

    return { 
        'asignaciones': asignaciones.tolist(),
        'centroides': centroides.tolist(),
        'contador_puntos': contador_puntos.tolist(),
        'porcentajes_redondeados': porcentajes_redondeados.tolist()
     }

@app.post('/k-means-no-vectorizado')
def return_post_data_nv(data: dict):
    # access posted data 
    print(data)
    datos = cargar_datos_desde_arff(archivo_arff)

    asignaciones, centroides, contador_puntos, porcentajes_redondeados = k_means_no_vectorizado(datos, data['clusters'], data['atributos'], data['iteraciones'], data['seed_inicial'])

    return { 
        'asignaciones': asignaciones,
        'centroides': centroides,
        'contador_puntos': contador_puntos,
        'porcentajes_redondeados': porcentajes_redondeados
     }