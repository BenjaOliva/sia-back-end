from scipy.io import arff
import numpy as np

def cargar_datos_desde_arff(ruta):
    """Carga datos desde un archivo ARFF, normaliza y devuelve un array de NumPy."""
    data, meta = arff.loadarff(ruta)
    
    # Convertir los datos de tipo estructurado a un array de NumPy
    data_array = np.array(data.tolist(), dtype=float)

    # Normalizar los datos
    data_array = (data_array - data_array.min(axis=0)) / (data_array.max(axis=0) - data_array.min(axis=0))
    return data_array