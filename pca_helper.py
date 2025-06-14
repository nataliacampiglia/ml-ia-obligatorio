import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import feature
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import chain

def extract_hog_features(images, desc='Extrayendo características HOG'):
    """
    Extrae características HOG de una lista de imágenes.
    
    Args:
        images: Lista de imágenes a procesar
        desc: Descripción para la barra de progreso
        
    Returns:
        array numpy con las características HOG
    """
    return np.array([feature.hog(im) for im in tqdm(images, desc=desc)])

def prepare_data(faces, backgrounds, test_size=0.3, random_state=42, n_components=500):
    """
    Prepara los datos para entrenamiento combinando caras y fondos, extrayendo características,
    y dividiendo en conjuntos de entrenamiento y prueba.
    
    Args:
        faces: Lista de imágenes de caras
        backgrounds: Lista de imágenes de fondo
        test_size: Proporción de datos a usar para prueba
        random_state: Semilla aleatoria para reproducibilidad
        
    Returns:
        X_train, X_test, y_train, y_test, scaler, pca
    """
    # Combinar imágenes y crear etiquetas
    # images = list(faces) + list(backgrounds)
    X = extract_hog_features(chain(faces, backgrounds), desc='Construyendo X')
    y = np.zeros(len(X))
    y[:len(faces)] = 1

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Escalar y transformar datos
    scaler = StandardScaler()
    scaler.fit(X_train)  # Primero ajustamos el scaler
    X_train_std = scaler.transform(X_train)  # Luego transformamos los datos
    
    # Aplicar PCA
    pca = PCA(n_components=n_components, whiten=False)
    X_train_pca = pca.fit_transform(X_train_std)
    
    return X_train_pca, X_test, y_train, y_test, scaler, pca

def process_test_images(test_images, scaler, pca):
    """
    Procesa imágenes de prueba usando las mismas transformaciones que los datos de entrenamiento.
    
    Args:
        test_images: Lista de imágenes de prueba
        scaler: StandardScaler ajustado
        pca: PCA ajustado
        
    Returns:
        Datos de prueba transformados listos para predicción
    """
    X_test = extract_hog_features(test_images, desc='Procesando imágenes de prueba')
    X_test_std = scaler.transform(X_test)  # Solo transformamos, no ajustamos
    X_test_pca = pca.transform(X_test_std)
    return X_test_pca

def evaluate_model(model, X_test, y_test, scaler, pca):
    """
    Evalúa el rendimiento del modelo en datos de prueba.
    
    Args:
        model: Modelo entrenado
        X_test: Características de prueba
        y_test: Etiquetas de prueba
        scaler: StandardScaler ajustado
        pca: PCA ajustado
        
    Returns:
        Puntuación de precisión
    """
    X_test_std = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_std)
    y_pred = model.predict(X_test_pca)
    return accuracy_score(y_test, y_pred)

def create_submission(predictions, file_ids, submission_name='submission'):
    """
    Crea un archivo de submission para Kaggle a partir de las predicciones.
    
    Args:
        predictions: Predicciones del modelo (0 o 1)
        file_ids: Lista de IDs de archivo correspondientes a las predicciones
        submission_name: Nombre para el archivo CSV de salida
        
    Returns:
        DataFrame con las predicciones y guarda en CSV
    """
    submission_dict = {file_ids[i]: predictions[i] for i in range(len(file_ids))}
    submission_df = pd.DataFrame(list(submission_dict.items()), columns=['id', 'target_feature'])
    submission_df['id'] = submission_df['id'].astype(int)
    submission_df['target_feature'] = submission_df['target_feature'].astype(int)
    submission_df.sort_values(by='id', inplace=True)
    
    # Guardar en CSV
    submission_df.to_csv(f'{submission_name}.csv', index=False)
    return submission_df 