import cv2
import numpy as np
from skimage import feature
from sklearn.preprocessing import StandardScaler

def sliding_window(image, stepSize, windowSize):
    """
    Desliza una ventana sobre la imagen.
    """
    for y in range(0, image.shape[0] - windowSize[1] + 1, stepSize):
        for x in range(0, image.shape[1] - windowSize[0] + 1, stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def detect_faces(input_image, model, scaler, pca, use_hog=True,
                 window_size=(64, 64), step_size=16):
    """
    Detecta rostros en una imagen usando el modelo entrenado.
    """
    # Convertir imagen a escala de grises si es necesario
    if len(input_image.shape) == 3 and input_image.shape[2] == 3:
        gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = input_image.copy()

    bboxes = []
    
    for (x, y, window) in sliding_window(gray, step_size, window_size):
        if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
            continue

        # Preprocesar patch
        patch = window.reshape(1, -1).astype(np.float32)
        patch_std = scaler.transform(patch)
        patch_pca = pca.transform(patch_std)

        if use_hog:
            hog_features = feature.hog(window,
                                       pixels_per_cell=(8, 8),
                                       cells_per_block=(2, 2),
                                       feature_vector=True)
            patch_pca = hog_features.reshape(1, -1)

        prediction = model.predict(patch_pca)
        if prediction == 1:
            bboxes.append((x, y, x + window_size[0], y + window_size[1]))
    
    # Dibujar los bounding boxes
    output_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return output_image, bboxes
