#!/usr/bin/env python3
"""
Script para generar fondos con transformaciones adicionales
Incluye rotaciones, recortes y zoom para aumentar la diversidad del dataset
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import data, color
from skimage.transform import resize, rotate
from skimage.util import crop
from sklearn.feature_extraction.image import PatchExtractor
from random import random, uniform, randint
import imageio

# Configuración
BACKGROUND_PATH = 'fondos_pgm2'
SIZE = (64, 64)
SCALES = [0.1, 0.25, 0.5, 0.75, 1]
TRANSFORMATIONS = ['rotation', 'crop', 'zoom']
TRANSFORM_PROB = 0.4

def extract_patches(img, N, scale=1.0, patch_size=SIZE):
    """Extrae parches de una imagen con diferentes escalas"""
    extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))
    extractor = PatchExtractor(patch_size=extracted_patch_size, max_patches=N, random_state=0)
    patches = extractor.transform(img[np.newaxis])
    
    if scale != 1:
        patches = np.array([resize(patch, patch_size) for patch in patches])
    
    return patches

def apply_rotation(img, angle_range=(-30, 30)):
    """Aplica una rotación aleatoria a la imagen"""
    angle = uniform(angle_range[0], angle_range[1])
    return rotate(img, angle, mode='reflect', preserve_range=True)

def apply_crop(img, crop_ratio=0.1):
    """Aplica un recorte aleatorio a la imagen"""
    h, w = img.shape
    crop_h = int(h * crop_ratio)
    crop_w = int(w * crop_ratio)
    
    top = randint(0, crop_h)
    bottom = randint(0, crop_h)
    left = randint(0, crop_w)
    right = randint(0, crop_w)
    
    cropped = crop(img, ((top, bottom), (left, right)))
    return resize(cropped, (h, w))

def apply_zoom(img, zoom_range=(0.8, 1.2)):
    """Aplica un zoom aleatorio a la imagen"""
    zoom_factor = uniform(zoom_range[0], zoom_range[1])
    h, w = img.shape
    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
    
    zoomed = resize(img, (new_h, new_w))
    
    if zoom_factor > 1:
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        return zoomed[start_h:start_h+h, start_w:start_w+w]
    else:
        result = np.zeros((h, w))
        start_h = (h - new_h) // 2
        start_w = (w - new_w) // 2
        result[start_h:start_h+new_h, start_w:start_w+new_w] = zoomed
        return result

def apply_transformations(img, transformations=TRANSFORMATIONS, prob=0.5):
    """Aplica transformaciones aleatorias a la imagen"""
    result = img.copy()
    
    for transform in transformations:
        if random() < prob:
            if transform == 'rotation':
                result = apply_rotation(result)
            elif transform == 'crop':
                result = apply_crop(result)
            elif transform == 'zoom':
                result = apply_zoom(result)
    
    return result

def extract_patches_with_transformations(img, N, scale=1.0, patch_size=SIZE, 
                                       transformations=TRANSFORMATIONS, 
                                       transform_prob=TRANSFORM_PROB):
    """Extrae parches y aplica transformaciones adicionales"""
    patches = extract_patches(img, N, scale, patch_size)
    
    transformed_patches = []
    for patch in patches:
        transformed_patches.append(patch)
        
        if random() < transform_prob:
            transformed = apply_transformations(patch, transformations, prob=0.7)
            transformed_patches.append(transformed)
    
    return np.array(transformed_patches)

def load_images():
    """Carga las imágenes de fondo"""
    print("Cargando imágenes de fondo...")
    
    # Imágenes de sklearn
    imgs = ['text', 'coins', 'moon', 'page', 'clock', 
            'immunohistochemistry', 'chelsea', 'coffee', 'hubble_deep_field']
    
    images = []
    for name in imgs:
        img = getattr(data, name)()
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = color.rgb2gray(img)
        images.append(resize(img, (100, 100)))
    
    # Imágenes locales
    for i in range(31):
        filename = f'Background/{i}.jpg'
        if os.path.exists(filename):
            img = plt.imread(filename)
            img = color.rgb2gray(img)
            images.append(resize(img, (100, 100)))
    
    print(f"Cargadas {len(images)} imágenes")
    return images

def generate_backgrounds():
    """Genera los fondos con transformaciones"""
    print("Generando fondos con transformaciones...")
    
    images = load_images()
    
    negative_patches = []
    for im in tqdm(images, desc='Procesando imágenes'):
        for scale in SCALES:
            patches = extract_patches_with_transformations(
                im, 64, scale, SIZE, TRANSFORMATIONS, TRANSFORM_PROB
            )
            negative_patches.extend(patches)
    
    negative_patches = np.array(negative_patches)
    print(f"Forma final de los parches: {negative_patches.shape}")
    print(f"Número total de fondos generados: {len(negative_patches)}")
    
    return negative_patches

def save_backgrounds(negative_patches):
    """Guarda los fondos generados"""
    print(f"Guardando {len(negative_patches)} fondos en {BACKGROUND_PATH}...")
    
    os.makedirs(BACKGROUND_PATH, exist_ok=True)
    
    for i, patch in enumerate(tqdm(negative_patches, desc='Guardando fondos')):
        if patch.ndim == 3:
            patch = patch[:, :, 0]
        patch_uint8 = (patch * 255).astype(np.uint8)
        imageio.imwrite(f"{BACKGROUND_PATH}/b_{i:04}.pgm", patch_uint8)
    
    print(f"¡Completado! Se generaron {len(negative_patches)} fondos con transformaciones.")

def show_statistics(negative_patches):
    """Muestra estadísticas de los fondos generados"""
    print("\n=== ESTADÍSTICAS DE LOS FONDOS GENERADOS ===")
    print(f"Total de fondos: {len(negative_patches)}")
    print(f"Forma de cada fondo: {negative_patches[0].shape}")
    print(f"Rango de valores: [{negative_patches.min():.3f}, {negative_patches.max():.3f}]")
    print(f"Valor medio: {negative_patches.mean():.3f}")
    print(f"Desviación estándar: {negative_patches.std():.3f}")

def demo_transformations():
    """Demuestra las transformaciones aplicadas"""
    print("Generando demostración de transformaciones...")
    
    # Cargar una imagen de ejemplo
    sample_img = getattr(data, 'coins')()
    if len(sample_img.shape) == 3 and sample_img.shape[2] == 3:
        sample_img = color.rgb2gray(sample_img)
    sample_img = resize(sample_img, (100, 100))
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    # Original
    axes[0, 0].imshow(sample_img, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Transformaciones individuales
    axes[0, 1].imshow(apply_rotation(sample_img), cmap='gray')
    axes[0, 1].set_title('Rotación')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(apply_crop(sample_img), cmap='gray')
    axes[0, 2].set_title('Recorte')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(apply_zoom(sample_img), cmap='gray')
    axes[0, 3].set_title('Zoom')
    axes[0, 3].axis('off')
    
    # Combinaciones
    axes[1, 0].imshow(apply_transformations(sample_img, ['rotation', 'crop']), cmap='gray')
    axes[1, 0].set_title('Rotación + Recorte')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(apply_transformations(sample_img, ['rotation', 'zoom']), cmap='gray')
    axes[1, 1].set_title('Rotación + Zoom')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(apply_transformations(sample_img, ['crop', 'zoom']), cmap='gray')
    axes[1, 2].set_title('Recorte + Zoom')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(apply_transformations(sample_img, ['rotation', 'crop', 'zoom']), cmap='gray')
    axes[1, 3].set_title('Todas las transformaciones')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('transformaciones_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Demostración guardada como 'transformaciones_demo.png'")

def main():
    """Función principal"""
    print("=== GENERADOR DE FONDOS CON TRANSFORMACIONES ===")
    print(f"Configuración:")
    print(f"- Tamaño de fondo: {SIZE}")
    print(f"- Escalas: {SCALES}")
    print(f"- Transformaciones: {TRANSFORMATIONS}")
    print(f"- Probabilidad de transformación: {TRANSFORM_PROB}")
    print()
    
    # Demostración de transformaciones
    demo_transformations()
    
    # Generar fondos
    negative_patches = generate_backgrounds()
    
    # Guardar fondos
    save_backgrounds(negative_patches)
    
    # Mostrar estadísticas
    show_statistics(negative_patches)
    
    print("\n¡Proceso completado exitosamente!")

if __name__ == "__main__":
    main() 