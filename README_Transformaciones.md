# Generador de Fondos con Transformaciones

Este proyecto incluye funcionalidades mejoradas para generar fondos aplicando transformaciones como rotaciones, recortes y zoom, lo que permite aumentar significativamente la diversidad del dataset de fondos.

## Archivos Creados

1. **`Generar_Fondos_Mejorado.ipynb`** - Notebook Jupyter con todas las funcionalidades
2. **`generar_fondos_transformaciones.py`** - Script de Python independiente
3. **`README_Transformaciones.md`** - Este archivo de documentación

## Transformaciones Implementadas

### 1. Rotación
- **Rango**: -30° a +30° (configurable)
- **Efecto**: Rota la imagen de forma aleatoria
- **Uso**: Aumenta la variabilidad de orientación

### 2. Recorte
- **Ratio**: 10% del tamaño (configurable)
- **Efecto**: Recorta bordes aleatorios y redimensiona
- **Uso**: Simula diferentes puntos de vista

### 3. Zoom
- **Rango**: 0.8x a 1.2x (configurable)
- **Efecto**: Aplica zoom in/out aleatorio
- **Uso**: Simula diferentes distancias de la cámara

## Cómo Usar

### Opción 1: Notebook Jupyter
```bash
jupyter notebook Generar_Fondos_Mejorado.ipynb
```

### Opción 2: Script de Python
```bash
python generar_fondos_transformaciones.py
```

## Configuración

Puedes modificar los parámetros en el script:

```python
# Configuración actual
SIZE = (64, 64)                    # Tamaño de los fondos
SCALES = [0.1, 0.25, 0.5, 0.75, 1] # Escalas de extracción
TRANSFORMATIONS = ['rotation', 'crop', 'zoom']  # Tipos de transformación
TRANSFORM_PROB = 0.4               # Probabilidad de aplicar transformaciones
```

## Parámetros de Transformación

### Rotación
```python
def apply_rotation(img, angle_range=(-30, 30)):
    # angle_range: tupla con rango de ángulos en grados
```

### Recorte
```python
def apply_crop(img, crop_ratio=0.1):
    # crop_ratio: proporción del tamaño a recortar (0.1 = 10%)
```

### Zoom
```python
def apply_zoom(img, zoom_range=(0.8, 1.2)):
    # zoom_range: tupla con factores de zoom (0.8 = 80%, 1.2 = 120%)
```

## Resultados Esperados

### Comparación con Método Original
- **Fondos originales**: ~12,800 fondos
- **Fondos con transformaciones**: ~18,000-25,000 fondos
- **Incremento**: 40-95% más fondos
- **Factor de aumento**: 1.4x - 2x

### Características de los Fondos Generados
- **Formato**: PGM (Portable Gray Map)
- **Tamaño**: 64x64 píxeles
- **Rango de valores**: 0-255 (uint8)
- **Ubicación**: Carpeta `fondos_pgm2/`

## Ventajas de las Transformaciones

1. **Mayor Diversidad**: Cada transformación crea variaciones únicas
2. **Robustez**: El modelo se vuelve más robusto a variaciones
3. **Data Augmentation**: Técnica estándar en machine learning
4. **Escalabilidad**: Fácil de ajustar y extender

## Personalización

### Agregar Nuevas Transformaciones
```python
def apply_new_transformation(img, param=value):
    # Tu nueva transformación aquí
    return transformed_img

# Agregar a la lista de transformaciones
TRANSFORMATIONS = ['rotation', 'crop', 'zoom', 'new_transformation']
```

### Modificar Probabilidades
```python
# Cambiar probabilidad global
TRANSFORM_PROB = 0.6  # 60% de probabilidad

# Cambiar probabilidad por transformación
def apply_transformations(img, transformations, prob=0.7):  # 70% por transformación
```

## Ejemplos de Uso

### Solo Rotaciones
```python
TRANSFORMATIONS = ['rotation']
TRANSFORM_PROB = 0.5
```

### Solo Recortes y Zoom
```python
TRANSFORMATIONS = ['crop', 'zoom']
TRANSFORM_PROB = 0.3
```

### Transformaciones Agresivas
```python
# Rotaciones más extremas
def apply_rotation(img, angle_range=(-45, 45)):

# Recortes más grandes
def apply_crop(img, crop_ratio=0.2):

# Zoom más extremo
def apply_zoom(img, zoom_range=(0.6, 1.4)):
```

## Troubleshooting

### Error: "No module named 'skimage'"
```bash
pip install scikit-image
```

### Error: "No module named 'tqdm'"
```bash
pip install tqdm
```

### Error: "Background folder not found"
- Asegúrate de que existe la carpeta `Background/` con las imágenes `.jpg`
- O modifica la función `load_images()` para usar solo imágenes de sklearn

### Fondos muy similares
- Aumenta `TRANSFORM_PROB`
- Modifica los rangos de transformación
- Agrega más tipos de transformación

## Integración con el Proyecto Principal

Los fondos generados se guardan en formato compatible con tu proyecto existente:

```python
# Usar los nuevos fondos
from constants import BACKGROUND_PATH
BACKGROUND_PATH = 'fondos_pgm2'  # Cambiar a la nueva carpeta
```

## Rendimiento

- **Tiempo estimado**: 2-5 minutos (depende del hardware)
- **Memoria**: ~500MB-1GB para 25,000 fondos
- **Espacio en disco**: ~100MB para 25,000 fondos PGM

## Contribuciones

Para agregar nuevas transformaciones:

1. Implementa la función de transformación
2. Agrega la transformación a la lista `TRANSFORMATIONS`
3. Actualiza la función `apply_transformations()`
4. Documenta los parámetros y efectos

## Licencia

Este código es parte del proyecto ML-IA Obligatorio y sigue las mismas condiciones de licencia. 