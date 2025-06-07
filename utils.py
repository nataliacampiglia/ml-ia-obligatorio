import os
from tqdm import tqdm
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.decomposition import PCA

def create_images_list(paths):
  suffix = '.pgm'
  pgm_files = []
  images = []
  
  for path in paths:
    if not os.path.isdir(path):
            continue  # Ignorar rutas que no sean directorios

    all_files = os.listdir(path)
    full_paths = [os.path.join(path, f) for f in all_files if f.endswith(suffix)]
    pgm_files.extend(full_paths)
    for filename in tqdm(pgm_files):
      with open(filename, 'rb') as pgmf:
        image = plt.imread(pgmf)
      images.append(image)

  return np.array(images)
    
	
	
                
	