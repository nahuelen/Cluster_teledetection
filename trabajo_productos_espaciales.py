

""" Visualizar la imagen y realizar un análisis estadístico básico de una banda espectral, obtener el histograma y las estadísticas descriptivas.


"""

#importar la biblioteca drive en la notebook
from google.colab import drive
drive.mount('/content/drive')

###Iporto librerias
import numpy as np
from osgeo import gdal

from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import sys, os
ruta_base = '/content/drive/MyDrive/Uba/Maestria datos financieros/Metodos datos no estructurados/Trabajos finales/Bandera_Belgrano/S2_Bandera_2024_03_24_B2_3_4_8_11_12.tif'

ruta = os.path.join(ruta_base)
print(ruta)
img_sentinel2 = gdal.Open(ruta)

gt = img_sentinel2.GetGeoTransform()
src = img_sentinel2.GetProjection()
num_bandas = img_sentinel2.RasterCount
print(num_bandas)##veo numero de bandas
###ojo orden de bandas B2 orden 0, B3 orden 1, B4 orden 2, B8 orden 3,B11 orden 4, B12 orden 5

# Obtener los nombres de las bandas de la imagen sentinel 2
nombres_bandas = []
for i in range(1, num_bandas + 1):
    banda = img_sentinel2.GetRasterBand(i)
    nombres_bandas.append(banda.GetDescription() or f'Banda {i}')
print("Nombres de las bandas:", nombres_bandas)

img_sentinel2_array = img_sentinel2.ReadAsArray()##transformo a array

# Sustituir NaNs por ceros
img_sentinel2_array = np.nan_to_num(img_sentinel2_array, nan=0.0)

dim = img_sentinel2_array.shape##veo dimensiones de la imagen 6 bandas por 1278 px de alto y 2585 px de ancho
print(dim)

def scale(array,p = 0, nodata = None):
    '''
    Esta función escala o estira la imagen a determinado % del histograma (trabaja con percentiles)
    Si p = 0 (valor por defecto) entonces toma el mínimo y máximo de la imagen.
    Devuelve un arreglo nuevo, escalado de 0 a 1
    '''
    a = array.copy()
    a_min, a_max = np.percentile(a[a!=nodata],p), np.percentile(a[a!=nodata],100-p)
    a[a<a_min]=a_min
    a[a>a_max]=a_max
    return ((a - a_min)/(a_max - a_min))

# Función para realizar la combinación y el estiramiento de bandas de la imagen
def get_rgb(array,band_list, p = 0, nodata = None):
    '''
    Esta función toma como parámetros de entrada la matriz a ser ploteada, una lista de índices correspondientes
    a las bandas que queremos usar, en el orden que deben estar (ej: [1,2,3]), y un parámetro
    p que es opcional, y por defecto es 0 (es el estiramiento a aplicar cuando llama a scale()).

    Devuelve una matriz con las 3 bandas escaladas

    Nota: Se espera una matriz con estas dimensiones de entrada: [bandas, filas, columnas]
    '''
    r = band_list[0]
    g = band_list[1]
    b = band_list[2]


    r1 = scale(array[r-1,:,:],p, nodata)
    g1 = scale(array[g-1,:,:],p, nodata)
    b1 = scale(array[b-1,:,:],p, nodata)


    a = np.dstack((r1,g1,b1))
    return a

def plot_rgb(array,band_list, p = 0, nodata = None, figsize = (12,6)):
    '''
    Esta función toma como parámetros de entrada la matriz a ser ploteada, una lista de índices correspondientes
    a las bandas que queremos usar, en el orden que deben estar (ej: [1,2,3]), y un parámetro
    p que es opcional, y por defecto es 0 (es el estiramiento a aplicar cuando llama a get_rgb(), que a su vez llama a scale()).

    Por defecto tambien asigna un tamaño de figura en (12,6), que también puede ser modificado.

    Devuelve solamente un ploteo, no modifica el arreglo original.
    Nota: Se espera una matriz con estas dimensiones de entrada: [bandas, filas, columnas]
    '''
    r = band_list[0]
    g = band_list[1]
    b = band_list[2]

    a = get_rgb(array, band_list, p, nodata)

    plt.figure(figsize = figsize)
    plt.title(f'Combinación {r}, {g}, {b} \n (estirado al {p}%)' , size = 12)
    plt.imshow(a)
    plt.show()

##visualización de la Imagen Sentinel 2 utilizando la combinación de falso color, asignando a los canales RGB las bandas B8, B11 y B4 respectivamente y escalando la imagen con un valor de percentil "p" del 5%.
plot_rgb(img_sentinel2_array,[3,4,2],p=5)

"""3.c) Calcular un índice de vegetación a elección."""

#Cálculo  y visualización del NDVI
b_nir = img_sentinel2_array[3,:,:]
b_red = img_sentinel2_array[2,:,:]

ndvi = (b_nir - b_red) / (b_nir + b_red)
plt.figure(figsize = (12,6))
plt.imshow(ndvi, vmin = 0, vmax = 1, cmap = 'gray')
plt.title("NDVI")
plt.show()

plt.figure(figsize = (12,6))
plt.imshow(ndvi, vmin = 0, vmax = 1, cmap = 'terrain_r')
plt.title("NDVI")
# Add colorbar to show the index
plt.colorbar()
plt.show()

"""3.d) Grafica el histograma de la banda 11. Calcula los valores mínimo, máximo, media y desvío estándar a partir de los valores de los píxeles."""

plt.hist(img_sentinel2_array[4,:,:].ravel(), 50, density=True, color='g', alpha=0.75)
plt.title('Histograma banda 11')
plt.show()

import pandas as pd
banda11=img_sentinel2_array[4,:,:]
min_b11,max_b11,media_b11=np.min(banda11),np.max(banda11),np.mean(banda11)
print(f'Mínimo:{min_b11}, Máximo:{max_b11},Media:{media_b11}')

"""
 Aplico una clasificación no supervisada K-means, especificando el número de clusters que consideres adecuado para diferenciar las coberturas en el área de estudio seleccionada.
"""


# Importación de librerías


import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

import rasterio
import rasterio.mask


from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.mask import mask

import geopandas as gpd
from shapely.geometry import mapping

import pandas as pd
import seaborn as sns


import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep

ruta_base = os.path.join('/content/drive/MyDrive/Uba/Maestria datos financieros/Metodos datos no estructurados/Trabajos finales/Bandera_Belgrano/S2_Bandera_2024_03_24_B2_3_4_8_11_12.tif')
with rasterio.open(ruta_base) as src:
    img = src.read([1,2,3,4,5,6])
    crs = src.crs
    gt = src.transform

# Guardo los datos espectrales, descarto la info espacial.
d,x,y = img.shape
X = img.reshape([d,x*y]).T
print(X.shape)

# Sustituir NaNs por ceros
X = np.nan_to_num(X, nan=0.0)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=36)
kmeans.fit(X)
L = kmeans.labels_
Yimg = L.reshape([x,y])
plt.figure(figsize=(15,10))
plt.title('Clusterizacion')
show(Yimg)
plt.show()
#Yimg_kmeans6 = Yimg

# Cuantifico la superficie (en ha) de cada categoría identificada."""

# Obtener las etiquetas de los clústeres
L = kmeans.labels_

# Contar la cantidad de datos en cada clúster
unique, counts = np.unique(L, return_counts=True)
cluster_counts = dict(zip(unique, counts))

# Mostrar los resultados
print("Cantidad de datos en cada clúster:")
for cluster_id, count in cluster_counts.items():
    print(f"Clúster {cluster_id}: {count} datos")

resolucion_pixel = src.res[0]

# Calcular el área total de los píxeles por cada clúster
area_total_por_cluster = {}
for cluster_id, count in cluster_counts.items():
    area_total = count * resolucion_pixel**2
    area_total_hectareas = area_total / 10000
    area_total_por_cluster[cluster_id] = area_total_hectareas
    print(f"Clúster {cluster_id}: Área total = {area_total_hectareas} hectáreas")


