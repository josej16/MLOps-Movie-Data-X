# --------------------------------------------------------------------------------------------------------------------------------------
'''En esta parte procederemos a entretar un modelo de ML capaz de recomendar una pelicula en base a
   pelicula vista, esto mediante la comparacion de score que asignaron usuarios parecidos, es decir
   este sera en este caso un filtro colaborativo item - item.
   
   En este caso utilizaremos de la libreria surprise el modeli KNNWithMeans, el cual sera perfecto
   para nuestro formato de data (user item rating)'''
# --------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
from surprise import Reader, Dataset, KNNWithMeans


'''Se procede a cargar el archivo formato [userId, movieId, score]'''

df_svd = pd.read_csv('Movie_Rating_ML.csv', sep=';', encoding='utf-8')


'''El archivo cargado genera un problema y es el tamaño del mismo.

   Dado que tiene un peso mayor a los 200MB y de 11 millones de registros, por nuestras
   limitaciones de recursos, como internet y potencia de procesamiento, necesitaremos
   tomas una muestra que cumpla con nuestras capacidades, en este caso 1500, generando a futuro un modelo
   de 16MB.
   
   En este caso esta limitacion puede afectar la eficacia de nuestro modelo, pero dado las circunstancias
   sera lo mejor.'''

df_svd = df_svd.sample(n=1500,random_state=33)


'''Dicho eso procedemos.

   Primeros instanciaremos la funcion render que nos servira luego para normalizar el formato de la data,
   el parametro rating_scale nos servira para especificar el rango de calificacion de la columna score.
   
   Luego instanciaremos el modelo KNNWithMeans donde en el parametro sim_options se le especifica que
   no sea basado en usuarios para que funcione con solo como entrada la pelicula vista en vez de el
   usuario y la pelicula, ademas de cosine que especifica el parametro por el cual se verificara el tipo
   de similitud.
   
   Luego usamos el metodo load_from_df para darle el formato requerido al DataFrame'''

reader = Reader(rating_scale=(0,5))
options = {'name':'cosine',
          'user_based': False}
knn = KNNWithMeans(sim_options=options)
data = Dataset.load_from_df(df_svd,reader)


'''Ahora para el modelo KNNWithMeans no trabajo con el formato bruto que retorna el metodo load_from_df
   es decir que debemos mediante build_full_trainset cambiarlo a un formato especifico para KNNWM.
   
   Luego entrenamos.'''

train = data.build_full_trainset()
knn.fit(train)


'''Ahora lo que hacemos para que la API no maneje el dataset de 11 millones de filas en sus limitadas
   capacidades es guardarlo como un archivo pickle, el cual es mas eficiente en espacio y puede ser
   almacenado para su uso posterior.
   
   Lo mismo para el train y su futura decodificacion del movieId'''

import pickle

filename = 'Model_KNNM_Movie.pkl'
with open(filename, 'wb') as file:
    pickle.dump(knn, file)

filename2 = 'train.pkl'
with open(filename2, 'wb') as file:
    pickle.dump(train, file)
