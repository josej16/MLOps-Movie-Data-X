# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''Desde este punto termina el proceso de ETL y se empieza con la creacion de la API con el framework
   Fastapi'''
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''Importamos la data limpia a trabajar y el diccionario usado en el ETL para comodidad'''

import pandas as pd

df = pd.read_csv('MLOpsCleanData/MovieCleanData.csv',sep=';',encoding='utf-8')

platforms = {'amazon': 'a','hulu': 'h','disney': 'd','netflix': 'n'}


'''Aqui cargaremos el modelo y train_set en formato pickle para su posterior uso en un decorador'''

import pickle
from surprise import Reader, Dataset, KNNWithMeans

filename = 'MLOpsML/Model_KNNM_Movie.pkl'
filename2 = 'MLOpsML/train.pkl'
with open(filename, 'rb') as file:
    knn = pickle.load(file)

with open(filename, 'rb') as file:
    train = pickle.load(file)

'''Aqui importamos el modulo desde fastapi el modulo FasAPI para iniciar el proceso'''

from fastapi import FastAPI

app = FastAPI(title='Data Movie Fast API',
              description='Data movies form Amazon Prime, Hulu, Disney+ and Netflix')


'''Primero para cada funcion iremos creando su decorador correspondiente, el cual indicara la ruta
   URL y parametros requeridos para consumirla.
   
   En esta ocacion creamos una donde los parametros de entrada seran el año de salida de la pelicula,
   la plataforma y su tipo de duracion, que para este caso solo sera en minutos.
   Con estos datos filtramos el dataset devolviendo los titulos con mayor duracion para dicho año,
   plataforma y tipo de duracion'''

@app.get('/get_max_duration/{year}/{platform}/{duration_type}')
def get_max_duration(year: int, platform: str, duration_type: str):
    answer = df['title'][(df['duration_int'] == df['duration_int'][(
            df['release_year'] == year) 
            & (df['duration_type'] == duration_type)
            & (df['type'] == 'movie')
            & (df['id'].str.startswith( platforms[platform]))].max())
            & (df['id'].str.startswith(platforms[platform]))].values
            
    answer = answer[0]

    return {'pelicula': answer}


'''Aqui empezamos con la funcion que tiene como entrada la plataforma, la calificacion y el año.

   Luego se filtra el DataFrame por plataforma, año de lanzamiento, por tipo pelicula y que sean
   mayores al parametro scored con respecto al del DataFrame'''

@app.get('/get_score_count/{platform}/{scored}/{year}')
def get_score_count(platform: str, scored: float, year: int):
    answer = df[(df['id'].str.startswith(platforms[platform]))
                & (df['release_year'] == year)
                & (df['score'] > scored)
                & (df['type'] == 'movie')]
    answer = len(answer)
    return {
        'plataforma': platform,
        'cantidad': answer,
        'anio': year,
        'score': scored
    }


'''Para la siguiente funcion se toma como entrada la plataforma para luego retornar el numero total
   de peliculas'''

@app.get('/get_count_platform/{platform}')
def get_count_platform(platform: str):
    answer = len(df[(df['id'].str.startswith(platforms[platform]))
                    & (df['type'] == 'movie')])

    return {'plataforma': platform, 'peliculas': answer}


'''get_actor toma como entrada la plataforma y el año para retornar el actor ("cast") que estuvo en mas
   peliculas, esto lo hace filtrando el DataFrame por plataforma, año y que no tenga valores
   vacios.
   
   Luego hace un split de cada celda para obtener una lista con los actores de dicha pelicula.
   
   Ahora fue necesario agregar todas las ocurrencias de los actores en una lista, para su posterior
   values_counts() y asi devolver el primer valor, es decir el que estuvo en mas peliculas '''

@app.get('/get_actor/{platform}/{year}')
def get_actor(platform: str, year: int):
    answer = df['cast'][(df['id'].str.startswith(platforms[platform]))
                        & (df['release_year'] == year)
                        & (pd.isnull(df['cast']) == False)]
    answer = answer.str.split(', ')

    actor_counts = []
    for i in answer:
        for j in i:
            actor_counts.append(j)

    actor_counts = pd.Series(actor_counts)
    actor_counts = actor_counts.value_counts(sort=True)
    try:
        answer = actor_counts.keys()[0]
        occurrences = int(actor_counts[0])
    except:
        answer = ''

    return {'plataforma': platform, 'anio': year, 'actor': answer, 'apariciones': occurrences}

    
'''Para prod_per_county se quizo hacer un servicio que devolviera la cantidad de productos por pais
   en un año especifico, para eso se necesitaba el tipo de producto, el pais y el año para filtrar.
   
   Finalmente retornamos la cantidad de productos por pais en un año especifico'''

@app.get('/prod_per_county/{tipo}/{pais}/{anio}')
def prod_per_county(tipo: str,pais: str,anio: int):
    answer = len(df[(df['type'] == tipo)
                 & (df['country'] == pais)
                 & (df['release_year'] == anio)])

    return {'pais': pais, 'anio': anio, 'peliculas': answer}


'''En esta funcion se requeria que retornara la cantidad de productos con un rating especifico de
   audiencia, para ello se filtro por el parametro rating y luego se retorno la cantidad del mismo'''

@app.get('/get_contents/{rating}')
def get_contents(rating: str):
    answer = len(df[df['rating'] == rating])

    return {'rating': rating, 'contenido': answer}




@app.get('/get_recomendation/{title}')
def get_recomendation(title):
    movie_id_inner = train.to_inner_iid(str(df['id'][df['title'] == title]))
    recomended_movies = knn.get_neighbors(movie_id_inner, k=5)
    answer = []
    for i in recomended_movies:
        movie = train.to_raw_iid(i)
        answer.append(df['title'][df['id'] == movie])

    return {'recomendacion':answer}








