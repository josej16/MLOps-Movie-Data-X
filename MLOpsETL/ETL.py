import pandas as pd
from numpy import nan

'''Importamos la data para cada plataforma de contenido'''

amazon_data = pd.read_csv('MLOpsReviews/amazon_prime_titles.csv',sep=',',encoding='utf-8')

netflix_data = pd.read_csv('MLOpsReviews/netflix_titles.csv', sep=',', encoding='utf-8')

disney_data = pd.read_csv('MLOpsReviews/disney_plus_titles.csv', sep=',', encoding='utf-8')

hulu_data = pd.read_csv('MLOpsReviews/hulu_titles.csv', sep=',', encoding='utf-8')


'''Para el resto del proceso de etl nos sera muy util el diccionario platforms,
   con el cual nos ahorraremos varias lineas de codigo'''

platforms = {'amazon': 'a','hulu': 'h','disney': 'd','netflix': 'n'}


'''Ahora se procedera a crear un ID diferenciado para cada plataforma para su posterior union,
   esto nos ayudara para evitar colisiones de ID parecidos entre plataformas y ademas su correcta
   coincidencia con el futuro dataset de rating'''

def CreateId(data,platform):
    if 'id' not in data.columns:
        try:
            data['id'] = platforms[platform] + data['show_id']
        except:
            print('Error: data has no atribute show_id')

CreateId(amazon_data,'amazon')
CreateId(netflix_data,'netflix')
CreateId(disney_data,'disney')
CreateId(hulu_data,'hulu')


'''Se concatenan los DataFrames ya que tienen el mismo formato y nos facilitara la labor en un futuro'''

df = pd.concat((amazon_data,netflix_data,disney_data, hulu_data))
df = df.reset_index(drop=True)


'''La siguiente funcion es necesaria para arreglar un error en la data, principalmente de Hulu, donde los valores
   de "rating" tienen algunos los valores de "duration", por lo tanto hay que intercambiarlos.
   
   Recorreremos la columna rating con la funcion enumerate, ya que necesitaremos el numero index del dato a manejar
   para poder sustituirlo en esa misma fila en la columna "duration" y colocar como nulo en "rating".
   
   para eso nos fijamos en el patron de la columna duration en el cual al hacer el split su ultimo valor debera
   ser una medida de tiempo entre ellas min, season y seasons, de ser asi se intercambian los valores.'''

def RatingtoDuration(data):
    for index, r_value in enumerate(data['rating']):
        if not pd.isnull(r_value):
            aux = r_value.split(' ')

            if aux[-1] in ['min','Season','Seasons']:
                data['duration'][index] = r_value

                data['rating'][index] = nan

RatingtoDuration(df)


'''Esta funcion toma como input el DataFrame unido y en dado caso de que posea datos vacios en la columna
   "rating" los sustituye por la calificacion "G" que corresponde a maturity rating:
   "general for all audiences"'''

def FillNan(data):
    if len(data['rating'][pd.isnull(data['rating']) == True]) != 0:
        data['rating'][pd.isnull(data['rating']) == True] = 'G'
    elif 'rating' in data.columns:
        print('already done')

FillNan(df)


'''Esta funcion normaliza la columna "date_added" del DataFrame al formato de fecha AAAA-mm-dd'''

def NormalizeDate(data):
    data['date_added'] = pd.to_datetime(data['date_added'],yearfirst=True,errors='ignore')

NormalizeDate(df)


'''La siguiente funcion transforma a minusculas todos los datos de las columnas tipo object
   que en general se maneja como string'''

def LowerString(data):
    for i in data.columns:
        if data[i].dtype == 'object':
            data[i] = data[i].str.lower()

LowerString(df)


'''Esta funcion crea dos nuevas columnas a partir de un split en la columna "duration", creando dos,
   una con la cantidad de tiempo y otra con el tipo de tiempo, ya sea "min","season", etc...'''

def SplitDuration(data):
    if ('duration_int' not in data.columns) and ('duration_type' not in data.columns):
        data[['duration_int','duration_type']] = data['duration'].str.split(' ', 1, expand=True)

        data['duration_int'][pd.isnull(data['duration_int']) == True] = pd.to_numeric(data['duration_int'][pd.isnull(data['duration_int']) != True]
                                                                                      ,downcast='integer').mean()

        data['duration_int'] = data['duration_int'].astype('int64',errors='ignore')
    else:
        print('already exists "duration_int, duration_type"')

SplitDuration(df)


'''Aqui necesitaremos cargar nueva data para un servicio de nuestra api, la cual contiene informacion
   sobre el rating de los usuarios hacia las peliculas en concreto'''

score1 = pd.read_csv('MLOpsReviews/ratings/1.csv')
score2 = pd.read_csv('MLOpsReviews/ratings/2.csv')
score3 = pd.read_csv('MLOpsReviews/ratings/3.csv')
score4 = pd.read_csv('MLOpsReviews/ratings/4.csv')
score5 = pd.read_csv('MLOpsReviews/ratings/5.csv')
score6 = pd.read_csv('MLOpsReviews/ratings/6.csv')
score7 = pd.read_csv('MLOpsReviews/ratings/7.csv')
score8 = pd.read_csv('MLOpsReviews/ratings/8.csv')

rating_score = pd.concat((score1,score2,score3,score4,score5,score6,score7,score8))


'''En estas transformaciones renombramos la columna score para evitar problemas al unir dicha columna
   al DataFrame principal.
   
   Luego sacamos el promedio de calificacion por pelicula redondeado a 2 decimales para posteriormente
    agregarla al DataFrame'''

rating_score.rename(columns={'rating': 'score'},inplace=True)
movie_rating = rating_score[['score','movieId']].groupby('movieId').agg('mean').round(2)
df = pd.merge(left=df,right=movie_rating, left_on='id',right_on='movieId', how='left')
# df.to_csv('MLOps-Movie-Data\MLOpsCleanData\MovieCleanData.csv',sep=';',encoding='utf-8',index=False)


'''Ya que nos aproximamos a el apartado de ML seria buena idea arreglar el dataset rating_score para
   dejarle el formato deseado para la funcion reader de la libreria surprise.
   
   Para ello procuramos con el metodo to_numeric de pandas pasarlos al valor mas bajo posible de integer
   y float para salvar un poco de espacio en este dataset de 11 millones de registros.
   
   Finalizamos guardandolo para su posterior uso en la parte de ML.'''

df_knn = rating_score[['userId','movieId','score']]
df_knn['userId'] = pd.to_numeric(df_knn['userId'],downcast='integer')
df_knn['score'] = pd.to_numeric(df_knn['score'],downcast='float')
# df_knn.to_csv('MLOps-Movie-Data\MLOpsML\Movie_Rating_ML.csv', sep=';', encoding='utf-8', index=False)