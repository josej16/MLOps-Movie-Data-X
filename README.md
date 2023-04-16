# API con Sistema de recomendación de películas

Este proyecto utiliza Python para construir un sistema de recomendación de películas utilizando el modelo KNNWithMeans de la librería Surprise, además de varios endpoints para la data de plataformas como Hulu, Amazon Prime Video, Netflix y Disney+. El proyecto utiliza las bibliotecas Pandas y Surprise para la parte de machine learning y Render para construir una API.

## Descripción del proyecto

Este proyecto tiene como objetivo construir varios endpoints y uno basado en un sistema de recomendación de películas por la calificación por similitud del coseno. Para lograrlo, se realiza un proceso de ETL (Extracción, Transformación y Carga) sobre los datos de películas y calificaciones. A continuación, se aplican técnicas de EDA (Exploración de Datos) para analizar y visualizar los datos. Luego se entrena un modelo de machine learning utilizando la biblioteca Surprise, que utiliza algoritmos de filtrado colaborativo para realizar recomendaciones personalizadas. Finalmente, se crea una API con un deploy en Render, que permite a los usuarios obtener información y recomendaciones de películas a través de endpoints.

## Estructura del proyecto

El proyecto se organiza en las siguientes carpetas y archivos:

- `MLOpsCleanData/`: Carpeta que contiene los datos utilizados para el ETL y el entrenamiento del modelo.
- `MLOpsETL/`: Carpeta que contiene los scripts de Python utilizados para realizar el ETL
- `MLOpsML/`: Carpeta que contiene los scripts de Python utilizados para realizar el entrenamiento del modelo KNNWithMeans de la librería Surprise
- `main.py`: Archivo donde se encuentra todo el código de la API y sus endpoints para funcionar
- `requirements.txt`: Archivo que enumera las dependencias de Python necesarias para ejecutar la aplicación.

## Requerimientos

Para ejecutar el proyecto, se necesitan los siguientes requisitos:

- Python 3.7 o superior
- Las bibliotecas listadas en `requirements.txt`
- Los datos de películas y calificaciones en formato CSV

## Proceso

## Extracción de Datos:

El primer paso en el proceso de ETL es la extracción de los datos necesarios. Para este proyecto, se utilizaron 4 conjuntos de datos en formato CSV que contienen información sobre películas en diferentes plataformas de streaming, así como 8 conjuntos de datos adicionales en formato CSV del conjunto de datos en el formato de el famoso dataset de MovieLens, que incluyen información sobre calificaciones de usuarios para cada película.

Los 4 conjuntos de datos de las plataformas de streaming son de Amazon Prime, Disney+, Hulu y Netflix. Cada archivo contiene información como el título, el año de lanzamiento, el género y la calificación por madurez de cada película.

Además de estos conjuntos de datos, se descargaron 8 archivos CSV adicionales del conjunto de datos de MovieLens. Cada archivo contiene información sobre las calificaciones de los usuarios para las películas. Los archivos están estructurados con una columna para cada uno de los siguientes campos: user, item, rating y timestamp.

Una vez descargados los archivos CSV, se realizaron las transformaciones especificadas en la sección siguiente. Estas transformaciones se aplicaron a cada conjunto de datos para asegurar la uniformidad de los datos y para cumplir con los requisitos de formato para el proceso de carga en la base de datos.

## Transformaciones en los datos

Durante la etapa de Extracción, Transformación y Carga (ETL), se llevaron a cabo transformaciones específicas en los datos. Estas transformaciones se realizaron con el objetivo de preparar los datos para el análisis y la construcción del modelo de recomendación. A continuación se detallan las transformaciones aplicadas:

### Generación de campo ID

Se generó un campo ID compuesto por la primera letra del nombre de la plataforma, seguido del `show_id` ya presente en los datasets. Por ejemplo, para los títulos de Amazon, el ID se compone de "as" seguido del `show_id`.

### Unión de Data

Luego de obtener el ID para cada película, se pueden unir los 4 CSV para una transformación más simple.

### Recuperar data

Dado que en la data de Hulu y Netflix principalmente hay un error en el que la información de la columna "duration" se encuentra en la de "rating", fue necesario arreglarlo y cambiar su posición de una columna a otra, dejando como nulo en la columna "rating".

### Reemplazo de valores nulos

Los valores nulos del campo rating se reemplazaron por el string "G", correspondiente al maturity rating: "general for all audiences".

### Formato de fechas

Si se incluían fechas en los datos, se aseguró que estuvieran en formato `AAAA-mm-dd`.

### Formato de texto

Todos los campos de texto se convirtieron a minúsculas, sin excepción.

### Conversión del campo duration

El campo duration se dividió en dos campos (`duration_int` y `duration_type`). El primero es un valor numérico entero que indica la duración en minutos o temporadas, dependiendo del tipo de contenido. El segundo campo, `duration_type`, es un string que indica la unidad de medición de la duración, que puede ser "min" (minutos) o "season" (temporadas).

Estas transformaciones permitieron que los datos fueran consistentes y adecuados para su análisis y uso en la construcción del modelo de recomendación y de la API. Luego se procede a guardarlos en un CSV limpio para consumir.

## Entrenamiento del modelo

Para realizar la recomendación de películas, se utilizó la librería Surprise de Pandas. Se importó el archivo CSV unificado de ratings, el cual solo pudimos usar 1750 registros debido a ciertas limitaciones de espacio en GitHub y de poder de procesamiento gratuito en Render.

Para la creación del modelo de recomendación, se utilizó el algoritmo KNNWithMeans, el cual nos basamos en la similitud del coseno para recomendar películas según la similitud del rating.

Es importante mencionar que para obtener recomendaciones precisas, es necesario pedir recomendaciones de películas que hayan sido calificadas por diferentes usuarios más de una vez, ya que si no recuerre a las mismas recomendaciones.

Una vez generado el modelo, se guardó en formato pickle para que pueda ser utilizado en otro script sin necesidad de volver a entrenarlo, ahorrando tiempo en el proceso de recomendación.

## Desarrollo de la API

En el desarrollo de la API, se utilizaron las bibliotecas FastAPI, Surprise, Pickle y Pandas para crear diferentes endpoints que se consumen en la API. Se cargó el archivo CSV de la data limpia que se unificó previamente, así como los archivos pickle que contienen el modelo entrenado y el trainset para desencriptar el ID de las películas que se encriptaron durante el entrenamiento del modelo.

Se crearon seis funciones que se corresponden con los siguientes requerimientos:

1. Película (sólo película, no serie, etc) con mayor duración según año, plataforma y tipo de duración. La función llamada `get_max_duration(year, platform, duration_type)` devuelve el nombre de la película como un string.

2. Cantidad de películas (sólo películas, no series, etc) según plataforma, con un puntaje mayor a XX en determinado año. La función llamada `get_score_count(platform, scored, year)` devuelve un int que representa el número total de películas que cumplen con los criterios establecidos.

3. Cantidad de películas (sólo películas, no series, etc) según plataforma. La función llamada `get_count_platform(platform)` devuelve un int que representa el número total de películas de esa plataforma. Las plataformas que se utilizaron son Amazon, Netflix, Hulu y Disney+.

4. Actor que más se repite según plataforma y año. La función llamada `get_actor(platform, year)` devuelve el nombre del actor que más se repite según la plataforma y el año especificados como un string.

5. La cantidad de contenidos/productos (películas, series, etc) que se publicó por país y año. La función llamada `prod_per_county(tipo,pais,anio)` devuelve un diccionario con las variables 'pais' (nombre del país), 'anio' (año) y 'pelicula' (cantidad de contenidos/productos) según el tipo de contenido (película o serie).

6. La cantidad total de contenidos/productos (películas, series, etc) según el rating de audiencia dado. La función llamada `get_contents(rating)` devuelve el número total de contenidos con ese rating de audiencias.

7. Para el último endpoint de la API, `get_recomendation(title)`, se ha implementado un sistema de recomendación basado en el modelo de K-Nearest Neighbors With Means(KNNWithMeans) utilizando la librería Surprise de Python. El endpoint recibe como entrada el título de una película y devuelve los 5 vecinos más cercanos basados en la similitud de los ratings de los usuarios.

Los resultados se presentan en un formato `JSON`.


Usamos la plataforma Render que es simple y gratuita, en donde a traves de la libreria uvicorn se puede montar la API, en su respectivo requeriments.txt estan las librerias usadas en el proyecto para que funcione correctamente, para ello tienes el siguiente tutorial https://github.com/HX-FNegrete/render-fastapi-tutorial .

Y con esto dicho Aqui esta la API listo para consumir https://mlops-movie-data-x.onrender.com/ .
