# Clasificación de noticias falsas con énfasis en el ámbito político en español con modelos clásicos, secuenciales y Transformers

Este repositorio contiene la implementación de un sistema de clasificación de noticias falsas en español enfocadas en política.  
Se comparan tres familias de modelos: modelos clásicos basados en TF-IDF, modelos secuenciales con embeddings Word2Vec y un modelo Transformer en español (BETO) ajustado mediante fine-tuning.  

El objetivo principal es evaluar qué enfoque ofrece mejor rendimiento en la tarea de detección de noticias falsas, partiendo de un mismo conjunto de datos y de un pipeline de preprocesamiento unificado.

<br>

## Dataset

El proyecto utiliza el conjunto de datos público de noticias en español disponible en Kaggle:  

https://www.kaggle.com/datasets/arseniitretiakov/noticias-falsas-en-espaol?select=onlytrue1000.csv  

Se trata de un corpus de noticias en español etiquetadas como verdaderas o falsas. En este proyecto se trabaja con un total aproximado de 2 000 instancias, empleadas para entrenar y evaluar los distintos modelos de clasificación binaria.  

El preprocesamiento incluye limpieza del texto, normalización, eliminación de ruido y preparación de las entradas para cada tipo de modelo (TF-IDF, embeddings Word2Vec y tokens para BETO).

<br>

## Estructura del repositorio

La organización principal del código es la siguiente:

```text
Proyecto/
    Archivos/
        Modelos/
            Clasicos/
                busquedas.py
                train.py
                visual.py

            Secuenciales/
                train.py
                visual.py

            Transformers/
                Transformers.ipynb

        BETO.py
        TF-IDF.py
        loader.py
        preprocesamiento.py
        visuales_td.py
        word2vec.py

Data/
    (dataset y salidas de cada modelo: modelos entrenados, gráficas y resultados)
requirements.txt
README.md

```
## Descripción general de los módulos más relevantes:


**loader.py** se encarga de centralizar las rutas, cargar el dataset desde la carpeta Data y generar las particiones de entrenamiento y validación.<br>
**preprocesamiento.py** implementa el pipeline de limpieza de texto que se aplica de forma coherente a todos los modelos.<br>
**TF-IDF.py** construye y guarda el vectorizador TF-IDF y genera las matrices de características que consumen los modelos clásicos.<br>
**word2vec.py** entrena o carga embeddings Word2Vec y produce representaciones vectoriales de las noticias para los modelos secuenciales.<br>
**BETO.py** concentra utilidades y configuraciones específicas asociadas al modelo BETO.<br>
**visuales_td.py** genera visualizaciones y comparativas globales a partir de las métricas de los diferentes modelos.

En la carpeta Modelos se encuentran los scripts específicos de cada familia.<br>
En **Modelos/Clasicos** se realiza la búsqueda de hiperparámetros, el entrenamiento de SVM, regresión logística y Random Forest con TF-IDF, y la generación de gráficas y reportes.<br>
En **Modelos/Secuenciales** se entrenan y evalúan los modelos LSTM y BiLSTM a partir de las representaciones Word2Vec.<br>
En **Modelos/Transformers** ahora se concentra todo el flujo de trabajo de BETO en un único notebook Jupyter, que incluye preprocesamiento específico, definición del modelo, entrenamiento, evaluación, generación de gráficas y exportación de métricas a archivos CSV.<br>

<br>
 
## Requisitos e instalación

Este proyecto fue desarrollado utilizando Python 3.12.8. Se recomienda ejecutar el trabajo dentro de un entorno virtual, con el fin de mantener aisladas las dependencias y evitar conflictos entre versiones, especialmente debido al uso conjunto de bibliotecas como TensorFlow, PyTorch, scikit-learn, gensim y Hugging Face Transformers.

Todas las dependencias necesarias se encuentran concentradas en el archivo requirements.txt. Dicho archivo constituye la referencia oficial que recoge las versiones exactas empleadas en el entorno del proyecto, por lo que debe utilizarse como base para cualquier intento de reproducción o reinstalación del entorno de trabajo.<br>

El entrenamiento del modelo Transformer basado en BETO se llevó a cabo desde Google Colab, aprovechando la disponibilidad de aceleración por GPU y evitando incompatibilidades locales en la configuración. El notebook presente en Proyecto/Archivos/Modelos/Transformers está preparado para ejecutarse desde Colab sin requerir instalación manual adicional, mientras que el resto de los modelos (clásicos y secuenciales) pueden ejecutarse en un entorno local siempre que se mantengan las dependencias indicadas y se respete la estructura del repositorio.

<br>

## Ejecución de los modelos

Antes de ejecutar cualquiera de los modelos es necesario contar con el dataset dentro de la carpeta Data y verificar que las rutas definidas en loader.py correspondan correctamente a la ubicación real del archivo. Una vez confirmado lo anterior, cada enfoque se ejecuta desde su respectiva ubicación dentro del repositorio, respetando el flujo de trabajo previsto para cada arquitectura.
<br>
Los modelos clásicos basados en TF-IDF se encuentran dentro de Proyecto/Archivos/Modelos/Clasicos. El archivo responsable del entrenamiento es train.py, mientras que la evaluación y generación de resultados se realiza desde visual.py. Al finalizar su ejecución, se generan métricas, archivos auxiliares y salidas asociadas dentro de la carpeta Data o en las rutas definidas en el código.
<br>
Los modelos secuenciales con embeddings Word2Vec se ejecutan desde Proyecto/Archivos/Modelos/Secuenciales. El archivo train.py contiene el proceso de entrenamiento de redes tipo LSTM o BiLSTM, y visual.py se encarga del análisis posterior. Este flujo produce métricas, reportes y representaciones vectoriales asociadas al modelo entrenado, las cuales se almacenan dentro de las carpetas internas de Data.
<br>
El modelo Transformer basado en BETO no se ejecuta desde la terminal local sino desde Google Colab, utilizando el notebook disponible en Proyecto/Archivos/Modelos/Transformers. Dicho notebook contiene el flujo completo: carga del dataset, preprocesamiento específico, configuración del modelo, ajuste fino (fine-tuning), métricas finales, gráficas y generación de archivos CSV. Este cuaderno está preparado para ejecutarse en Colab sin necesidad de configuraciones adicionales, aprovechando la aceleración por GPU para el entrenamiento.
<br>
En todos los casos se recomienda mantener la organización de carpetas, el entorno virtual para las dependencias y la versión de Python especificada previamente (Python 3.12.8), con el fin de asegurar compatibilidad y reproducibilidad en los experimentos.

## Optimización y selección de parámetros

Cada modelo implementado en este proyecto cuenta con su propio proceso de ajuste u optimización de parámetros. En los **modelos clásicos** basados en TF-IDF se emplea la búsqueda sistemática de configuraciones mediante funciones internas dedicadas a la selección de hiperparámetros. En los **modelos secuenciales** con embeddings Word2Vec se realiza la configuración y ajuste de arquitecturas recurrentes mediante prácticas de experimentación controlada. Para el caso del **modelo Transformer** basado en BETO, el ajuste fino (fine-tuning) se realiza dentro del notebook diseñado para Google Colab, ejecutando el proceso completo de entrenamiento, validación y almacenamiento de resultados.  

Debido a la naturaleza comparativa del proyecto, cada modelo genera resultados y métricas propias, pero se decidió no incluir una tabla completa dentro de este archivo. Dichos resultados pueden consultarse directamente en las salidas generadas, ya sea en formato CSV, métricas registradas, o durante la ejecución del notebook en Google Colab.

##  Autor
**Autor**: Axel Jair Segundo León<br>
**Dataset**: https://www.kaggle.com/datasets/arseniitretiakov/noticias-falsas-en-espaol

Este proyecto se desarrolló con fines académicos y puede servir como base para experimentos adicionales de clasificación de texto en español y comparación de arquitecturas.
