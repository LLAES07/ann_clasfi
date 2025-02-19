Esta aplicación web predice la probabilidad de que un cliente abandone (churn) utilizando una red neuronal entrenada con TensorFlow/Keras. La interfaz interactiva fue desarrollada con Streamlit para facilitar su uso.

## Características
Preprocesamiento completo de datos:

## Eliminación de columnas irrelevantes.
Codificación de variables categóricas (Label Encoding y One-Hot Encoding).
Escalado de características mediante StandardScaler.
Modelo de red neuronal:

## Arquitectura de dos hidden layers y Dropout
Opciones para ajustar el threshold de decisión y manejar el desbalance de clases.
Despliegue en Streamlit:

## Interfaz amigable para cargar datos o ingresar características manualmente.
Visualización de resultados y métricas de predicción en tiempo real.


## Flujo de Trabajo

### Preprocesamiento
- Limpieza y transformación de datos (eliminación de columnas irrelevantes, codificación y escalado).
- Manejo del desbalance de clases a través de técnicas como undersampling o ponderación de clases.

### Entrenamiento del Modelo
- Entrenamiento de una red neuronal simple con dos capas ocultas y dropout.
- Ajuste de hiperparámetros y validación mediante técnicas de EarlyStopping y reducción de la tasa de aprendizaje.

### Despliegue
- Integración del modelo en una aplicación Streamlit para una experiencia interactiva.
