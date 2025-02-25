# **Análisis y Modelado de Datos para Predicción de Cantidades de Mercadeo**

## **Descripción del Proyecto**
Este proyecto se centra en el análisis, modelado y evaluación de modelos de Machine Learning para la predicción de cantidades de mercadeo. Se implementaron técnicas avanzadas de transformación de datos, ingeniería de características y manejo de outliers para mejorar la calidad del conjunto de datos. Se compararon distintos modelos de aprendizaje automático, incluyendo XGBoost, Random Forest y LightGBM, con el objetivo de identificar el más preciso y eficiente.

Además, se utilizó MLflow para el rastreo de experimentos, almacenamiento de métricas y análisis comparativo de modelos. Se desarrolló un dashboard en Power BI para la visualización de resultados y la interpretación de las predicciones.

## **Objetivos Clave**
- **Realizar un anailisis exploratorio de datos**: xaminar y comprender los datos mediante un análisis detallado y estructurado.
- **Exploración y Transformación de Datos:** Implementar técnicas de preprocesamiento, detección y tratamiento de outliers, y creación de nuevas variables para optimizar la calidad de los datos y de los modelos
- **Entrenamiento y Evaluación de Modelos:** Comparar diferentes algoritmos de Machine Learning, como XGBoost, Random Forest y LightGBM, evaluando su desempeño mediante métricas relevantes.
- **Seguimiento de Experimentos con MLflow:** Registrar ejecuciones de modelos, visualizar métricas y analizar el impacto de los hiperparámetros.
- **Visualización y Comunicación de Resultados:** Presentar los hallazgos de manera clara mediante gráficos en Power BI y herramientas de visualización en Python.

## **Tecnologías Utilizadas**
- **Python:** Lenguaje principal para el procesamiento de datos y modelado.
- **Pandas:** Manipulación y limpieza de datos.
- **Scikit-learn:** Implementación y evaluación de modelos de Machine Learning.
- **XGBoost, Random Forest y LightGBM:** Algoritmos de aprendizaje automático para la predicción de cantidades de mercadeo.
- **MLflow:** Registro y análisis comparativo de modelos.
- **Docker y PostgreSQL:** Uso de un servidor local para la gestión de datos.

## **Estructura del Proyecto**
- **`modelos/`** → Código de los modelos de Machine Learning implementados.
- **`evaluaciones/`** → Código para la evaluación y métricas de los modelos.
- **`analisis/`** → Funciones de visualización utilizadas en el análisis exploratorio (EDA) y en la evaluación de modelos.
- **`imagenes/`** → Capturas del proceso de experimentación en MLflow.
- **`modelo.ipynb`** → Desarrollo y análisis detallado del tercer punto.
- **`transformaciondatos/`** → Contiene scripts para:
  - **Detección y manejo de outliers.**
  - **Ingeniería de características (creación de nuevas variables y One-Hot Encoding).**
- **`mlflowxgboost.py`** → Script para la ejecución y carga del modelo XGBoost en MLflow, incluyendo hiperparámetros y métricas.
- **`mlflowrandomforest.py`** → Script para la ejecución y carga del modelo Random Forest en MLflow.
- **`mlflowlightgbm.py`** → Script para la ejecución y carga del modelo LightGBM en MLflow.
- **`docker-compose.yml`** → Archivo de configuración para levantar un servidor local de PostgreSQL con Docker.

## **Cómo Visualizar el Proyecto**
**Jupyter Notebook:**
   - Accede al notebook en Deepnote para una mejor visualización: [Enlace aquí](https://deepnote.com/app/mateofernandez/Analisis-Exploratorio-de-Datos-y-Modelado-de-Prediccion-de-Ventas-f418be9d-f75b-406a-96c2-1d7387647953)
   - Abre los archivos en la carpeta correspondiente ( `modelo.ipynb`) para revisar el análisis detallado y las conclusiones. 
