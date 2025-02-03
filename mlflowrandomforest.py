import pandas as pd
import numpy as np
import mlflow
from mlflow.models import infer_signature
from modelos.modelos import random_forest
from evaluaciones.evaluaciones import evaluaciones

df = pd.read_csv("D:\Documentos\Github\Prueba_tecnica_Azzorti\Datosmodelo.csv")

# Definir los hiperparámetros del modelo

params = {
        "n_estimators": 100,  # Número de árboles en el bosque. El valor por defecto es 100.
        "max_depth": None,  # Profundidad máxima de los árboles. Por defecto es None (sin límite).
        "min_samples_split": 2,  # Número mínimo de muestras para dividir un nodo. Por defecto es 2.
        "min_samples_leaf": 1,  # Número mínimo de muestras en un nodo hoja. Por defecto es 1. # Número máximo de características a considerar para dividir un nodo. Por defecto es 'auto'.
        "bootstrap": True,  # Si se usan muestras bootstrap (con reemplazo). Por defecto es True.
        "oob_score": False,  # Si se calcula la puntuación fuera de la bolsa (out-of-bag). Por defecto es False.
        "n_jobs": 1,  # Número de trabajos (hilos) a usar para el procesamiento paralelo. Por defecto es 1.
        "random_state": 42,  # Semilla del generador aleatorio para reproducibilidad. Por defecto es None.
        "warm_start": False,  # Si se reutilizan los árboles previos para agregar más árboles. Por defecto es False.  # Función de medición de la calidad de la división de los nodos. 'mse' por defecto.
        "max_samples": None,  # Número de muestras para ajustar cada árbol. Por defecto es None.
    }
modelo, X_train, X_test, y_train, y_test, X, y = random_forest(df, params)

# Predecir en el conjunto de prueba
y_pred = modelo.predict(X_test)

# Calcular métricas de evaluación
cv_scores, cv_rmse, mae, rmse, r2 = evaluaciones(modelo, X, y, y_pred, y_test)
stddatos = df["VENTA"].std()

# Establecer la URI de seguimiento de MLflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Crear un nuevo experimento en MLflow
mlflow.set_experiment("MLflow Quickstart")

# Iniciar una ejecución en MLflow
with mlflow.start_run():
    # Registrar los hiperparámetros
    mlflow.log_params(params)
    
    # Registrar las métricas de evaluación
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("std_target", stddatos)
    mlflow.log_metric("cv_score", -np.mean(cv_scores))
    mlflow.log_metric("cv_rmse", -np.mean(cv_rmse))
    
    # Establecer una etiqueta para identificar esta ejecución
    mlflow.set_tag("Final run random    forest", "Final parameters")
    
    # Inferir la firma del modelo
    signature = infer_signature(X_train, modelo.predict(X_train))
    
    # Registrar el modelo en MLflow
    model_info = mlflow.sklearn.log_model(
        sk_model=modelo,
        artifact_path="modelo_ventas",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-randomforest",
    )

# Cargar el modelo para hacer predicciones como una función genérica en Python
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

# Hacer predicciones con el modelo cargado
predictions = loaded_model.predict(X_test)

# Crear un DataFrame con los resultados
feature_names = df.columns
result = pd.DataFrame(X_test, columns=feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions

# Mostrar las primeras 4 filas del resultado
result[:4]
