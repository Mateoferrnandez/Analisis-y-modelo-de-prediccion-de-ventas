import pandas as pd
import numpy as np
import mlflow
from mlflow.models import infer_signature
from modelos.modelos import lightgbm
from evaluaciones.evaluaciones import evaluaciones

df = pd.read_csv("D:\Documentos\Github\Prueba_tecnica_Azzorti\Datosmodelo.csv")

# Definir los hiperparámetros del modelo LightGBM
params = {
    'objective': 'regression',  # Tipo de problema (regresión)
    'metric': 'rmse',  # Métrica de evaluación
    'boosting_type': 'gbdt',  # Tipo de boosting
    'num_iterations': 100,  # Número de iteraciones
    'num_leaves': 31,  # Número de hojas
    'learning_rate': 0.05,  # Tasa de aprendizaje
    'feature_fraction': 0.9  # Fracción de características a usar
}

modelo, X_train, X_test, y_train, y_test, X, y, mejor_iteracion, modelo1 = lightgbm(df, params)

# Predecir en el conjunto de prueba
y_pred = modelo1.predict(X_test)

# Calcular métricas de evaluación
cv_scores, cv_rmse, mae, rmse, r2 = evaluaciones(modelo1, X, y, y_pred, y_test)
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
    mlflow.set_tag("Second run lightgbm", "Second parameters")
    
    # Inferir la firma del modelo
    signature = infer_signature(X_train, modelo.predict(X_train))
    
    # Registrar el modelo en MLflow
    model_info = mlflow.sklearn.log_model(
        sk_model=modelo,
        artifact_path="modelo_ventas",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-lightgbm",
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
