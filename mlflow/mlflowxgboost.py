import pandas as pd
import numpy as np
import mlflow
from mlflow.models import infer_signature
from modelos import xgb
from evaluaciones import evaluaciones
df = pd.read_csv("D:\Documentos\Github\Prueba_tecnica_Azzorti\Datosmodelo.csv")

# Definir los hiperparámetros del modelo
params = {
    "objective": "reg:squarederror",  # Para regresión con MSE
    "learning_rate": 0.03,
    "n_estimators": 500,
    "max_depth": 5,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "rmse",  # Métrica de evaluación durante el entrenamiento
    "random_state": 8888,
}

modelo, X_train, X_test, y_train, y_test, X, y = xgb(df, params)

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
    mlflow.set_tag("First run XGBOOST", "First parameters")
    
    # Inferir la firma del modelo
    signature = infer_signature(X_train, modelo.predict(X_train))
    
    # Registrar el modelo en MLflow
    model_info = mlflow.sklearn.log_model(
        sk_model=modelo,
        artifact_path="modelo_ventas",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-xgboost",
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
