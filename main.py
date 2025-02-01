import mlflow



import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df=pd.read_csv("D:\Documentos\Github\Prueba_tecnica_Azzorti\Datosmodelo.csv")
df=df.select_dtypes(include=[int,float])
X, y = df.drop(columns=['VENTA']),df['VENTA']


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Define the model hyperparameters
params = {
    "objective": "reg:squarederror",  # Para regresión con MSE
    "learning_rate": 0.1,
    "n_estimators": 500,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "rmse",  # Métrica de evaluación durante el entrenamiento
    "random_state": 8888,
}
# Train the model

model = XGBRegressor(**params)  
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate metrics
#Validación cruzada para obtener el rendimiento del modelo en múltiples particiones
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')  # MAE negativo para que los valores sean positivos
cv_rmse = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')  # RMSE negativo

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)  # squared=False da la raíz cuadrada (RMSE)
r2 = r2_score(y_test, y_pred)
                                
#mlflow.set_tracking_uri(uri="http://locashost:5000")
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2",r2)
    mlflow.log_metric("cv_scores",cv_scores)
    mlflow.log_metric("cv_rmse",cv_rmse)
    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("First run mlflow", "First parameters")

    # Infer the model signature
    signature = infer_signature(X_train, model.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="modelo_ventas",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )
# Load the model back for predictions as a generic Python Function model
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = loaded_model.predict(X_test)

feature_names = df.feature_names

result = pd.DataFrame(X_test, columns=feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions

result[:4]
