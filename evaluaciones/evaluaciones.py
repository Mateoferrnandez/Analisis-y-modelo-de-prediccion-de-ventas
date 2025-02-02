import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

def evaluaciones(modelo,X,y,y_pred,y_test):
    try:
        cv_scores = cross_val_score(modelo, X, y, cv=5, scoring='neg_mean_absolute_error')  # MAE negativo para que los valores sean positivos
        cv_rmse = cross_val_score(modelo, X, y, cv=5, scoring='neg_root_mean_squared_error')  # RMSE negativo

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)  # squared=False da la raíz cuadrada (RMSE)
        r2 = r2_score(y_test, y_pred)
        return cv_scores,cv_rmse,mae,rmse,r2
    except:
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)  # squared=False da la raíz cuadrada (RMSE)
        r2 = r2_score(y_test, y_pred)
        return mae,rmse,r2