
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

def xgb(df,params):
    df=df.select_dtypes(include=[int,float])
    X, y = df.drop(columns=['VENTA']),df['VENTA']

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    # Entrenar el modelo

    model = XGBRegressor(**params)  
    model.fit(X_train, y_train)
    return model,X_train, X_test, y_train, y_test,X,y


def random_forest(df,params):
    df_numeric = df.select_dtypes(include=[int, float])
    X, y = df_numeric.drop(columns=['VENTA']), df['VENTA'] 
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    # Crear el modelo de regresión con Random Forest
    model = RandomForestRegressor(**params)

    # Entrenar el modelo
    model.fit(X_train, y_train)
    return model,X_train, X_test, y_train, y_test,X,y

def lightgbm(df,params):
    df_numeric = df.select_dtypes(include=[int, float])
    X, y = df_numeric.drop(columns=['VENTA']), df['VENTA'] 
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    # Crear el dataset para LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)


    # Entrenar el modelo
    callbacks = [lgb.early_stopping(stopping_rounds=50)]
    model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[test_data],callbacks=callbacks) # Número máximo de iteracionesvalid_sets=[test_data],  # Conjunto de validacióncallbacks=callbacks  # Callback para early stopping)
    df_numeric = df.select_dtypes(include=[int, float])
    X, y = df_numeric.drop(columns=['VENTA']), df['VENTA'] 
    
    # Crear el modelo de LightGBM compatible con scikit-learn
    model1 = lgb.LGBMRegressor(**params)
    model1.fit(X_train, y_train)

    return model,X_train, X_test, y_train, y_test,X,y,model.best_iteration,model1