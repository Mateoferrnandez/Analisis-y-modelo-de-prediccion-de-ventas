
import pandas as pd
import numpy as np

def outliers(df,tipo):
    if tipo =="z-score":
        tresshold=3
        z_scores = np.abs((df - df.mean()) / df.std())
        outliers = z_scores > tresshold
        return outliers
    elif tipo=="IQR":
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
        return outliers
def handleoutliers(df,metodo,outliers):
    if metodo == "remove":
            df_cleaned = df[(~outliers).all(axis=1)]
    elif metodo == "cap":
            df_cleaned = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)
    return df_cleaned
