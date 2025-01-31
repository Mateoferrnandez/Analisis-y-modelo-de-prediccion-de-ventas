import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt


def univariadonumerico(df,feature:str):

    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], kde=True, bins=30)
    plt.title(f"Distribucion de {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frecuencia")
    plt.show()
    plt.figure(figsize=(10, 6))
    
