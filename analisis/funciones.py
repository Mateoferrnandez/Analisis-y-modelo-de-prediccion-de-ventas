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
    
def univariadocategorico(df,feature:str):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=feature, data=df, palette="muted")
    plt.title(f"Distribucion de {feature}")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()