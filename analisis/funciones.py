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
def bivariatecategoricalvsnumerical(df,feature1,feature2):
    plt.figure(figsize=(15, 6))
    sns.boxplot(x=feature1, y=feature2, data=df)
    plt.title(f"{feature1} vs {feature2}")
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.xticks(rotation=45)
    plt.show()
def bivariatenumericasvsnumerical(df,feature1,feature2):
    plt.figure(figsize=(15, 6))
    sns.scatterplot(x=feature1, y=feature2, data=df)
    plt.title(f"{feature1} vs {feature2}")
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.show()
def multivarado(df):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout
    plt.show()


def visualizacionmodelo(y_test,y_pred,titulo:str):
    plt.figure(figsize=(16, 8))
    sns.histplot(y_test, bins=30, kde=True, color="blue", label="y_test", alpha=0.5)
    sns.histplot(y_pred, bins=30, kde=True, color="orange", label="y_pred", alpha=0.5)

    plt.xlabel("Valores")
    plt.ylabel("Frecuencia")
    plt.title(titulo)
    plt.legend()
    plt.show()