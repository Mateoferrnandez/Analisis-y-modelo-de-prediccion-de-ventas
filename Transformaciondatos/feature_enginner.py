


import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
encoder = OneHotEncoder(sparse_output=False, drop=None)
def talla(row):
    if row in (['T-6', 'T-8','T-10','T-12','T-14','T-16']):
        return "Pequeña"
    elif row in (['T-XS','T-S', 'T-M', 'T-L']):
        return "Mediana"
    elif row in (['T-XL', 'T-XXL', 'T-UNI']):
        return "Grande"
def ropa(row):
    if row in (['201 RE-BLUSAS FEM', '206 RE-CAMISAS FEM', '207 RE-CAMISETAS FEM','210 RE-CHALECOS FEM','211 RE-CHAQUETAS FEM','202 RE-BODYS FEM','204 RE-BUZOS FEM']):
        return "Ropa_superior"
    if row in (['209 RE-CAPRIS FEM', '215 RE-FALDAS FEM', '216 RE-JEANS FEM','217 RE-JOGGERS FEM','218 RE-LEGGINS FEM','220 RE-PANTALONES FEM','221 RE-PESCADORES FEM','223 RE-SHORTS FEM']):
        return "Ropa_inferior"
    if row in (['200 RE-PAQUETONES', '213 RE-CONJUNTOS FEM','214 RE-ENTERIZOS FEM','219 RE-OVEROLES FEM','224 RE-SOBRETODOS FEM','225 RE-VESTIDOS FEM']):
        return "Conjuntos y otros"
def trimestres(row):
    if row in (['201901', '201902', '201903']):
        return 'Primer_trimestre'
    if row in (['201904', '201905', '201906']):
        return 'Segundo_trimestre'
    if row in (['201907', '201908', '201909']):
        return 'Tercer_trimestre'
    if row in (['201910', '201911', '201912']):
        return 'Cuarto_trimestre'
encoder = OneHotEncoder(sparse_output=False, drop=None)
def onehotencoding(df: pd.DataFrame,features):
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(encoder.fit_transform(df[features]),columns=encoder.get_feature_names_out(features))
        df_transformed = df_transformed.drop(columns=features).reset_index(drop=True)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        return df_transformed