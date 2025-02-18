import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from category_encoders import TargetEncoder
from sklearn.preprocessing import MinMaxScaler

def runLinearRegressor(df: pd.DataFrame):
    # """ *************** Leitura do DataFrame  *************** """
    # DATA_PATH = "data/output"
    # FINAL_PATH = f"{DATA_PATH}/DATAS/FINAL_DATABASE"
    # FINAL_FILE_NAME = "MOVIES"

    # df_original = pd.read_parquet(f"{FINAL_PATH}/{FINAL_FILE_NAME}.parquet")
    # df = df_original.drop(['Cast_3', 'Cast_2', 'Title'], axis=1, inplace=False) #Removida por alta multicolinearidade
    # """ ****************************************************** """
    """ **************** Remoção de Outliers em Public_Total ************* """

    Q1 = df['Public_Total'].quantile(0.25)
    Q3 = df['Public_Total'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['Public_Total'] < (Q1 - 1.5 * IQR)) | (df['Public_Total'] > (Q3 + 1.5 * IQR)))].copy()

    """ ****************************************************************** """
    """ ******************** RESUMOS DA BASE DE DADOS ******************** """

    # print(df.info())
    # print(df.describe())
    #Resumo estatístico para colunas específicas
    # print(df[['Public_Total', 'Vote_Average', 'IMDB_Rating']].describe())
    #Filtrar filmes com mais de 1 milhão de espectadores
    popular_movies = df[df['Public_Total'] > 1_000_000]
    print(popular_movies[['Release_Date', 'Public_Total', 'Vote_Average', 'Vote_Average']])
    df[['Public_Total', 'Days_in_exibithion', 'Runtime', 'IMDB_Rating', 'Vote_Average']].hist(bins=50, figsize=(15, 10))
    plt.show()

    """ ****************************************************************** """
    """ ********************* Transformações de Data  ******************** """

    df.loc[:, 'Month'] = df['Release_Date'].dt.month
    df.loc[:, 'Day_of_Week'] = df['Release_Date'].dt.dayofweek
    df.loc[:, 'Day_of_Week_sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
    df.loc[:, 'Day_of_Week_cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
    df.drop(['Release_Date', 'Day_of_Week'], axis=1, inplace=True)

    """ ****************************************************************** """
    """ ************** Variáveis categóricas e numéricas ***************** """

    numeric_cols = ['Runtime', 'IMDB_Rating', 'Days_in_exibithion', 'Vote_Average']
    categorical_cols = ['Genre_1', 'Prodution_country', 'Production_Companies', 'Director_1', 'Cast_1']

    """ ****************************************************************** """
    """ **************** Aplicação de Target Encoding ******************** """

    encoder = TargetEncoder(cols=categorical_cols)
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols], df['Public_Total'])
    # Verificação de valores ausentes
    df = df.dropna(subset=numeric_cols + categorical_cols)

    """ ****************************************************************** """
    """ ************ Normalização das variáveis numéricas **************** """

    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    """ ****************************************************************** """
    """ ***************** Separação em treino e teste ******************** """

    X = df.drop(['Public_Total'], axis=1)
    y = df['Public_Total']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    """ ****************************************************************** """
    """ ***** Treinamento e avaliação do modelo com cross-validation ***** """

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    """ ****************************************************************** """
    """ ********************* Avaliação do modelo ************************ """

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')

    print(f"Linear Regression - MSE: {mse:.2f}")
    print(f"Linear Regression - R²: {r2:.2f}")
    print(f"Cross-Validation R² Scores: {scores}")
    print(f"Mean R²: {scores.mean()}")

    """ ****************************************************************** """
    """ ******************* Diagnóstico de resíduos ********************** """

    plt.scatter(y_test, y_test - y_pred)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Resíduos vs Valores Reais')
    plt.xlabel('Valores Reais')
    plt.ylabel('Resíduos')
    plt.show()

    # Teste de normalidade dos resíduos
    stats.probplot(y_test - y_pred, dist="norm", plot=plt)
    plt.show()

    """ ****************************************************************** """