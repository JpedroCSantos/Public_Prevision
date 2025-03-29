import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from category_encoders import TargetEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from prediction.statistics_analysis import analyze_statistics  

def apply_transformations(df):
    """
    Aplica transformações para corrigir a normalidade das variáveis numéricas.
    """
    print("\n📌 Aplicando correções para normalidade...")

    # 1️⃣ Transformação Box-Cox (apenas para valores positivos)
    boxcox_transformer = PowerTransformer(method='box-cox', standardize=True)
    df['Runtime'] = boxcox_transformer.fit_transform(df[['Runtime']])
    df['IMDB_Rating'] = boxcox_transformer.fit_transform(df[['IMDB_Rating']])

    # 2️⃣ Transformação log1p (para corrigir assimetria positiva)
    df['Days_in_exibithion'] = np.log1p(df['Days_in_exibithion'])

    # 3️⃣ Transformação log-invertida para assimetria negativa
    max_value = df['Vote_Average'].max()
    df['Vote_Average'] = np.log1p(max_value - df['Vote_Average'])

    # 4️⃣ Transformação log1p para variável alvo (Public_Total)
    df['Public_Total'] = np.log1p(df['Public_Total'])

    return df

def plot_variable_distributions(df):
    """
    Gera histogramas das variáveis independentes com títulos ajustados.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))

    # Variáveis e novos títulos
    variables = ["Days_in_exibithion", "Runtime", "IMDB_Rating", "Vote_Average"]
    new_titles = ["Dias em Exibição", "Tempo de Execução", "Avaliação do Público", "Avaliação dos Críticos"]

    for ax, var, title in zip(axes.flat, variables, new_titles):
        ax.hist(df[var], bins=50, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel(var)
        ax.set_ylabel("Frequência")

    plt.tight_layout()
    plt.show()

def runLinearRegressor(df: pd.DataFrame):
    # """ *************** Leitura do DataFrame  *************** """
    # DATA_PATH = "data/output"
    # FINAL_PATH = f"{DATA_PATH}/DATAS/FINAL_DATABASE"
    # FINAL_FILE_NAME = "MOVIES"

    # df_original = pd.read_parquet(f"{FINAL_PATH}/{FINAL_FILE_NAME}.parquet")
    # df = df_original.drop(['Cast_3', 'Cast_2', 'Title'], axis=1, inplace=False) #Removida por alta multicolinearidade
    """ ****************************************************************** """
    """ **************** Remoção de Outliers em Public_Total ************* """
    Q1 = df['Public_Total'].quantile(0.25)
    Q3 = df['Public_Total'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['Public_Total'] < (Q1 - 1.5 * IQR)) | (df['Public_Total'] > (Q3 + 1.5 * IQR)))].copy()

    """ ****************************************************************** """
    """ ******************** RESUMOS DA BASE DE DADOS ******************** """

    popular_movies = df[df['Public_Total'] > 1000000]
    print(popular_movies[['Release_Date', 'Public_Total', 'Vote_Average', 'Vote_Average']])
    # df[['Days_in_exibithion', 'Runtime', 'IMDB_Rating', 'Vote_Average']].hist(bins=50, figsize=(15, 10))
    # plot_variable_distributions(df)
    # plt.show()

    """ ****************************************************************** """
    """ ********************* Transformações de Data  ******************** """

    df.loc[:, 'Month'] = df['Release_Date'].dt.month
    df.loc[:, 'Day_of_Week'] = df['Release_Date'].dt.dayofweek
    df.loc[:, 'Day_of_Week_sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
    # df.loc[:, 'Day_of_Week_cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
    df.drop(['Release_Date', 'Day_of_Week'], axis=1, inplace=True)

    """ ****************************************************************** """
    """ ************** Variáveis categóricas e numéricas ***************** """

    numeric_cols = ['Runtime', 'IMDB_Rating', 'Days_in_exibithion', 'Vote_Average']
    categorical_cols = list(set(df.columns) - set(numeric_cols))

    print(categorical_cols)
    """ ****************************************************************** """
    """ **************** Aplicação de Target Encoding ******************** """

    encoder = TargetEncoder(cols=categorical_cols)
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols], df['Public_Total'])
    df = df.dropna(subset=numeric_cols + categorical_cols)

    """ ****************************************************************** """
    # """ *** Verificação estatistica (Normalidade e homocedasticidade) **** """

    # dependent_var = 'Public_Total'
    # independent_vars = numeric_cols
    # analyze_statistics(df, dependent_var, independent_vars)

    # """ ****************************************************************** """
    """ ************ Normalização das variáveis numéricas **************** """
    
    df = apply_transformations(df)
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    """ ****************************************************************** """
    """ ******************** Teste de Correlação ******************** """
    X = df.drop(columns=['Public_Total'])

    correlation_matrix = df.corr()
    print(correlation_matrix)
    # Adiciona uma constante (intercepto) ao modelo
    X = sm.add_constant(X)

    vif_data = pd.DataFrame()
    vif_data["Variável"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    print(vif_data)

    """ ****************************************************************** """
    # """ ** 2º Verificação estatistica (Normalidade e homocedasticidade) ** """

    # dependent_var = 'Public_Total'
    # independent_vars = numeric_cols
    # analyze_statistics(df, dependent_var, independent_vars)


    # """ ****************************************************************** """
    """ ***************** Separação em treino e teste ******************** """
    X = df.drop(['Public_Total'], axis=1)
    y = df['Public_Total']

    model = sm.OLS(y, X).fit()
    print(model.summary())

    # X = sm.add_constant(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    """ ****************************************************************** """
    """ ***** Treinamento e avaliação do modelo com cross-validation ***** """

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    """ ****************************************************************** """
    """ ****************** Equação da Regressão Linear ******************* """

    coeficientes = model.coef_
    interceptor = model.intercept_

    print("\n📌 Equação da Regressão Linear:")
    equation = f"Public_Total = {interceptor:.4f} "
    for feature, coef in zip(X.columns, coeficientes):
        equation += f"+ ({coef:.4f} * {feature}) "
    
    print(equation)
    """ ****************************************************************** """
    """ ********************* Avaliação do modelo ************************ """

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')

    print(f"\n📊 Resultados da Regressão Linear:")
    print(f"🔹 Mean Squared Error (MSE): {mse:.2f}")
    print(f"🔹 R²: {r2:.2f}")
    print(f"🔹 Cross-Validation R² Scores: {scores}")
    print(f"🔹 Média dos R²: {scores.mean():.3f}")

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