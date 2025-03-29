import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import category_encoders as ce
import matplotlib.pyplot as plt

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from prediction.statistics_analysis import analyze_distribution, analyze_homoscedasticity
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def plot_boxplots_and_detect_outliers(df, numerical_cols):
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df[numerical_cols])
    plt.title('Boxplot das Variáveis Numéricas')
    plt.xticks(rotation=45)
    plt.show()

    outliers = {}
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        outliers[col] = df[(df[col] < lower_limit) | (df[col] > upper_limit)]
        if not outliers[col].empty:
            print(f"Outliers em {col}:")
            print(outliers[col][col])
            print("\n")
    return outliers

def remove_outliers_using_iqr(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_limit) & (df[col] <= upper_limit)]
        
    return df

def runLinearRegressor(df: pd.DataFrame):
    """ ************* Carregamento e Visão Geral dos Dados  ************* """
    # DATA_PATH           = "data/output"
    # FINAL_PATH          = f"{DATA_PATH}/DATAS/FINAL_DATABASE"
    # FINAL_FILE_NAME     = "MOVIES"
    # DF_FILE = FINAL_PATH + "/" + FINAL_FILE_NAME + ".parquet"

    # df = pd.read_parquet(DF_FILE)
    print(df.head())
    print(df.info())
    """ ****************************************************************** """
    """ ******************** Estatísticas Descritivas ******************** """
    print(df.describe())
    """ ****************************************************************** """
    """ ******************* Remoção de dados faltantes ******************* """
    df = df.dropna()
    df = df.drop(['Title'], axis=1, inplace=False)
    # df = df.drop(['Title', 'Cast_2', 'Cast_3'], axis=1, inplace=False)
    """ ****************************************************************** """
    """ *************** Detecção e Tratamento de Outliers **************** """
    # Selecionando as variáveis numéricas para análise
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    outliers = plot_boxplots_and_detect_outliers(df, numerical_cols)
    df = remove_outliers_using_iqr(df, numerical_cols)
    plot_boxplots_and_detect_outliers(df, numerical_cols)
    """ ****************************************************************** """
    """ ****************** Visualizar as Distribuições ******************* """
    plt.figure(figsize=(12, 8))
    df[numerical_cols].hist(bins=50, figsize=(12, 10))
    plt.show()

    dependent_var = 'Public_Total'
    independent_vars = numerical_cols
    analyze_distribution(df, independent_vars)
    """ ****************************************************************** """
    """ ***************** Transformar as Distribuições ******************* """
    df['Public_Total'] = df['Public_Total'].apply(lambda x: np.log(x + 1))
    df['Days_in_exibithion'] = df['Days_in_exibithion'].apply(lambda x: np.log(x + 1))

    df[['Public_Total', 'Days_in_exibithion']].hist(bins=50, figsize=(12, 8))
    plt.tight_layout()
    plt.show()
    analyze_distribution(df, independent_vars)
    """ ****************************************************************** """
    """ ******************** Análise de Correlação *********************** """
    numeric_cols = [col for col in numerical_cols if col != 'Public_Total']
    correlation_matrix = df[numeric_cols].corr()
    print(correlation_matrix)

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Matriz de Correlação')
    plt.show()

    X = df[numeric_cols]
    X = add_constant(X)

    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif_data)
    """ ****************************************************************** """
    """ ******************** Análise de Homocedasticidade **************** """
    analyze_homoscedasticity(X, y)
    """ ****************************************************************** """
    """ ********************* Transformações de Data  ******************** """
    df.loc[:, 'Month'] = df['Release_Date'].dt.month
    df.loc[:, 'Day_of_Week'] = df['Release_Date'].dt.dayofweek
    df.loc[:, 'Day_of_Week_sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
    df.drop(['Release_Date', 'Day_of_Week'], axis=1, inplace=True)
    """ ****************************************************************** """
    """ ****************** Codificação das Variáveis ********************* """
    categorical_columns = ['Production_Companies', 'Prodution_country', 'Genre_1', 'Cast_1', 'Cast_2', 'Cast_3', 'Director_1']
    # categorical_columns = ['Production_Companies', 'Prodution_country', 'Genre_1', 'Cast_1', 'Director_1']
    target_encoder = ce.TargetEncoder(cols=categorical_columns)
    model = LinearRegression()
    df = target_encoder.fit_transform(df, df['Public_Total'])

    X = df.drop(columns='Public_Total')
    y = df['Public_Total']
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')

    print("Scores de MSE para cada fold:", scores)
    print("Média do MSE:", np.mean(scores))
    print(df.head())
    """ ****************************************************************** """
    """ ******************* Adicionando correlações ********************** """
    df['Interaction_Cast'] = df['Cast_1'] * df['Cast_2'] * df['Cast_3']
    df = df.drop(['Cast_1', 'Cast_2', 'Cast_3'], axis=1, inplace=False)
    """ ****************************************************************** """
    """ ****************** Padronização das Variáveis ********************* """
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    print(df.head())
    """ ****************************************************************** """
    """ ******************** Análise de Correlação *********************** """
    col = [col for col in df.columns if col != 'Public_Total']
    correlation_matrix = df[col].corr()
    print(correlation_matrix)

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Matriz de Correlação')
    plt.show()


    X = df[col]
    X = add_constant(X)

    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif_data)
    """ ****************************************************************** """
    """ ***************** Separação em treino e teste ******************** """
    X = df.drop(['Public_Total'], axis=1)
    y = df['Public_Total']

    model = sm.OLS(y, X).fit()
    print(model.summary())

    X = sm.add_constant(X)
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

    stats.probplot(y_test - y_pred, dist="norm", plot=plt)
    plt.show()

    """ ****************************************************************** """
    
if __name__ == "__main__":
    runLinearRegressor()