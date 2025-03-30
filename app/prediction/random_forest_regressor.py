import pandas as pd
import numpy as np
import category_encoders as ce
import time

from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def plot_boxplots_and_detect_outliers(df, numerical_cols):
    # plt.figure(figsize=(12, 8))
    # sns.boxplot(data=df[numerical_cols])
    # plt.title('Boxplot das Variáveis Numéricas')
    # plt.xticks(rotation=45)
    # plt.show()

    outliers = {}
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        outliers[col] = df[(df[col] < lower_limit) | (df[col] > upper_limit)]
        # if not outliers[col].empty:
        #     print(f"Outliers em {col}:")
        #     print(outliers[col][col])
        #     print("\n")
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

def random_forest_model(df: pd.DataFrame):
    """ ************* Carregamento e Visão Geral dos Dados  ************* """
    # DATA_PATH           = "data/output"
    # FINAL_PATH          = f"{DATA_PATH}/DATAS/FINAL_DATABASE"
    # FINAL_FILE_NAME     = "MOVIES"
    # DF_FILE = FINAL_PATH + "/" + FINAL_FILE_NAME + ".parquet"

    # df = pd.read_parquet(DF_FILE)
    print(df.head())
    # print(df.info())
    """ ****************************************************************** """
    """ ******************** Estatísticas Descritivas ******************** """
    # print(df.describe())
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
    """ ***************** Transformar as Distribuições ******************* """
    # df['Log_Public_Total'] = np.log1p(df['Public_Total'])
    # df['Log_Days_in_exibithion'] = np.log1p(df['Days_in_exibithion'])
    df['Public_Total'] = np.log1p(df['Public_Total'])
    df['Days_in_exibithion'] = np.log1p(df['Days_in_exibithion'])
    """ ****************************************************************** """
    """ ********************* Transformações de Data  ******************** """
    df.loc[:, 'Month'] = df['Release_Date'].dt.month
    df.loc[:, 'Day_of_Week'] = df['Release_Date'].dt.dayofweek
    df.loc[:, 'Day_of_Week_sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
    df.drop(['Release_Date', 'Day_of_Week'], axis=1, inplace=True)
    """ ****************************************************************** """
    """ ****************** Codificação das Variáveis ********************* """
    # encoder = ce.TargetEncoder(cols=['Prodution_country', 'Genre_1', 'Production_Companies', 'Cast_1', 'Director_1'])
    encoder = ce.TargetEncoder(cols=categorical_cols)
    X_encoded = encoder.fit_transform(df.drop(columns=['Public_Total']), df['Public_Total'])
    """ ****************************************************************** """
    """ ****************** Junção de Variáveis ********************* """
    X_encoded['Cast'] = X_encoded[['Cast_1', 'Cast_2', 'Cast_3']].mean(axis=1)
    X_encoded.drop(columns=['Cast_1', 'Cast_2', 'Cast_3'], inplace=True)
    """ ****************************************************************** """
    """ ****************** Padronização das Variáveis ********************* """
    scaler = StandardScaler()
    columns = ['Days_in_exibithion', 'Vote_Average', 'Runtime', 'IMDB_Rating', 'Month', 'Day_of_Week_sin']
    X_encoded[columns] = scaler.fit_transform(
        X_encoded[columns]
    )
    """ ****************************************************************** """
    """ ***************** Separação em treino e teste ******************** """
    X = X_encoded
    y = df['Public_Total']  # Variável dependente
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    """ ****************************************************************** """
    """ ***** Treinamento e avaliação do modelo com cross-validation ***** """

    print("🔄 Treinando o modelo Random Forest... isso pode levar algum tempo.")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Adicionando a barra de progresso
    for i in tqdm(range(1), desc="Treinando modelo..."):  # A barra de progresso será mostrada enquanto o treinamento ocorrer
        rf_model.fit(X_train, y_train)

    # rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    # rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_squared_error')

    importances = rf_model.feature_importances_
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

    print(f"\n📊 Resultados do Random Forest:")
    print(f"🔹 Mean Squared Error (MSE): {mse:.2f}")
    print(f"🔹 R²: {r2:.2f}")
    print(f"🔹 'Cross-validation MSE': {-cv_scores.mean()}")
    print(f"🔹 'Feature Importance': {feature_importance.sort_values(by='Importance', ascending=False)}")
    """ ****************************************************************** """
    # """ ****************** Ajuste de Hiperparâmetros ********************* """

    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [None, 10, 20],
    #     'min_samples_split': [2, 10, 20],
    #     'min_samples_leaf': [1, 5, 10],
    #     'max_features': ['auto', 'sqrt', 'log2']
    # }

    # grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=5)
    # for _ in tqdm(range(1), desc="Ajustando hiperparâmetros..."):
    #     grid_search.fit(X_train, y_train)

    # print(f"🔹 'Best Parameters': {grid_search.best_params_}")
    
    # """ ****************************************************************** """