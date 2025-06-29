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

    print(df.head())
    print(df.info())

    """ ****************************************************************** """
    """ ******************** Estatísticas Descritivas ******************** """

    # print(df.describe())

    """ ****************************************************************** """
    """ ******************* Limpeza e Pré-processamento ****************** """
    
    # Conforme Hair (2009), é essencial garantir que os dados estejam limpos
    # antes de prosseguir com a análise. A remoção de dados faltantes (NaN)
    # é um passo fundamental nesse processo.
    df = df.dropna()
    df = df.drop(['Title'], axis=1)

    """ ****************************************************************** """
    """ ***************** Transformações de Variáveis ******************** """
    # Hair (2009) discute a importância de verificar as premissas da regressão linear,
    # como a normalidade dos resíduos. Transformar variáveis com distribuições
    # muito assimétricas (como visto nos histogramas) pode ajudar a atender a essa premissa.
    # A transformação logarítmica é uma das mais comuns para reduzir a assimetria positiva.
    df['Public_Total'] = df['Public_Total'].apply(lambda x: np.log(x + 1))
    df['Days_in_exibithion'] = df['Days_in_exibithion'].apply(lambda x: np.log(x + 1))

    # plot_boxplots_and_detect_outliers(df, [item for item in numerical_cols if item != "Public_Total"])
    # df = remove_outliers_using_iqr(df, numerical_cols)
    # plot_boxplots_and_detect_outliers(df, [item for item in numerical_cols if item != "Public_Total"])

    """ ****************************************************************** """
    """ **************** Feature Engineering e Seleção ******************* """
    # A criação de variáveis a partir de outras existentes pode melhorar o modelo.
    # No entanto, variáveis de interação baseadas em colunas categóricas
    # devem ser criadas APÓS a codificação dessas colunas.
    
    # Transformação de data em features ciclicas e mensais para capturar sazonalidade.
    df.loc[:, 'Month'] = df['Release_Date'].dt.month
    df.loc[:, 'Day_of_Week_sin'] = np.sin(2 * np.pi * df['Release_Date'].dt.dayofweek / 7)
    
    # Remoção de colunas que não serão mais úteis ou que foram combinadas.
    # A multicolinearidade é um problema sério na regressão, e o VIF (Variance Inflation Factor)
    # é a principal medida para diagnosticá-la (Hair, 2009). 'Runtime' foi previamente
    # identificado com alto VIF, indicando que sua informação já está contida em outras variáveis.
    COLS_TO_DROP = ['Release_Date', 'Runtime']
    df = df.drop(COLS_TO_DROP, axis=1)

    """ ****************************************************************** """
    """ ***************** Separação em treino e teste ******************** """
    # Este é um dos passos mais críticos. Segundo Hair (2009) e todas as boas práticas
    # de modelagem, devemos dividir os dados ANTES de qualquer transformação que aprenda
    # parâmetros com os dados (como encoding e scaling). Isso previne o "data leakage",
    # garantindo que a avaliação do modelo seja feita em dados verdadeiramente "não vistos".
    X = df.drop(['Public_Total'], axis=1)
    y = df['Public_Total']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    
    """ ****************************************************************** """
    """ ************ Codificação, Feature Engineering e Padronização *********** """

    # 1. Identificando colunas para as transformações corretas
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # 2. Target Encoding para variáveis categóricas.
    # O encoder é "treinado" (fit) APENAS no conjunto de treino e depois aplicado em ambos.
    target_encoder = ce.TargetEncoder(cols=categorical_cols)
    X_train_encoded = target_encoder.fit_transform(X_train, y_train)
    X_test_encoded = target_encoder.transform(X_test)
    
    # 3. Feature Engineering (Pós-Codificação)
    # Agora que as colunas categóricas são numéricas, podemos criar interações.
    # Hair (2009) discute o uso de interações para capturar efeitos complexos.
    X_train_encoded['Ratting_Movie'] = X_train_encoded['Vote_Average'] * X_train_encoded['IMDB_Rating']
    X_train_encoded['Interaction_Cast'] = X_train_encoded['Cast_1'] * X_train_encoded['Cast_2'] * X_train_encoded['Cast_3'] * X_train_encoded['Director_1']

    X_test_encoded['Ratting_Movie'] = X_test_encoded['Vote_Average'] * X_test_encoded['IMDB_Rating']
    X_test_encoded['Interaction_Cast'] = X_test_encoded['Cast_1'] * X_test_encoded['Cast_2'] * X_test_encoded['Cast_3'] * X_test_encoded['Director_1']

    # 4. Removendo as colunas originais que formaram as interações
    cols_to_drop_post_encoding = ['Vote_Average', 'IMDB_Rating', 'Cast_1', 'Cast_2', 'Cast_3', 'Director_1']
    X_train_final = X_train_encoded.drop(columns=cols_to_drop_post_encoding)
    X_test_final = X_test_encoded.drop(columns=cols_to_drop_post_encoding)

    # 5. Padronização das variáveis
    # O scaler é "treinado" (fit) APENAS no conjunto de treino.
    # Hair (2009) recomenda a padronização para comparar a importância dos coeficientes.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_test_scaled = scaler.transform(X_test_final)
    
    # Recriando os DataFrames com as colunas e índices corretos após a padronização
    X_train_final = pd.DataFrame(X_train_scaled, columns=X_train_final.columns, index=X_train_final.index)
    X_test_final = pd.DataFrame(X_test_scaled, columns=X_test_final.columns, index=X_test_final.index)

    """ ****************************************************************** """
    """ ***************** Análise de Multicolinearidade ****************** """
    
    # A verificação do VIF deve ser feita nos dados de treino processados,
    # pois é com eles que o modelo será construído.
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_train_final.columns
    vif_data["VIF"] = [variance_inflation_factor(X_train_final.values, i) for i in range(X_train_final.shape[1])]
    print("\nFator de Inflação de Variância (VIF) nos dados de treino:")
    print(vif_data)
    print("_____________________________________________________\n")

    """ ****************************************************************** """
    """ ***** Treinamento e avaliação do modelo com cross-validation ***** """

    # O Statsmodels é excelente para inferência estatística, fornecendo um resumo
    # detalhado do modelo, incluindo p-valores para cada coeficiente, o que é
    # essencial para a análise de significância (Hair, 2009).
    X_train_sm = sm.add_constant(X_train_final) # Adicionar o intercepto para o OLS
    model_sm = sm.OLS(y_train, X_train_sm).fit()
    print(model_sm.summary())

    # O Scikit-learn é mais focado em previsão. Usaremos para treinar o modelo final
    # e fazer as predições no conjunto de teste.
    model = LinearRegression()
    model.fit(X_train_final, y_train)
    y_pred = model.predict(X_test_final)

    """ ****************************************************************** """
    """ ****************** Equação da Regressão Linear ******************* """

    coeficientes = model.coef_
    interceptor = model.intercept_

    print("\n📌 Equação da Regressão Linear:")
    equation = f"Public_Total = {interceptor:.4f} "
    for feature, coef in zip(X_train_final.columns, coeficientes):
        equation += f"+ ({coef:.4f} * {feature}) "
    
    print(equation)
    
    """ ****************************************************************** """
    """ ********************* Avaliação do modelo ************************ """
    # Avaliamos o modelo no conjunto de teste para ter uma medida de seu
    # poder de generalização. R² indica a proporção da variância na variável
    # dependente que é previsível a partir das variáveis independentes.
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # A validação cruzada (Cross-Validation) fornece uma estimativa mais robusta
    # do desempenho do modelo do que uma única divisão treino-teste (Hair, 2009).
    # Ela treina e avalia o modelo múltiplas vezes em diferentes subconjuntos dos dados.
    # IMPORTANTE: Para evitar data leakage na validação cruzada com o TargetEncoder,
    # seria necessário criar um pipeline mais complexo. Por simplicidade, faremos a CV
    # nos dados já processados do conjunto de treino, mas o ideal seria encapsular
    # o encoder e o scaler em um Pipeline do scikit-learn.
    scores = cross_val_score(model, X_train_final, y_train, cv=5, scoring='r2')

    print(f"\n📊 Resultados da Regressão Linear:")
    print(f"🔹 Mean Squared Error (MSE) no teste: {mse:.2f}")
    print(f"🔹 R² no teste: {r2:.2f}")
    print(f"🔹 Cross-Validation R² Scores (no treino): {scores}")
    print(f"🔹 Média dos R² (CV no treino): {scores.mean():.3f}")

    """ ****************************************************************** """
    """ ******************* Diagnóstico de resíduos ********************** """
    # A análise de resíduos é fundamental para verificar as premissas do modelo.
    # O gráfico de resíduos vs. valores previstos ajuda a identificar heterocedasticidade
    # (a variância dos erros não é constante). O ideal é uma nuvem de pontos aleatória
    # em torno da linha zero, sem padrões (funil, curva, etc.).
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Resíduos vs Valores Previstos (no Teste)')
    plt.xlabel('Valores Previstos')
    plt.ylabel('Resíduos')
    plt.show()

    # O Q-Q plot (Gráfico de Probabilidade Normal) compara a distribuição dos resíduos
    # com uma distribuição normal teórica. Se os pontos seguem a linha diagonal,
    # a premissa de normalidade dos resíduos é satisfeita.
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.show()

    """ ****************************************************************** """
    
if __name__ == "__main__":
    # Carregando os dados para execução do script
    DATA_PATH           = "data/output"
    FINAL_PATH          = f"{DATA_PATH}/DATAS/FINAL_DATABASE"
    FINAL_FILE_NAME     = "MOVIES"
    DF_FILE = FINAL_PATH + "/" + FINAL_FILE_NAME + ".parquet"

    df = pd.read_parquet(DF_FILE)
    runLinearRegressor(df)





    // ... existing code ...
    # Treinando o pipeline
    model_pipeline.fit(X_train, y_train)
    print("Pipeline de pré-processamento e treinamento concluído.")