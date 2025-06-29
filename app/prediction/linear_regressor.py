import pandas as pd
import numpy as np
import statsmodels.api as sm
import category_encoders as ce
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

def runLinearRegressor(df: pd.DataFrame):
    """
    Executa um fluxo completo de regressão linear com base no arquivo MOVIES.csv,
    seguindo as melhores práticas de pré-processamento e avaliação de modelo.
    """
    # 1. Carregamento e Limpeza Inicial dos Dados
    df = df.dropna().drop_duplicates()
    df = df.drop(['Title'], axis=1, errors='ignore')
    print(f"Formato após limpeza inicial (dropna, drop_duplicates): {df.shape}")

    # ===== PLANO DE AÇÃO (ITERAÇÃO 2) =====
    # Remoção de variáveis não significativas e das que causam overfitting para criar um modelo base.
    vars_to_drop = [
        'Runtime', 'Vote_Average', 'IMDB_Rating', 'Month', 'Day_of_Week_sin',
        'Director_1', 'Release_Date', 'budget'
    ]
    df = df.drop(columns=vars_to_drop, errors='ignore')
    print(f"Variáveis removidas. Mantendo 'Cast_1, 2, 3' para engenharia de features. Novo formato: {df.shape}")
    
    # Transformação logarítmica da variável alvo
    df['Public_Total'] = np.log1p(df['Public_Total'])

    # 2. Separação em Conjuntos de Treino e Teste
    X = df.drop('Public_Total', axis=1)
    y = df['Public_Total']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    
    # 3. Engenharia de Features: Níveis de Elenco (Pós-Split)
    print("Iniciando engenharia de features: Níveis de Elenco...")
    # Contar a frequência de cada ator APENAS no conjunto de treino
    cast_cols = ['Cast_1', 'Cast_2', 'Cast_3']
    actor_counts = pd.concat([X_train[col] for col in cast_cols]).value_counts()
    
    # Definir os limiares para os tiers (quantis)
    q1 = actor_counts.quantile(0.66) # 66% menos frequentes
    q2 = actor_counts.quantile(0.95) # 95% mais frequentes
    
    def get_actor_tier(actor, counts, q1, q2):
        count = counts.get(actor, 0)
        if count > q2:
            return 3 # Tier 1 (A-Lister)
        elif count > q1:
            return 2 # Tier 2 (Regular)
        else:
            return 1 # Tier 3 (Outros)

    for col in cast_cols:
        X_train[f'{col}_tier'] = X_train[col].apply(lambda x: get_actor_tier(x, actor_counts, q1, q2))
        X_test[f'{col}_tier'] = X_test[col].apply(lambda x: get_actor_tier(x, actor_counts, q1, q2))

    # Criar uma única feature de "força do elenco"
    X_train['Cast_Power'] = X_train[[f'{col}_tier' for col in cast_cols]].max(axis=1)
    X_test['Cast_Power'] = X_test[[f'{col}_tier' for col in cast_cols]].max(axis=1)

    # Remover as colunas originais e de tier
    cols_to_drop_post_feature_eng = cast_cols + [f'{col}_tier' for col in cast_cols]
    X_train = X_train.drop(columns=cols_to_drop_post_feature_eng)
    X_test = X_test.drop(columns=cols_to_drop_post_feature_eng)
    print("Feature 'Cast_Power' criada e colunas originais de elenco removidas.")

    print(f"Dados divididos em treino ({X_train.shape[0]} amostras) e teste ({X_test.shape[0]} amostras).")

    # 4. Codificação e Padronização
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # Target Encoding para variáveis categóricas
    # O encoder aprende a média da variável alvo para cada categoria APENAS nos dados de treino
    encoder = ce.TargetEncoder(cols=categorical_cols)
    X_train_encoded = encoder.fit_transform(X_train, y_train)
    X_test_encoded = encoder.transform(X_test)

    # Padronização (Standard Scaling)
    # Coloca todas as variáveis na mesma escala (média 0, desvio padrão 1)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_encoded), columns=X_train_encoded.columns, index=X_train_encoded.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_encoded), columns=X_test_encoded.columns, index=X_test_encoded.index)
    print("Codificação de variáveis categóricas e padronização concluídas.")

    # 5. Análise de Multicolinearidade (VIF)
    # Hair (2009) recomenda verificar o VIF para garantir que as variáveis preditoras não sejam redundantes.
    # Um VIF > 10 é geralmente considerado um sinal de alta multicolinearidade.
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_train_scaled.columns
    vif_data["VIF"] = [variance_inflation_factor(X_train_scaled.values, i) for i in range(len(X_train_scaled.columns))]
    print("\n--- Análise de Fator de Inflação de Variância (VIF) ---")
    print(vif_data)

    # 6. Treinamento do Modelo com Statsmodels para Análise Inferencial
    X_train_sm = sm.add_constant(X_train_scaled) # Adiciona o intercepto
    model = sm.OLS(y_train, X_train_sm).fit()
    print("\n--- OLS Regression Results ---")
    print(model.summary())

    # 7. Avaliação do Modelo no Conjunto de Teste
    X_test_sm = sm.add_constant(X_test_scaled)
    y_pred = model.predict(X_test_sm)

    # Extraindo a equação do modelo treinado
    print("\n--- Equação da Regressão Linear ---")
    coefs = model.params
    equation = f"Public_Total = {coefs['const']:.4f} "
    for feature, coef in coefs.drop('const').items():
        equation += f"+ ({coef:.4f} * {feature}) "
    print(equation)

    # Validação Cruzada com Scikit-learn para uma avaliação mais robusta do R² no treino
    lr_model_cv = LinearRegression()
    scores = cross_val_score(lr_model_cv, X_train_scaled, y_train, cv=5, scoring='r2')

    # Métricas de avaliação
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 Resultados da Regressão Linear:")
    print(f"🔹 Mean Squared Error (MSE) no teste: {mse:.2f}")
    print(f"🔹 R² no teste: {r2:.2f}")
    print(f"🔹 Cross-Validation R² Scores (no treino): {scores}")
    print(f"🔹 Média dos R² (CV no treino): {scores.mean():.3f}")

    # 8. Diagnóstico de Resíduos
    # Hair (2009) enfatiza a importância de analisar os resíduos para validar as premissas da regressão.
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Gráfico de Resíduos vs. Valores Previstos: idealmente, pontos aleatórios em torno da linha zero.
    sns.scatterplot(x=y_pred, y=residuals, ax=axes[0], alpha=0.6)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_title('Gráfico de Resíduos vs. Valores Previstos')
    axes[0].set_xlabel('Valores Previstos (Log do Público)')
    axes[0].set_ylabel('Resíduos')

    # Gráfico Q-Q: idealmente, os pontos devem seguir a linha diagonal vermelha.
    sm.qqplot(residuals, line='s', ax=axes[1])
    axes[1].set_title('Gráfico de Probabilidade Normal (Q-Q Plot)')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    runLinearRegressor()
