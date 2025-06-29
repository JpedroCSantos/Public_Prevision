import pandas as pd
import numpy as np
import statsmodels.api as sm
import category_encoders as ce
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
        'Director_1', 'Release_Date', 'budget',
        # Removendo temporariamente para teste de ablação (isolar impacto)
        'Prodution_country', 'runtime'
    ]
    df = df.drop(columns=vars_to_drop, errors='ignore')
    print(f"Shape após remoção de variáveis para teste de ablação: {df.shape}")
    
    # Simplificando a coluna de produtoras para usar apenas a primeira.
    # Assumimos que a primeira da lista é a principal.
    if 'Production_Companies' in df.columns:
        df['Production_Companies'] = df['Production_Companies'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else x)

    # Transformação logarítmica da variável alvo
    df['Public_Total'] = np.log1p(df['Public_Total'])

    # 2. Separação em Conjuntos de Treino e Teste
    X = df.drop('Public_Total', axis=1)
    y = df['Public_Total']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    
    # ANÁLISE DE CARDINALIDADE E AGRUPAMENTO DE CATEGORIAS RARAS
    # Para evitar a criação de um número excessivo de colunas com o OneHotEncoder
    # e melhorar a performance, vamos agrupar as produtoras menos frequentes.
    if 'Production_Companies' in X_train.columns:
        # 1. Identificar as N produtoras mais comuns APENAS no conjunto de treino
        top_companies = X_train['Production_Companies'].value_counts().nlargest(20).index.tolist()
        
        # 2. Substituir as produtoras raras por 'Other' em ambos os conjuntos
        X_train['Production_Companies'] = X_train['Production_Companies'].apply(lambda x: x if x in top_companies else 'Other')
        X_test['Production_Companies'] = X_test['Production_Companies'].apply(lambda x: x if x in top_companies else 'Other')
        print("Produtoras menos frequentes agrupadas em 'Other' (para uso no OneHotEncoder).")
    
    # 3. Engenharia de Features: Níveis de Elenco (Pós-Split)
    print("Iniciando engenharia de features: Níveis de Elenco (Cast_Power)...")
    cast_cols = ['Cast_1', 'Cast_2', 'Cast_3']
    
    # Garantir que as colunas de elenco existam antes de prosseguir
    existing_cast_cols = [col for col in cast_cols if col in X_train.columns]

    if existing_cast_cols:
        # Contar a frequência de cada ator APENAS no conjunto de treino
        actor_counts = pd.concat([X_train[col] for col in existing_cast_cols]).value_counts()
        
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

        tier_cols = []
        for col in existing_cast_cols:
            tier_col_name = f'{col}_tier'
            X_train[tier_col_name] = X_train[col].apply(lambda x: get_actor_tier(x, actor_counts, q1, q2))
            X_test[tier_col_name] = X_test[col].apply(lambda x: get_actor_tier(x, actor_counts, q1, q2))
            tier_cols.append(tier_col_name)

        # Criar uma única feature de "força do elenco"
        X_train['Cast_Power'] = X_train[tier_cols].max(axis=1)
        X_test['Cast_Power'] = X_test[tier_cols].max(axis=1)

        # Remover as colunas originais e de tier
        cols_to_drop_post_feature_eng = existing_cast_cols + tier_cols
        X_train = X_train.drop(columns=cols_to_drop_post_feature_eng)
        X_test = X_test.drop(columns=cols_to_drop_post_feature_eng)
        print("Feature 'Cast_Power' criada e colunas originais de elenco removidas.")

    print(f"Dados divididos em treino ({X_train.shape[0]} amostras) e teste ({X_test.shape[0]} amostras).")

    # 4. Pipeline de Pré-processamento
    # Definindo as colunas numéricas e categóricas
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Colunas Numéricas: {numeric_features}")
    print(f"Colunas Categóricas: {categorical_features}")

    # Criando o transformador para as colunas
    # OneHotEncoder para as categóricas (robusto contra overfitting)
    # StandardScaler para as numéricas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # 5. Treinamento e Avaliação do Modelo
    # Usaremos um pipeline para encadear o pré-processamento e o modelo
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', LinearRegression())])

    # Treinando o pipeline
    model_pipeline.fit(X_train, y_train)
    print("Pipeline de pré-processamento e treinamento concluído.")
    
    # 6. Análise Inferencial com Statsmodels (após o treino do pipeline)
    print("\n--- Análise Inferencial do Modelo (Statsmodels) ---")
    try:
        # Extrair os nomes das features do pipeline
        numeric_features_list = list(numeric_features)
        all_feature_names = numeric_features_list # Começa com as numéricas

        # Apenas tenta extrair nomes do OneHotEncoder se ele foi treinado (se havia features categóricas)
        if categorical_features:
            ohe_feature_names = list(model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features))
            all_feature_names = numeric_features_list + ohe_feature_names
        
        # Transformar os dados de treino para o formato que o statsmodels espera
        X_train_transformed = model_pipeline.named_steps['preprocessor'].transform(X_train).toarray()
        X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=all_feature_names, index=X_train.index)
        
        # Adicionar a constante (intercepto) e treinar o modelo OLS
        X_train_sm = sm.add_constant(X_train_transformed_df)
        model_sm = sm.OLS(y_train, X_train_sm).fit()
        
        print(model_sm.summary())

    except Exception as e:
        print(f"Não foi possível gerar o sumário do Statsmodels: {e}")

    # 7. Avaliação do Modelo no Conjunto de Teste
    y_pred = model_pipeline.predict(X_test)
    
    # Validação Cruzada para uma avaliação mais robusta do R² no treino
    cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='r2')
    # Métricas de avaliação
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 Resultados da Regressão Linear (com Gênero e Produtora):")
    print(f"🔹 Mean Squared Error (MSE) no teste: {mse:.3f}")
    print(f"🔹 R² no teste: {r2:.3f}")
    print(f"🔹 Cross-Validation R² Scores (no treino): {cv_scores}")
    print(f"🔹 Média dos R² (CV no treino): {np.mean(cv_scores):.3f}")

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
    # runLinearRegressor()
    print ("testes")