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

def _run_diagnostics(model_sm, df_original, X_train, X_test, y_test, y_pred):
    """
    Executa e plota uma série de diagnósticos do modelo e de seus resíduos.
    """
    residuals = y_test - y_pred
    
    # --- Diagnóstico das Premissas ---
    print("\n--- Análise Diagnóstica do Modelo ---")
    
    # 1. Normalidade dos Resíduos
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.histplot(residuals, kde=True, ax=axes[0])
    axes[0].set_title('Histograma dos Resíduos')
    sm.qqplot(residuals, line='s', ax=axes[1])
    axes[1].set_title('Gráfico de Probabilidade Normal (Q-Q Plot)')
    fig.suptitle('Diagnóstico da Normalidade dos Resíduos', fontsize=16)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()

    # 2. Homocedasticidade dos Resíduos
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, ax=ax, alpha=0.6)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_title('Gráfico de Resíduos vs. Valores Previstos (Homocedasticidade)')
    ax.set_xlabel('Valores Previstos (Log do Público)')
    ax.set_ylabel('Resíduos')
    plt.show()

    # 3. Análise de Pontos Influentes (Alavancagem)
    fig, ax = plt.subplots(figsize=(8, 6))
    influence = model_sm.get_influence()
    # O gráfico de influência plota resíduos studentizados vs. alavancagem
    sm.graphics.influence_plot(model_sm, ax=ax, criterion="cooks")
    ax.set_title('Diagnóstico de Pontos Influentes (Resíduos vs. Alavancagem)')
    plt.tight_layout()
    plt.show()

    # 4. Normalidade das Variáveis Preditoras Numéricas
    numeric_features = X_train.select_dtypes(include=np.number).columns
    if not numeric_features.empty:
        print("\nVerificando a Normalidade das Variáveis Preditoras Numéricas (no treino)...")
        X_train[numeric_features].hist(bins=20, figsize=(15, 10), layout=(-1, 3))
        plt.suptitle('Distribuição das Variáveis Preditoras Numéricas')
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()

    # --- Análise Aprofundada dos Resíduos por Gênero ---
    print("\n--- Análise de Erros por Gênero ---")
    # Juntar os dados de teste com os resíduos para análise
    X_test_with_genre = X_test.copy()
    X_test_with_genre['residuals'] = residuals
    
    # O `Genre_1` foi transformado em dummies, precisamos do original.
    # Vamos buscar no dataframe original 'df' usando os índices de teste.
    original_genres = df_original.loc[X_test.index, 'Genre_1']
    X_test_with_genre['Genre_1_original'] = original_genres
    
    if not X_test_with_genre.empty:
        mean_residuals_by_genre = X_test_with_genre.groupby('Genre_1_original')['residuals'].mean().sort_values()
        print("Média dos Resíduos (Erro de Previsão) por Gênero:")
        print(mean_residuals_by_genre)

        fig, ax = plt.subplots(figsize=(12, 7))
        mean_residuals_by_genre.plot(kind='barh', ax=ax)
        ax.axvline(0, color='black', linestyle='--')
        ax.set_title('Média do Erro de Previsão por Gênero')
        ax.set_xlabel('Média dos Resíduos (Log do Público)')
        ax.set_ylabel('Gênero')
        plt.tight_layout()
    plt.show()


def runLinearRegressor(df: pd.DataFrame, run_diagnostics: bool = False):
    """
    Executa um fluxo completo de regressão linear com base no arquivo MOVIES.csv,
    seguindo as melhores práticas de pré-processamento e avaliação de modelo.
    
    Args:
        df (pd.DataFrame): O dataframe contendo os dados dos filmes.
        run_diagnostics (bool): Se True, executa e plota os gráficos de diagnóstico do modelo.
    """
    # 1. Carregamento e Limpeza Inicial dos Dados
    df = df.dropna().drop_duplicates()
    df_original_for_analysis = df.copy() # Salva uma cópia para análise de resíduos
    df = df.drop(['Title'], axis=1, errors='ignore')
    print(f"Formato após limpeza inicial (dropna, drop_duplicates): {df.shape}")

    # ===== Modelo Base Aprimorado =====
    # Variáveis a serem removidas APÓS a engenharia de features ou por não serem úteis.
    vars_to_drop_base = [
        'Runtime', 'Vote_Average', 'IMDB_Rating', 'Month', 'Day_of_Week_sin',
        'Release_Date', 'budget'
    ]
    df = df.drop(columns=vars_to_drop_base, errors='ignore')
    print(f"Shape após remoção de variáveis base: {df.shape}")
    
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

    # 4. Engenharia de Features: Níveis de Diretor (Pós-Split)
    print("Iniciando engenharia de features: Níveis de Diretor (Director_Power)...")
    if 'Director_1' in X_train.columns:
        # Contar a frequência de cada diretor APENAS no conjunto de treino
        director_counts = X_train['Director_1'].value_counts()
        
        # Definir os limiares para os tiers (quantis)
        q1_dir = director_counts.quantile(0.66)
        q2_dir = director_counts.quantile(0.95)
        
        def get_director_tier(director, counts, q1, q2):
            count = counts.get(director, 0)
            if count > q2:
                return 3 # Tier 1 (A-Lister Director)
            elif count > q1:
                return 2 # Tier 2 (Regular Director)
            else:
                return 1 # Tier 3 (Other/New Director)

        # Criar a feature de "força do diretor"
        X_train['Director_Power'] = X_train['Director_1'].apply(lambda x: get_director_tier(x, director_counts, q1_dir, q2_dir))
        X_test['Director_Power'] = X_test['Director_1'].apply(lambda x: get_director_tier(x, director_counts, q1_dir, q2_dir))

        # Remover a coluna original
        X_train = X_train.drop(columns=['Director_1'])
        X_test = X_test.drop(columns=['Director_1'])
        print("Feature 'Director_Power' criada e coluna original de diretor removida.")

    # 5. Engenharia de Features: Tiers de Mercado (Pós-Split) - CORRIGIDO
    print("Iniciando engenharia de features: Níveis de Mercado (Market_Tier)...")
    country_col = 'Prodution_country' # Corrigindo o nome da coluna
    if country_col in X_train.columns:
        # Definir os tiers de mercado com base no conhecimento de domínio
        major_markets = ['United Kingdom', 'France', 'Germany', 'Canada', 'Japan', 'China', 'India']

        def get_market_tier(country):
            if country == 'USA' or country == 'ESTADOS UNIDOS': # Acomodar variações
                return 3 # Tier 3 (Mercado Dominante)
            elif country in major_markets:
                return 2 # Tier 2 (Grandes Mercados Internacionais)
            else:
                return 1 # Tier 1 (Mercados Locais/Outros)

        # Criar a nova feature
        X_train['Market_Tier'] = X_train[country_col].apply(get_market_tier)
        X_test['Market_Tier'] = X_test[country_col].apply(get_market_tier)
        
        # Remover a coluna original
        X_train = X_train.drop(columns=[country_col, 'Production_Companies'], errors='ignore')
        X_test = X_test.drop(columns=[country_col, 'Production_Companies'], errors='ignore')
        print("Feature 'Market_Tier' criada e colunas originais removidas.")

    print(f"Dados divididos em treino ({X_train.shape[0]} amostras) e teste ({X_test.shape[0]} amostras).")

    # 6. Pipeline de Pré-processamento
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
    
    # 7. Treinamento e Avaliação do Modelo
    # Usaremos um pipeline para encadear o pré-processamento e o modelo
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', LinearRegression())])

    # Treinando o pipeline
    model_pipeline.fit(X_train, y_train)
    print("Pipeline de pré-processamento e treinamento concluído.")
    
    # 8. Análise Inferencial com Statsmodels (após o treino do pipeline)
    print("\n--- Análise Inferencial do Modelo (Statsmodels) ---")
    model_sm = None
    try:
        # Extrair os nomes das features após o pré-processamento
        feature_names = preprocessor.get_feature_names_out()
        
        # Dados de treino transformados
        X_train_processed = preprocessor.transform(X_train).toarray()
        X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
        X_train_processed = sm.add_constant(X_train_processed) # Adicionar o intercepto

        # Criar e treinar o modelo do statsmodels
        model_sm = sm.OLS(y_train, X_train_processed)
        model_sm = model_sm.fit(cov_type='HC3')

        print(model_sm.summary())

    except Exception as e:
        print(f"Não foi possível gerar o sumário do Statsmodels: {e}")

    # 9. Avaliação do Modelo no Conjunto de Teste
    print("\n--- Avaliação do Modelo no Conjunto de Teste ---")
    y_pred = model_pipeline.predict(X_test)
    r2_test = r2_score(y_test, y_pred)
    mse_test = mean_squared_error(y_test, y_pred)
    print(f"R² (Coeficiente de Determinação) no Teste: {r2_test:.4f}")
    print(f"Erro Quadrático Médio (MSE) no Teste: {mse_test:.4f}")

    # 10. Análise de Resíduos e Maiores Erros
    residuals = y_test - y_pred
    
    # Juntar os dados de teste originais com os resultados para identificar os filmes
    results_df = df_original_for_analysis.loc[y_test.index].copy()
    results_df['Public_Total_Real_Log'] = y_test
    results_df['Public_Total_Previsto_Log'] = y_pred
    results_df['Residual'] = residuals
    
    # Adicionar a feature Cast_Power que foi criada dinamicamente
    if 'Cast_Power' in X_test.columns:
        results_df['Cast_Power'] = X_test['Cast_Power']
        
    # Reverter a transformação log para ver os valores de público originais
    results_df['Public_Total_Real'] = np.expm1(results_df['Public_Total_Real_Log'])
    results_df['Public_Total_Previsto'] = np.expm1(results_df['Public_Total_Previsto_Log'])

    # Ordenar pelos maiores erros negativos (modelo foi otimista demais)
    maiores_erros = results_df.sort_values(by='Residual', ascending=True).head(15)
    
    print("\n--- Análise dos Filmes com Maiores Erros de Previsão (Resíduos Negativos) ---")
    print("O modelo previu um público muito maior do que o real para estes filmes:")
    
    cols_to_display = ['Title', 'Public_Total_Real', 'Public_Total_Previsto', 'Residual', 'Genre_1', 'Cast_Power']
    existing_cols_to_display = [col for col in cols_to_display if col in maiores_erros.columns]
    
    print(maiores_erros[existing_cols_to_display])

    # 11. Execução dos Diagnósticos (Opcional)
    if run_diagnostics and model_sm:
        print("\n--- Executando Diagnósticos do Modelo ---")
        _run_diagnostics(model_sm, df_original_for_analysis, X_train, X_test, y_test, y_pred)
    else:
        print("\nDiagnósticos do Statsmodels não foram executados (parâmetro 'run_diagnostics' é False ou modelo SM falhou).")


if __name__ == '__main__':
    # Carregando os dados
    df = pd.read_csv('data/output/dataset_final_com_publico.csv')
    
    # Executando o modelo e a análise com diagnósticos ativados
    runLinearRegressor(df, run_diagnostics=True)