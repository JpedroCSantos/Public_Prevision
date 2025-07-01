import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from category_encoders import TargetEncoder
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

def get_vif(X_processed, feature_names):
    """
    Calcula o Fator de Inflação de Variância (VIF) para cada variável preditora.
    """
    vif_data = pd.DataFrame()
    vif_data["feature"] = feature_names
    vif_data["VIF"] = [variance_inflation_factor(X_processed, i) for i in range(X_processed.shape[1])]
    return vif_data.sort_values(by='VIF', ascending=False)

def print_regression_equation(model, feature_names):
    """Imprime a equação da regressão de forma legível."""
    try:
        intercept = model.params['const']
        coeffs = model.params.drop('const')
    except KeyError:
        print("Modelo sem intercepto (constante). A equação pode estar incompleta.")
        intercept = 0
        coeffs = model.params

    equation = f"Log(Public_Total) = {intercept:.4f} "
    
    # Limita o número de termos para não poluir a saída
    terms_to_show = min(len(feature_names), 7)
    
    for i in range(terms_to_show):
        name = feature_names[i]
        coeff = coeffs.iloc[i]
        if coeff >= 0:
            equation += f"+ {coeff:.3f} * ({name}) "
        else:
            equation += f"- {abs(coeff):.3f} * ({name}) "

    if len(feature_names) > terms_to_show:
        equation += "+ ..."
        
    print("--- Equação da Regressão (Parcial) ---")
    print(equation)
    print("\n---> DIAGNÓSTICO: A equação mostra como cada variável é ponderada pelo modelo.")
    print("No entanto, com tantas variáveis, é difícil de interpretar, e os altos p-valores e VIFs tornam os coeficientes pouco confiáveis.\n")


def plot_diagnostic_plots(sm_model, vif_df):
    """
    Gera e exibe gráficos de diagnóstico para o modelo de regressão.
    1. Resíduos vs. Valores Previstos (para heterocedasticidade)
    2. Q-Q Plot dos Resíduos (para normalidade)
    3. Histograma dos Resíduos (para normalidade)
    4. Gráfico de Barras do VIF (para multicolinearidade)
    """
    residuals = sm_model.resid
    fitted_values = sm_model.fittedvalues

    plt.style.use('seaborn-v2-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Gráficos de Diagnóstico do Modelo', fontsize=20)

    # 1. Resíduos vs. Valores Previstos
    sns.scatterplot(x=fitted_values, y=residuals, ax=axes[0, 0], alpha=0.6)
    axes[0, 0].axhline(0, ls='--', color='red')
    axes[0, 0].set_title('Resíduos vs. Valores Previstos', fontsize=14)
    axes[0, 0].set_xlabel('Valores Previstos (Log)', fontsize=12)
    axes[0, 0].set_ylabel('Resíduos (Log)', fontsize=12)
    axes[0, 0].text(0.95, 0.95, 'Padrão de funil indica Heterocedasticidade',
                    verticalalignment='top', horizontalalignment='right',
                    transform=axes[0, 0].transAxes, color='red', fontsize=10)

    # 2. Q-Q Plot
    sm.qqplot(residuals, line='s', ax=axes[0, 1], fit=True)
    axes[0, 1].set_title('Q-Q Plot dos Resíduos', fontsize=14)
    axes[0, 1].get_lines()[1].set_color('red')
    axes[0, 1].get_lines()[0].set_markerfacecolor('C0')
    axes[0, 1].get_lines()[0].set_markeredgecolor('C0')

    # 3. Histograma dos Resíduos
    sns.histplot(residuals, kde=True, ax=axes[1, 0], bins=30)
    axes[1, 0].set_title('Histograma dos Resíduos', fontsize=14)
    axes[1, 0].set_xlabel('Resíduos', fontsize=12)

    # 4. VIF Bar Plot
    vif_plot_df = vif_df.head(15) # Plot top 15 VIFs
    sns.barplot(x='VIF', y='feature', data=vif_plot_df, ax=axes[1, 1], palette='viridis')
    axes[1, 1].set_title('Top 15 Fatores de Inflação de Variância (VIF)', fontsize=14)
    axes[1, 1].axvline(x=5, color='r', linestyle='--', label='Tolerável (5)')
    axes[1, 1].axvline(x=10, color='darkred', linestyle='--', label='Problemático (10)')
    axes[1, 1].legend()
    
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    print("--- Gráficos de Diagnóstico ---")
    print("Exibindo gráficos... Feche a janela de plotagem para continuar.")
    plt.show()


def run_scenario_1(df: pd.DataFrame):
    """
    Simula o Cenário 1: O Modelo Inicial com Overfitting.
    
    Este cenário recria o primeiro modelo que construímos, que utilizava
    TargetEncoder em variáveis de alta cardinalidade.
    
    Problemas a serem observados:
    1. Overfitting: R² de treino muito alto e R² de teste muito baixo.
    2. Multicolinearidade: VIFs elevados para as variáveis de elenco.
    3. P-valores: Muitas variáveis não eram estatisticamente significativas.
    """
    print("="*80)
    print("Executing Scenario 1: The Initial Overfit Model")
    print("="*80)

    # 1. Definição das variáveis
    # Variáveis de alta cardinalidade que causarão overfitting com TargetEncoder
    high_cardinality_features = ['Director_1', 'Cast_1', 'Cast_2', 'Cast_3']
    
    # Variáveis categóricas de baixa cardinalidade
    categorical_features = ['Genre_1', 'Prodution_country', 'Belongs_to_collection']
    
    # Variáveis numéricas
    numerical_features = [
        'Days_in_exibithion', 
        'Number_of_exhibition_rooms',
        'Runtime',
        'Vote_Average'
        ]

    # Variável alvo
    target = 'Public_Total'

    # Removendo linhas com valores nulos essenciais
    features = high_cardinality_features + categorical_features + numerical_features
    df_scenario = df[features + [target]].dropna()

    X = df_scenario[features]
    y = df_scenario[target]

    # Aplicando transformação logarítmica na variável alvo
    y_log = np.log1p(y)

    # 2. Divisão dos dados (Train/Test Split)
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

    # 3. Criação do Pipeline de Pré-processamento
    # Pipeline para variáveis numéricas
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Pipeline para variáveis categóricas de alta cardinalidade
    high_card_transformer = Pipeline(steps=[
        ('target_encoder', TargetEncoder(handle_missing='value', handle_unknown='value'))
    ])

    # Pipeline para variáveis categóricas de baixa cardinalidade
    low_card_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # ColumnTransformer para aplicar transformações diferentes em colunas diferentes
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('high_card', high_card_transformer, high_cardinality_features),
            ('low_card', low_card_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # 4. Criação e Treinamento do Modelo Completo (Pipeline)
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    model_pipeline.fit(X_train, y_train)

    # 5. Avaliação do Desempenho (R²)
    # R² no treino (usando validação cruzada para um resultado mais robusto)
    train_r2_cv = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='r2').mean()
    
    # R² no teste
    test_r2 = model_pipeline.score(X_test, y_test)

    print(f"--- Model Performance ---")
    print(f"Train R² (Cross-Validation): {train_r2_cv:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    print("\n---> DIAGNOSIS: Severe overfitting detected! The model performs well on data it has seen,")
    print("but fails to generalize to new, unseen data.\n")

    # 6. Análise de Coeficientes e VIF com Statsmodels
    # Para o VIF e p-valores, precisamos dos dados processados
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    
    # Para statsmodels e VIF, precisamos de uma matriz densa.
    if hasattr(X_train_processed, 'toarray'):
        X_train_processed_dense = X_train_processed.toarray()
    else:
        X_train_processed_dense = X_train_processed
    
    # Recuperando os nomes das features após o one-hot encoding
    try:
        ohe_feature_names = preprocessor.named_transformers_['low_card']['onehot'].get_feature_names_out(categorical_features)
    except AttributeError: # Para versões mais antigas do scikit-learn
        ohe_feature_names = preprocessor.named_transformers_['low_card']['onehot'].get_feature_names(categorical_features)
        
    all_feature_names = numerical_features + high_cardinality_features + list(ohe_feature_names)
    
    # Adicionando a constante para o modelo do statsmodels
    X_train_processed_sm = sm.add_constant(X_train_processed_dense, has_constant='add')

    # Ajustando o modelo OLS
    sm_model = sm.OLS(y_train, X_train_processed_sm).fit()

    # 7. Exibição dos Resultados Detalhados
    print_regression_equation(sm_model, all_feature_names)

    print("--- Análise de Significância (P>|t|) ---")
    print(sm_model.summary())
    print("\n---> DIAGNÓSTICO: Note os altos valores de P>|t| para muitas variáveis (ex: Runtime),")
    print("indicando que não são estatisticamente significantes neste modelo.\n")
    
    print("--- Análise de Multicolinearidade (VIF) ---")
    # Para o cálculo do VIF, usamos a matriz densa
    vif_df = get_vif(X_train_processed_dense, all_feature_names)
    print(vif_df.head(10)) # Mostra as 10 maiores
    print("\n---> DIAGNÓSTICO: Valores de VIF para Cast_2 e Cast_3 estão altos, indicando multicolinearidade.")
    print("O modelo tem dificuldade em distinguir o impacto individual deles.")

    # 8. Gráficos de Diagnóstico
    plot_diagnostic_plots(sm_model, vif_df)


def run_scenario_dimensionality_curse(df: pd.DataFrame):
    """
    Simula o Experimento 1: A Maldição da Dimensionalidade.
    
    Este cenário utiliza OneHotEncoder em todas as variáveis categóricas,
    incluindo as de altíssima cardinalidade como país e produtoras.
    
    Problemas a serem observados:
    1. Baixo poder preditivo: R² de teste muito baixo (~0.2).
    2. Dimensionalidade excessiva: O número de features explode, tornando o modelo
       instável e difícil de generalizar.
    """
    print("="*80)
    print("Executing Scenario: The Curse of Dimensionality (Experiment 1)")
    print("="*80)

    # 1. Definição das variáveis
    # Variáveis categóricas, incluindo as de alta cardinalidade
    categorical_features = ['Genre_1', 'Prodution_country', 'Production_Companies', 'Belongs_to_collection']
    
    # Variáveis numéricas
    numerical_features = [
        'Days_in_exibithion', 
        'Number_of_exhibition_rooms',
        'Runtime',
        'Vote_Average'
    ]

    # Variável alvo
    target = 'Public_Total'

    # Removendo linhas com valores nulos essenciais
    features = categorical_features + numerical_features
    df_scenario = df[features + [target]].copy()
    
    # Simplificação: Tratar nulos em 'Production_Companies' como uma categoria 'Unknown'
    if 'Production_Companies' in df_scenario.columns:
        df_scenario['Production_Companies'] = df_scenario['Production_Companies'].fillna('Unknown')
    df_scenario.dropna(inplace=True)


    X = df_scenario[features]
    y = df_scenario[target]

    # Aplicando transformação logarítmica na variável alvo
    y_log = np.log1p(y)

    # 2. Divisão dos dados (Train/Test Split)
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

    # 3. Criação do Pipeline de Pré-processamento
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    
    # OneHotEncoder para todas as variáveis categóricas
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first')) # Ignora categorias raras
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # 4. Criação e Treinamento do Modelo
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    model_pipeline.fit(X_train, y_train)

    # 5. Avaliação do Desempenho (R²)
    train_r2_cv = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='r2').mean()
    test_r2 = model_pipeline.score(X_test, y_test)

    print(f"--- Model Performance ---")
    print(f"Train R² (Cross-Validation): {train_r2_cv:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    print("\n---> DIAGNÓSTICO: O R² do teste é extremamente baixo. A criação de centenas de colunas")
    print("a partir de países e produtoras destruiu a capacidade de generalização do modelo.\n")

    # 6. Análise de Coeficientes com Statsmodels
    X_train_processed = model_pipeline.named_steps['preprocessor'].fit_transform(X_train, y_train)
    
    if hasattr(X_train_processed, 'toarray'):
        X_train_processed_dense = X_train_processed.toarray()
    else:
        X_train_processed_dense = X_train_processed

    try:
        ohe_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
    except AttributeError:
        ohe_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat']['onehot'].get_feature_names(categorical_features)
        
    all_feature_names = numerical_features + list(ohe_feature_names)
    
    print(f"Número de features criadas pelo OneHotEncoder: {len(ohe_feature_names)}")
    print(f"Número total de preditores no modelo: {len(all_feature_names)}\n")

    X_train_processed_sm = sm.add_constant(X_train_processed_dense, has_constant='add')
    sm_model = sm.OLS(y_train, X_train_processed_sm).fit()

    # 7. Exibição dos Resultados
    print_regression_equation(sm_model, all_feature_names)
    print("--- Análise de Significância (P>|t|) ---")
    print(sm_model.summary())
    
    # 8. Gráficos de Diagnóstico
    vif_df = get_vif(X_train_processed_dense, all_feature_names)
    plot_diagnostic_plots(sm_model, vif_df)


if __name__ == '__main__':
    try:
        # Carregar o dataset
        DATA_PATH = "data/output"
        FINAL_PATH = f"{DATA_PATH}/DATAS/FINAL_DATABASE"
        FINAL_FILE_NAME = "MOVIES"
        DF_FILE = f"{FINAL_PATH}/{FINAL_FILE_NAME}.parquet"
        df_full = pd.read_parquet(DF_FILE)
        
        # Executar a simulação do primeiro cenário (Desativado)
        # run_scenario_1(df_full)
        
        # Executar a simulação do Experimento 1: Maldição da Dimensionalidade
        run_scenario_dimensionality_curse(df_full)
        
    except FileNotFoundError:
        print(f"Erro: O arquivo '{DF_FILE}' não foi encontrado.")
        print("Por favor, certifique-se de que o dataset foi gerado e está no caminho correto.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 