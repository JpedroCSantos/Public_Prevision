import statsmodels.api as sm
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline.transform import get_variable_dictionary

def analyze_distribution(df, independent_vars):
    """
    Realiza análise estatística e visual das variáveis.

    Parâmetros:
    df (pd.DataFrame): DataFrame com os dados.
    dependent_var (str): Nome da variável dependente.
    independent_vars (list): Lista com os nomes das variáveis independentes.

    Retorna:
    dict: Resultados dos testes de normalidade.
    """
    results = {}

    print("\n--- Análise de Normalidade ---")
    normality_results = {}
    
    for col in independent_vars:
        stat, p_value = stats.shapiro(df[col])
        normality_results[col] = {"statistic": stat, "p_value": p_value}
        print(f"Teste de Shapiro-Wilk para {col}: Estatística={stat:.3f}, p-valor={p_value:.3f}")
        if p_value > 0.05:
            print(f"✅ Os dados de '{col}' parecem seguir uma distribuição normal.")
        else:
            print(f"⚠️ Os dados de '{col}' não seguem uma distribuição normal.")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[col], bins=30, kde=True, ax=axes[0])
        column_name = get_variable_dictionary(col)
        axes[0].set_title(f'Histograma de {column_name}')
        axes[0].set_xlabel(column_name)
        axes[0].set_ylabel('Frequência')

        sm.qqplot(df[col], line='s', ax=axes[1])
        axes[1].set_title(f'Q-Q Plot de {column_name}')

        plt.tight_layout()
        plt.show()

    results["normality"] = normality_results

    return results

def analyze_homoscedasticity(df, dependent_var, independent_vars):
    """
    Realiza análise estatística e visual das variáveis.

    Parâmetros:
    df (pd.DataFrame): DataFrame com os dados.
    dependent_var (str): Nome da variável dependente.
    independent_vars (list): Lista com os nomes das variáveis independentes.

    Retorna:
    dict: Resultados dos testes de homocedasticidade.
    """
    results = {}

    print("\n--- Análise de Homocedasticidade ---")
    X = sm.add_constant(df[independent_vars])
    y = df[dependent_var]
    model = sm.OLS(y, X).fit()
    test_stat, p_value, _, _ = sm.stats.diagnostic.het_breuschpagan(model.resid, X)

    print(f"Teste de Breusch-Pagan: Estatística={test_stat:.3f}, p-valor={p_value:.3f}")
    if p_value > 0.05:
        print("✅ Os resíduos apresentam homocedasticidade.")
    else:
        print("⚠️ Os resíduos apresentam heterocedasticidade.")
        
    results["homoscedasticity"] = {"statistic": test_stat, "p_value": p_value}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(model.fittedvalues, model.resid, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Valores Preditos')
    axes[0].set_ylabel('Resíduos')
    axes[0].set_title('Resíduos vs Valores Preditos')

    sns.histplot(model.resid, bins=30, kde=True, ax=axes[1])
    axes[1].set_title('Distribuição dos Resíduos')
    axes[1].set_xlabel('Resíduos')

    plt.tight_layout()
    plt.show()

    return results