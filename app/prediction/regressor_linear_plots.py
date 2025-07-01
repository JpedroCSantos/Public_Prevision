import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dicionário para mapear nomes de colunas para nomes usados no trabalho
COLUMN_NAME_MAPPER = {
    'Public_Total': 'Público Total',
    'Days_in_exibithion': 'Dias em Exibição',
    'Number_of_exhibition_rooms': 'Salas de Exibição',
    'Runtime': 'Duração do Filme',
    'Vote_Average': 'Avaliação Média',
    'Cast_Power': 'Poder do Elenco',
    'Director_Popularity': 'Popularidade do Diretor',
    'Budget': 'Orçamento',
    'belongs_to_collection': 'Pertence a uma Coleção',
    # Adicione outros mapeamentos conforme necessário
}

def plot_numeric_distributions(df: pd.DataFrame):
    """
    Plota a distribuição de todas as variáveis numéricas no dataframe.
    """
    print("Gerando gráfico da distribuição das variáveis numéricas...")
    numeric_df = df.select_dtypes(include=np.number)
    # Remove a variável de orçamento dos plots, tratando possíveis nomes
    numeric_df = numeric_df.drop(columns=['Budget', 'budget'], errors='ignore')
    
    # Renomeia as colunas para os plots
    plot_df = numeric_df.rename(columns=COLUMN_NAME_MAPPER)
    
    # Plota os histogramas
    axes = plot_df.hist(bins=20, figsize=(20, 15), layout=(-1, 4))
    
    # Garante que os títulos dos subplots não fiquem sobrepostos
    for ax_row in axes:
        for ax in ax_row:
            ax.set_title(ax.get_title(), pad=20) # Adiciona padding

    plt.suptitle('Distribuição das Variáveis Preditivas Numéricas', fontsize=20)
    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    plt.show()

def plot_boxplot_comparison(df: pd.DataFrame):
    """
    Gera dois boxplots da variável Public_Total em figuras separadas:
    1. Com a distribuição original (incluindo outliers).
    2. Com os outliers removidos pelo método IQR, para visualização.
    """
    target_variable_name = COLUMN_NAME_MAPPER.get('Public_Total', 'Public_Total')

    print("Gerando boxplot da distribuição original...")
    sns.set_theme(style="whitegrid")

    # --- 1. Boxplot Original ---
    plt.figure(figsize=(8, 8))
    sns.boxplot(data=df, y='Public_Total', color='skyblue')
    plt.title(f'Boxplot de "{target_variable_name}"\\n1. Distribuição Original (Com Outliers)', fontsize=16)
    plt.ylabel(f'{target_variable_name} (Escala Original)')
    plt.ticklabel_format(style='plain', axis='y')
    plt.tight_layout()
    plt.show()

    # --- 2. Boxplot com Outliers Removidos (para visualização) ---
    print("Gerando boxplot da distribuição sem outliers (para visualização)...")
    Q1 = df['Public_Total'].quantile(0.25)
    Q3 = df['Public_Total'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df['Public_Total'] >= lower_bound) & (df['Public_Total'] <= upper_bound)]

    plt.figure(figsize=(8, 8))
    sns.boxplot(data=df_filtered, y='Public_Total', color='mediumseagreen')
    plt.title(f'Boxplot de "{target_variable_name}"\\n2. Visualização Sem Outliers (Método IQR)', fontsize=16)
    plt.ylabel(f'{target_variable_name} (Escala Reduzida)')
    plt.ticklabel_format(style='plain', axis='y')
    plt.tight_layout()
    plt.show()

def plot_target_variable_transformation(df: pd.DataFrame):
    """
    Plota os histogramas do "antes e depois" da transformação logarítmica na variável alvo.
    """
    target_variable_name = COLUMN_NAME_MAPPER.get('Public_Total', 'Public_Total')
    
    print("Gerando gráfico da transformação da variável alvo...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Tratamento de Outliers na Variável Dependente via Transformação Logarítmica', fontsize=16)

    sns.histplot(data=df, x='Public_Total', kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Antes da Transformação (Distribuição Original)')
    axes[0].set_xlabel(f'{target_variable_name} (Escala Original)')
    axes[0].set_ylabel('Frequência')
    axes[0].ticklabel_format(style='plain', axis='x')

    df_transformed = df.copy()
    df_transformed['Public_Total_Log'] = np.log1p(df_transformed['Public_Total'])
    sns.histplot(data=df_transformed, x='Public_Total_Log', kde=True, ax=axes[1], color='salmon')
    axes[1].set_title('Depois da Transformação Logarítmica (log1p)')
    axes[1].set_xlabel(f'Log({target_variable_name} + 1)')
    axes[1].set_ylabel('Frequência')

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()

if __name__ == '__main__':
    try:
        DATA_PATH = "data/output"
        FINAL_PATH = f"{DATA_PATH}/DATAS/FINAL_DATABASE"
        FINAL_FILE_NAME = "MOVIES"
        DF_FILE = f"{FINAL_PATH}/{FINAL_FILE_NAME}.parquet"
        df = pd.read_parquet(DF_FILE)
        
        # Gerar os gráficos de EDA
        plot_numeric_distributions(df)
        plot_boxplot_comparison(df)
        plot_target_variable_transformation(df)
        
    except FileNotFoundError:
        print(f"Erro: O arquivo '{DF_FILE}' não foi encontrado.")
        print("Por favor, certifique-se de que o dataset foi gerado e está no caminho correto.")