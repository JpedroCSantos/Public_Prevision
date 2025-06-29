import re
from typing import Dict, List

import pandas as pd

from tqdm import tqdm


def concatenate_dataframes(dataframe_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Função para transformar uma lista de dataframes em um único dataframe.

    Args:
        dataframe_list (List[pd.DataFrame]): Lista de dataframes a serem concatenados.

    Returns:
        pd.DataFrame: DataFrame único.
    """
    if not dataframe_list:
        return pd.DataFrame()
    return pd.concat(dataframe_list, ignore_index=True)

def remove_coluns(df: pd.DataFrame, columns_to_remove: List[str]) -> pd.DataFrame:
    """
    Remove coluna(s) de um dataframe, baseados no nome da coluna(s).

    Args:
        df (pd.DataFrame): Dataframe.
        columns_to_remove (List[str]): Colunas a serem removidas.

    Returns:
        pd.DataFrame: DataFrame com as colunas removidas.
    """
    print("Removendo Colunas")
    return df.drop(columns=columns_to_remove, axis=1)

def correct_movie_titles(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Corrige os títulos dos filmes movendo os artigos do final para o início.
    Exemplo: "HOMEM QUE COPIAVA, O" se torna "O HOMEM QUE COPIAVA".
    """
    
    def format_title(title):
        if isinstance(title, str):
            # Procura por títulos que terminam com ", O", ", A", etc. (e variações de espaço/capitalização)
            match = re.match(r'^(.*),\s*(O|A|OS|AS)$', title.strip(), re.IGNORECASE)
            if match:
                main_title = match.group(1).strip()
                article = match.group(2).strip().upper()
                # Retorna o título formatado, ex: "O HOMEM QUE COPIAVA"
                return f"{article} {main_title}"
        return title

    df_copy = df.copy()
    df_copy[column_name] = df_copy[column_name].apply(format_title)
    return df_copy

def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma o DataFrame agrupando por CPB_ROE, somando o público e calculando dias de exibição.
    Esta versão otimizada substitui o loop manual por operações vetorizadas do Pandas
    para melhor performance e clareza.
    """
    print("Transformando DataFrame")

    df_copy = df.copy()

    df_copy['DATA_EXIBICAO'] = pd.to_datetime(df_copy['DATA_EXIBICAO'], format="%d/%m/%Y", dayfirst=True, errors='coerce')
    df_copy['PUBLICO'] = pd.to_numeric(df_copy['PUBLICO'], errors='coerce').fillna(0).astype(int)
    
    # Filtra o público e remove datas nulas que podem ter surgido do 'coerce'
    df_copy = df_copy.query('PUBLICO <= 500').dropna(subset=['DATA_EXIBICAO'])

    df_copy = df_copy.sort_values(by=['CPB_ROE', 'DATA_EXIBICAO'])

    # Otimização: Substituindo o loop por operações vetorizadas
    # Calcula a diferença em dias para a exibição anterior dentro do mesmo grupo de filme
    df_copy['DIAS_DESDE_ULTIMA_EXIBICAO'] = df_copy.groupby('CPB_ROE')['DATA_EXIBICAO'].diff().dt.days

    # Consideramos apenas as lacunas de menos de um ano para o cálculo de dias corridos.
    # Lacunas maiores são tratadas como uma nova "temporada" de exibição e não somam ao total.
    df_copy['DIAS_EM_EXIBICAO_CONTINUA'] = df_copy['DIAS_DESDE_ULTIMA_EXIBICAO'].where(df_copy['DIAS_DESDE_ULTIMA_EXIBICAO'] < 365, 0)
    
    # Agrupamos os dados por filme para obter os resultados finais
    df_resultado = df_copy.groupby('CPB_ROE').agg(
        PUBLICO_sum=('PUBLICO', 'sum'),
        DIAS_EM_EXIBICAO=('DIAS_EM_EXIBICAO_CONTINUA', 'sum'),
        PAIS_OBRA_first=('PAIS_OBRA', 'first'),
        Title_first=('TITULO_ORIGINAL', 'first'),
        Brazilian_Title_first=('TITULO_BRASIL', 'first'),
        FIRST_EXIBICAO=('DATA_EXIBICAO', 'min'), # Pega a primeira data de exibição
        number_of_exhibition_rooms=('REGISTRO_SALA', 'nunique') # Conta salas de exibição únicas
    ).reset_index()

    df_resultado = correct_movie_titles(df_resultado, 'Title_first')

    return df_resultado

def _split_Columns(df: pd.DataFrame, columns_to_split: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    df_copy = df.copy()
    for column, config in columns_to_split.items():
        if column not in df_copy:
            continue
        
        df_copy[column] = df_copy[column].fillna("").astype(str)
        number_columns = config["number_columns"]
        
        split_data = df_copy[column].str.split(",", n=number_columns - 1, expand=True)
        
        for i in range(number_columns):
            col_name = f"{column}_{i+1}"
            df_copy[col_name] = split_data[i].str.strip() if i in split_data else None

    df_copy = df_copy.drop(columns=list(columns_to_split.keys()))
    return df_copy

def remove_zero_or_nan_rows(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Remove linhas de um DataFrame onde as colunas especificadas possuem valor 0 ou NaN.
    """
    print(f"DataFrame antes da remoção de linhas: {len(df)}")
    df_filtered = df.dropna(subset=columns)
    
    for col in columns:
        if pd.api.types.is_numeric_dtype(df_filtered[col]):
            df_filtered = df_filtered[df_filtered[col] != 0]
            
    print(f"DataFrame depois da remoção de linhas: {len(df_filtered)}")
    return df_filtered

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy = df_copy.drop_duplicates(subset=['CPB_ROE'])

    df_copy = df_copy.rename(columns={
        'PUBLICO_sum': 'Public_Total',
        'FIRST_EXIBICAO': 'Release_Date',
        'PAIS_OBRA_first': 'Prodution_country',
        'DIAS_EM_EXIBICAO': 'Days_in_exibithion',
        'vote_average': 'Vote_Average',
        'number_of_exhibition_rooms': 'Number_of_exhibition_rooms',
        'belongs_to_collection': 'Belongs_to_collection',
        'Brazilian_Title_first': 'Brazilian_Title'
    })
    
    if 'imdb_id' in df_copy.columns:
        df_copy = df_copy.loc[df_copy.groupby('imdb_id')['Days_in_exibithion'].idxmax()]

    cols_to_drop = ['production_cost', 'vote_count', 'id', 'Genre_2', 'Genre_3', 
                    'popularity', 'Metascore', 'imdb_id', 'Rated', 'CPB_ROE', 
                    'release_date', 'Brazilian_Title']
    
    # Garante que só tentaremos dropar as colunas que realmente existem
    existing_cols_to_drop = [col for col in cols_to_drop if col in df_copy.columns]
    df_copy = df_copy.drop(columns=existing_cols_to_drop)

    cols_to_check = ['Public_Total','Release_Date', 'Prodution_country',
                     'Days_in_exibithion', 'Runtime', 'Vote_Average', 'Genre_1',
                     'production_companies', 'Director', 'Cast', 'IMDB_Rating',
                     'Number_of_exhibition_rooms']
    
    existing_cols_to_check = [col for col in cols_to_check if col in df_copy.columns]
    df_copy = remove_zero_or_nan_rows(df_copy, existing_cols_to_check)
    df_copy['Belongs_to_collection'] = df_copy['Belongs_to_collection'].fillna(0)
    
    columns_to_split = {
        "Cast": {"number_columns": 3},
        "Director": {"number_columns": 1},
    }
    df_copy = _split_Columns(df_copy, columns_to_split)
    return df_copy

def get_variable_dictionary(variable: str) -> str:
    VAR_DICTIONARY = {
        "Runtime": "Tempo de Execução",
        "IMDB_Rating": "Avaliação do Público",
        "Vote_Average": "Avaliação dos Críticos",
        "Days_in_exibithion": "Dias em exibição",
        "Genre_1": "Gênero",
        "Prodution_country": "País de Produção",
        "production_companies": "Empresa Produtora",
        "Director_1": "Diretor",
        "Cast_1": "Ator Principal",
        "Public_Total": "Publico Total",
    }
    return VAR_DICTIONARY.get(variable, "Variável não encontrada")


if __name__ == "__main__":
    # Este bloco é para teste e depuração.
    # Certifique-se de que a função `read_data_csv` está disponível no módulo `extract`.
    try:
        from extract import read_data_csv
        
        df_list = read_data_csv("data/input/bilheteria-diaria-obras-por-exibidoras-csv")
        concatenated_df = concatenate_dataframes(df_list)
        transformed_df = transform_dataframe(concatenated_df)
        print(transformed_df.head())
    except ImportError:
        print("A função `read_data_csv` não pôde ser importada de `extract`.")
    except FileNotFoundError:
        print("O diretório de dados de entrada não foi encontrado. Execute este script da raiz do projeto.")
