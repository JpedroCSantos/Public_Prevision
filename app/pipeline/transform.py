import re
from typing import Dict, List

import pandas as pd

from tqdm import tqdm


def concatenate_dataframes(dataframe_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    função para transformar uma lista de dataframes em um único dataframe

    args: list_dataframes (List): Lista de dataframes a serem concatenados

    return: dataframe
    """
    concatenated_dataframe = pd.DataFrame()

    for df in tqdm(dataframe_list, desc="Concatenando DataFrames"):
        concatenated_dataframe = pd.concat([concatenated_dataframe, df], ignore_index=True)

    return concatenated_dataframe

def remove_coluns(df: pd.DataFrame, columns_to_remove: List[str]):
    """
    Remove coluna(s) de um dataframe, baseados no nome da coluna(s).

    args: df(pd.dataframe): Dataframe
          columns_to_remove (str or List): Coluna a ser removida

    return: dataframe
    """
    print("Removendo Colunas")
    return df.drop(columns_to_remove, axis=1)

def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma o DataFrame agrupando por CPB_ROE, somando o público e calculando dias de exibição.
    Filtra interrupções maiores que 2 semanas e descarta linhas com público acima de 500.
    Modificação para contar dias de exibição de acordo com a nova abordagem.
    """
    print("Transformando DataFrame")

    df = df.copy()

    df['DATA_EXIBICAO'] = pd.to_datetime(df['DATA_EXIBICAO'], format="%d/%m/%Y", dayfirst=True, errors='coerce')
    df['PUBLICO'] = pd.to_numeric(df['PUBLICO'], errors='coerce').fillna(0)
    df = df[df['PUBLICO'] <= 500]
    # df['EXIBICAO_UNICA'] = df.groupby('CPB_ROE')['DATA_EXIBICAO'].transform(lambda x: x.nunique())
    df.loc[:, 'EXIBICAO_UNICA'] = df.groupby('CPB_ROE')['DATA_EXIBICAO'].transform(lambda x: x.nunique())
    df = df.sort_values(by=['CPB_ROE', 'DATA_EXIBICAO'])

    df['FIRST_EXIBICAO'] = df.groupby('CPB_ROE')['DATA_EXIBICAO'].transform('min')
    df['DIAS_EM_EXIBICAO'] = 0
    for cpb, group in df.groupby('CPB_ROE'):
        last_exhibition = None
        total_days = 0

        for idx, row in group.iterrows():
            if last_exhibition is not None:
                days_diff = (row['DATA_EXIBICAO'] - last_exhibition).days
                if days_diff > 365:  # Se passar mais de 365 dias sem exibição, interrompe o cálculo
                    continue
                total_days += days_diff
            last_exhibition = row['DATA_EXIBICAO']
        df.loc[group.index, 'DIAS_EM_EXIBICAO'] = total_days

    df_resultado = df.groupby('CPB_ROE').agg(
        PUBLICO_sum=('PUBLICO', 'sum'),
        DIAS_EM_EXIBICAO=('DIAS_EM_EXIBICAO', 'first'),
        PAIS_OBRA_first=('PAIS_OBRA', 'first'),
        Title_first=('TITULO_ORIGINAL', 'first'),
        FIRST_EXIBICAO=('FIRST_EXIBICAO', 'first')
    ).reset_index()

    return df_resultado

# def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Transforma o DataFrame agrupando por CPB_ROE, somando o público e calculando dias de exibição.
#     Filtra interrupções maiores que 2 semanas e descarta linhas com público acima de 500.
#     Modificação para contar dias de exibição de acordo com a nova abordagem.
#     """
#     print("Transformando DataFrame")

#     df['DATA_EXIBICAO'] = pd.to_datetime(df['DATA_EXIBICAO'], format="%d/%m/%Y", dayfirst=True, errors='coerce')
#     df['PUBLICO'] = pd.to_numeric(df['PUBLICO'], errors='coerce').fillna(0)
#     df = df[df['PUBLICO'] <= 500]

#     df['EXIBICAO_UNICA'] = df.groupby('CPB_ROE')['DATA_EXIBICAO'].transform(lambda x: x.nunique())
#     df_resultado = df.groupby('CPB_ROE').agg(
#         PUBLICO_sum=('PUBLICO', 'sum'),
#         EXIBICAO_UNICA=('EXIBICAO_UNICA', 'first'),
#         PAIS_OBRA_first=('PAIS_OBRA', 'first'),
#         Title_first=('TITULO_ORIGINAL', 'first')
#     ).reset_index()

#     df_resultado = df_resultado.rename(columns={'EXIBICAO_UNICA': 'DIAS_EM_EXIBICAO'})

#     return df_resultado


def _split_Columns(df, columns_to_split):
    for column, config in columns_to_split.items():
        df.loc[:, column] = df[column].fillna("")
        number_columns = config["number_columns"]
        df.loc[:, column] = df[column].apply(lambda x: x.split(",")[:number_columns])

        for i in range(number_columns):
            df.loc[:, f"{column}_{i+1}"] = df[column].apply(
                lambda x: x[i] if len(x) > i else None
        )
    df = df.drop(columns=list(columns_to_split.keys()))
    return df

def remove_zero_or_nan_rows(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Remove linhas de um DataFrame onde as colunas especificadas possuem valor 0 ou NaN.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        columns (List[str]): Lista de colunas a serem verificadas.

    Returns:
        pd.DataFrame: Novo DataFrame com as linhas filtradas.
    """
    for col in columns:
        if col in df.columns:
            df = df[(df[col] != 0) & (df[col].notna())]
        else:
            print(f"A coluna '{col}' não foi encontrada no DataFrame.")
    
    return df

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset='CPB_ROE')

    df = df.rename(columns={'PUBLICO_sum': 'Public_Total'})
    # df = df.rename(columns={'release_date': 'Release_Date'})
    df = df.rename(columns={'FIRST_EXIBICAO': 'Release_Date'})
    df = df.rename(columns={'PAIS_OBRA_first': 'Prodution_country'})
    df = df.rename(columns={'DIAS_EM_EXIBICAO': 'Days_in_exibithion'})
    df = df.rename(columns={'vote_average': 'Vote_Average'})
    # df = df.rename(columns={'vote_count': 'Vote_Count'})
    df = df.loc[df.groupby('imdb_id')['Days_in_exibithion'].idxmax()]

    df = df.drop(['production_cost', 'vote_count', 'id', 'Genre_2', 'Genre_3', 
                  'popularity', 'Metascore', 'imdb_id', 'Rated', 'CPB_ROE', 'release_date'], axis=1)
    df = remove_zero_or_nan_rows(df, ['Public_Total','Release_Date', 'Title', 'Prodution_country',
                                        'Days_in_exibithion', 'Runtime', 'Vote_Average', 'Genre_1',
                                        'Production_Companies', 'Director', 'Cast', 'IMDB_Rating'])
    print(df)
    
    columns_to_split = {
        "Cast": {"number_columns": 3},
        "Director": {"number_columns": 1},
    }
    df = _split_Columns(df, columns_to_split)
    return df

def get_variable_dictionary(variable: str) -> str:
    VAR_DICTIONARY ={
        "Runtime": "Tempo de Execução",
        "IMDB_Rating": "Avaliação do Público",
        "Vote_Average": "Avaliação dos Críticos",
        "Days_in_exibithion": "Dias em exibição",
        "Genre_1": "Gênero",
        "Prodution_country": "País de Produção",
        "Production_Companies": "Empresa Produtora",
        "Director_1": "Diretor",
        "Cast_1": "Ator Principal",
        "Public_Total": "Publico Total",
    }

    return VAR_DICTIONARY[variable]


if __name__ == "__main__":
    from extract import read_data

    df = read_data(path="data/input")
    data_frame_list = concatenate_dataframes(df)
