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
    """
    print("Transformando DataFrame")

    df['DATA_EXIBICAO'] = pd.to_datetime(df['DATA_EXIBICAO'], format="%d/%m/%Y", dayfirst=True, errors='coerce')
    df['PUBLICO'] = pd.to_numeric(df['PUBLICO'], errors='coerce').fillna(0)
    df = df[df['PUBLICO'] <= 500]

    df['DIFF_DAYS'] = df.groupby('CPB_ROE')['DATA_EXIBICAO'].diff().dt.days
    df['VALID'] = (df['DIFF_DAYS'] <= 14) | df['DIFF_DAYS'].isna()
    df = df[df['VALID']].drop(columns=['DIFF_DAYS', 'VALID'])

    df_resultado = df.groupby('CPB_ROE').agg(
        DATA_EXIBICAO_min=('DATA_EXIBICAO', 'min'),
        DATA_EXIBICAO_max=('DATA_EXIBICAO', 'max'),
        PUBLICO_sum=('PUBLICO', 'sum'),
        PAIS_OBRA_first=('PAIS_OBRA', 'first'),
        Title_first=('TITULO_ORIGINAL', 'first')
    ).reset_index()

    df_resultado['DIAS_EM_EXIBICAO'] = (df_resultado['DATA_EXIBICAO_max'] - df_resultado['DATA_EXIBICAO_min']).dt.days
    df_resultado = df_resultado[df_resultado['DIAS_EM_EXIBICAO'] < 300]

    return df_resultado


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
    df = df.drop(['DATA_EXIBICAO_max', 'production_cost', 'release_date', 
                'vote_count', 'id', 'Genre_2', 'Genre_3', 'popularity',
                'Metascore', 'imdb_id', 'Rated', 'CPB_ROE'], axis=1)

    df = df.rename(columns={'PUBLICO_sum': 'Public_Total'})
    df = df.rename(columns={'DATA_EXIBICAO_min': 'Release_Date'})
    df = df.rename(columns={'PAIS_OBRA_first': 'Prodution_country'})
    df = df.rename(columns={'DIAS_EM_EXIBICAO': 'Days_in_exibithion'})
    df = df.rename(columns={'vote_average': 'Vote_Average'})
    df = df.rename(columns={'vote_count': 'Vote_Count'})

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

if __name__ == "__main__":
    from extract import read_data

    df = read_data(path="data/input")
    data_frame_list = concatenate_dataframes(df)
