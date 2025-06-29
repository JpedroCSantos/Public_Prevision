import re
from datetime import datetime
from typing import Dict, List

import pandas as pd
from api.classes.consult_class import MovieAPI
from tqdm import tqdm
import os

# def complete_df(df: pd.DataFrame, apis: List[MovieAPI], save_interval: int = 100) -> pd.DataFrame:
#     """
#     Função para completar as informações vazias do dataframe,
#     através de consultas a várias APIs, salvando progresso em formato Parquet.

#     Args:
#         df (pd.DataFrame): DataFrame a ser completado.
#         apis (List[MovieAPI]): Lista de instâncias de classes que implementam a interface MovieAPI para fazer as consultas à API.
#         save_interval (int): Número de linhas após o qual o progresso será salvo.

#     Returns:
#         pd.DataFrame: DataFrame completo.
#     """
#     all_rows = [api.get_api_dict() for api in apis]
#     ROWS = {}
#     for row in all_rows:
#         ROWS.update(row)

#     data_cache = {index: {} for index in df.index}
#     rows_to_update = []

#     for index, row in tqdm(df.iterrows(), total=len(df), desc="Consultando API"):
#         columns_to_update = [
#             value["row"]
#             for key, value in ROWS.items()
#             if pd.isna(pd.Series(row.get(value["row"]))).all()
#         ]
#         if columns_to_update:
#             rows_to_update.append(index)

#         for api in apis:
#             if any(columns_to_update) and not data_cache[index].get(api):
#                 data_cache[index][api] = api.search_movie(row, data_cache[index])

#     for index in tqdm(rows_to_update, total=len(rows_to_update), desc="Atualizando Dataframe"):
#         row = df.loc[index]
#         for api in apis:
#             if data_cache[index].get(api) is not None:
#                 df = insert_info_movie(data_cache[index][api], ROWS, df, index, row)

#     # Salvar o progresso final
#     save_progress(df, "data/output", "complete_database")
#     return df

def complete_df(df: pd.DataFrame, apis: List[MovieAPI]) -> pd.DataFrame:
    """
    Completa o DataFrame com informações das APIs.
    """
    updated_rows = []
    
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Consultando APIs"):
        current_row_data = row.to_dict()
        
        for api in apis:
            api_response = api.search_movie(current_row_data)
            
            if api_response:
                # Converte o schema da API (dataclass) para um dicionário
                api_data = vars(api_response)
                # Remove valores nulos para não sobrescrever dados existentes com nada
                api_data = {k: v for k, v in api_data.items() if pd.notna(v)}
                # Atualiza o dicionário da linha atual com os dados da API
                current_row_data.update(api_data)

        updated_rows.append(current_row_data)

    # Cria um novo DataFrame a partir das linhas atualizadas
    return pd.DataFrame(updated_rows)

def save_progress(df: pd.DataFrame, output_path: str, filename: str):
    """
    Função para salvar o progresso atual do DataFrame no formato Parquet.

    Args:
        df (pd.DataFrame): DataFrame a ser salvo.
        output_path (str): Caminho da pasta onde o arquivo será salvo.
        filename (str): Nome do arquivo (sem extensão).
    """
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, f"{filename}.parquet")
    df.to_parquet(file_path, index=False)
    print(f"Progresso salvo em: {file_path}")

def insert_info_movie(
    response: Dict, row_list: Dict, df: pd.DataFrame, index: int, row: List
) -> pd.DataFrame:
    """
    Função para inserir os dados retornados da api no dataframe

    args: response(Dict)*: Retorno da API
          row_list (Dict[str])*: Dicionario com os parâmetros de rotorno a serem utilizados
          df (pd.DataFrame)*: Dataframe onde serão inserido os dados retornados
          index (int)*: Index da linha iterada
          row (List)*: Conteudo da linha iterada


    return: df(pd.DataFrame): Dataframe com a linha atualizada
    """
    for key, value in dict(response).items():
        if key in row_list and value is not None:
            row_value = row_list[key]
            if row_value["row"] not in df.columns:
                df.loc[:, row_value["row"]] = None
            df.loc[index, row_value["row"]] = value
    return df


if __name__ == "__main__":
    import os
    import sys

    from consult_omdb import OMDBMovieAPI
    from consult_tmdb import TMDBMovieAPI
    from dotenv import load_dotenv

    # Adicione o diretório raiz do projeto ao PYTHONPATH
    sys.path.append(
        os.path.dirname(
            os.path.dirname(
                "C:/Users/JPedr/OneDrive/Documentos/TCC/Projeto/app/pipeline/transform.py"
            )
        )
    )
    sys.path.append(
        os.path.dirname(
            os.path.dirname(
                "C:/Users/JPedr/OneDrive/Documentos/TCC/Projeto/app/pipeline/extract.py"
            )
        )
    )

    load_dotenv(dotenv_path="env/.env")

    FINAL_PATH = "data/output"

    tmdb_api = TMDBMovieAPI(os.getenv("TMDB_KEY"))
    omdb_api = OMDBMovieAPI(os.getenv("OMDB_KEY"))
    df = pd.read_parquet(f"{FINAL_PATH}/public_analisys_database.parquet")
    print(df)

    df = complete_df(df=df, apis=[tmdb_api, omdb_api])
    print(df)