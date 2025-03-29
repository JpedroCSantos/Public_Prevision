import json
import os
from typing import List

import pandas as pd

def load_files(data_frame: pd.DataFrame, output_path: str, filename: str) -> str:
    """
    Recebe um dataframe e transforma em um arquivos csv e parquet

    args:
        data_frame (pd.dataframe): dataframe a ser convertido em excel
        output_path (str): caminho onde será salvo o arquivo
        filename (str): nome do arquivo a ser salvo

    return: "Arquivos salvos com sucesso"
    """
    print("Salvando Arquivos")
    load_parquet(data_frame, output_path, filename)
    load_csv(data_frame, output_path, filename)
    
    return "Arquivos criados com sucesso"

def load_csv(data_frame: pd.DataFrame, output_path: str, filename: str, delimiter: str = ";") -> str:
    """
    Recebe um dataframe e transforma em um arquivo csv

    args:
        data_frame (pd.dataframe): dataframe a ser convertido em excel
        output_path (str): caminho onde será salvo o arquivo
        filename (str): nome do arquivo a ser salvo

    return: "Arquivo salvo com sucesso"
    """
    print("Salvando CSV")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if os.path.exists(f"{output_path}/{filename}.csv"):
        os.remove(f"{output_path}/{filename}.csv")

    data_frame.to_csv(f"{output_path}/{filename}.csv", index=False, sep = delimiter)
    return "Arquivo CSV criado com sucesso"

def load_parquet(data_frame: pd.DataFrame, output_path: str, filename: str) -> str:
    """
    Recebe um dataframe e transforma em um arquivo parquet

    args:
        data_frame (pd.dataframe): dataframe a ser convertido em parquet
        output_path (str): caminho onde será salvo o arquivo
        filename (str): nome do arquivo a ser salvo

    return: "Arquivo salvo com sucesso"
    """
    print("Salvando Parquet")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if os.path.exists(f"{output_path}/{filename}.parquet"):
        os.remove(f"{output_path}/{filename}.parquet")

    data_frame.to_parquet(f"{output_path}/{filename}.parquet", index=False)
    return "Arquivo PARQUET criado com sucesso"