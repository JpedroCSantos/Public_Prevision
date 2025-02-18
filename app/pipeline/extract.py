import os  # biblioteca para manipular arquivos e pastas
import glob  # biblioteca para listar arquivos
import json
import pandas as pd
import unicodedata

from typing import List
from tqdm import tqdm

def exists_database(path: str, file_name: str) -> bool:
    """
    function para verificar se já existe um arquivo com o database. 
    Retorna um boolean caso exista.

    args: path (srt): caminho da pasta com o arquivo final
          file_name(str): Nome do arquivo da database

    return: bool
    """
    if (os.path.exists(f"{path}/{file_name}.csv")) or (
        os.path.exists(f"{path}/{file_name}.parquet") or (
        os.path.exists(f"{path}/{file_name}.db")
        )
    ):
        return True
    else:
        return False

def read_dataframe(path: str, file_name: str, encoding=False):
    print("Loading Dataframe")
    if (os.path.exists(f"{path}/{file_name}.parquet")):
        print("Lendo arquivo PARQUET")
        df = pd.read_parquet(f"{path}/{file_name}.parquet")
    
    # elif(os.path.exists(f"{path}/{file_name}.db")):
    #     return False
    
    elif(os.path.exists(f"{path}/{file_name}.csv")):
        print("Lendo arquivo CSV")
        df = pd.read_csv(f"{path}/{file_name}.csv", delimiter=";", encoding="utf-8")
    
    if encoding:
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].apply(fix_encoding_issues)

    return df


def fix_encoding_issues(text):
    if isinstance(text, str):
        replacements = {
            'Ã£': 'ã', 'Ã¡': 'á', 'Ã¢': 'â', 'Ãª': 'ê', 'Ã³': 'ó', 'Ãº': 'ú',
            'Ã©': 'é', 'Ã§': 'ç', 'Ã‘': 'Ñ', 'Ã¬': 'ì', 'Ã¹': 'ù',
            'Ã³': 'ó', 'Ã•': 'Õ', 'Ãµ': 'õ', 'Ã‰': 'É', 'Ã‡': 'Ç',
            'Ã€': 'À', 'Ã£': 'ã', 'Ãƒ': 'Â', 'Ãº': 'ú'
        }
        for corrupted, correct in replacements.items():
            text = text.replace(corrupted, correct)
    return text


def read_data_csv(path: str) -> List[pd.DataFrame]:
    """
    Função para ler arquivos .csv de uma pasta data/input
    e retornar uma lista de dataframes.

    args:
        path (str): Caminho da pasta com os arquivos.

    return:
        List[pd.DataFrame]: Lista de dataframes.
    """

    all_files = glob.glob(os.path.join(path, "*.csv"))
    data_frame_list = []

    for file in tqdm(all_files, desc="Lendo arquivos CSV"):
        data_frame_list.append(read_file_csv(path=file, delimiter=";"))

    return data_frame_list

def read_file_csv(path: str, delimiter: str = ",", encoding: str = "utf-8") -> pd.DataFrame:
    """
    function para ler um arquivo .csv"" e retornar uma dataframe

    args: path (srt): caminho arquivo
          delimiter(srt): Delimitador do arquivo csv (Default: ',')
          encoding(srt): Encode do arquivo csv (Default: 'utf-8')

    return: dataframe
    """

    try:
        df = pd.read_csv(path, delimiter=delimiter, encoding=encoding)
        return df
    except FileNotFoundError:
        print("Arquivo não encontrado:", path)
        return None
    except Exception as e:
        print("Ocorreu um erro ao ler o arquivo:", e)
        return None
    
if __name__ == "__main__":
    df, file_names = read_data_csv(path="data/input")
    print(file_names)