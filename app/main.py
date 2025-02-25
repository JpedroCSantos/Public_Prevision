import os
import pandas as pd

from api.consult import complete_df
from api.consult_omdb import OMDBMovieAPI
from api.consult_tmdb import TMDBMovieAPI
from dotenv import load_dotenv
from pipeline.extract import (
    read_data_csv,
    exists_database,
    read_dataframe
)
from pipeline.transform import (
    concatenate_dataframes,
    transform_dataframe,
    remove_coluns,
    filter_dataframe,
    _split_Columns
)
from pipeline.load import (
    load_files
)
from prediction.linear_regressor import (
    runLinearRegressor
)

def readDataframe() -> pd.DataFrame:
    return read_dataframe(f"{DATA_PATH}/DATAS/ANALISYS_DATABASE", 'PUBLIC_ANALISYS_DATABASE')

def build_apis_objects():
    apis = []
    apis.append(TMDBMovieAPI(os.getenv("TMDB_KEY")))
    apis.append(OMDBMovieAPI(os.getenv("OMDB_KEY")))
    return apis

INPUT_DATA_PATH     = "data/input/bilheteria-diaria-obras-por-exibidoras-csv"
DATA_PATH           = "data/output"
FINAL_PATH          = f"{DATA_PATH}/DATAS/FINAL_DATABASE"
TEMP_PARQUET_FILE   = f"{DATA_PATH}/Backup/TEMP_DATAFRAME.parquet"
FINAL_FILE_NAME     = "MOVIES"

UPDATE_DATAFRAME = {
    "NEW_DATAFRAME": False,
    "COMPLETE_DATAFRAME": {
        "API_CONSULT": False,
        "FILTER_DATAFRAME": True
    },
}
EXECUTE_PREVISION = {
    "LINEAR_REGRESSOR": True,
    "RANDON_FOREST": False,
}

load_dotenv(dotenv_path="env/.env")

if exists_database(f"{DATA_PATH}/DATAS/ANALISYS_DATABASE", 'PUBLIC_ANALISYS_DATABASE'):
    df = readDataframe()

if UPDATE_DATAFRAME["NEW_DATAFRAME"]:
    print("Criando o Dataframe")
    df = read_data_csv(INPUT_DATA_PATH)
    df = concatenate_dataframes(df)
    load_files(df, f"{DATA_PATH}/DATAS/FULL_DATABASE", "FULL_DATABASE")

    print("Limpando Dataframe")
    COLUNS_TO_DROP = ['REGISTRO_GRUPO_EXIBIDOR', 'REGISTRO_EXIBIDOR', 'REGISTRO_COMPLEXO', 
                    'NOME_SALA', 'REGISTRO_SALA', 'MUNICIPIO_SALA_COMPLEXO', 
                    'RAZAO_SOCIAL_EXIBIDORA', 'CNPJ_EXIBIDORA']
    df = remove_coluns(df, COLUNS_TO_DROP)
    load_files(df, f"{DATA_PATH}/DATAS/FILTER_DATABASE", "FILTER_FULL_DATABASE")
    
    print("Limpando Dataframe para realizar as Analises")
    COLUNS_TO_DROP = ['AUDIO', 'SESSAO', 'AUDIO', 'LEGENDADA', 
                    'UF_SALA_COMPLEXO', 'TITULO_BRASIL']
    df = remove_coluns(df, COLUNS_TO_DROP)
    df = transform_dataframe(df)
    load_files(df, f"{DATA_PATH}/DATAS/ANALISYS_DATABASE", 'PUBLIC_ANALISYS_DATABASE')

if UPDATE_DATAFRAME["COMPLETE_DATAFRAME"]["API_CONSULT"]:
    if os.path.exists(TEMP_PARQUET_FILE):
        print("Carregando progresso intermediário...")
        df = pd.read_parquet(TEMP_PARQUET_FILE)
    else:
        df = readDataframe()
        df = df.rename(columns={'Title_first': 'Title'})

    try:
        df = complete_df(df, build_apis_objects())
        load_files(df, f"{DATA_PATH}/DATAS/ANALISYS_DATABASE", "CONSULT_API_DATAFRAME")
    except KeyboardInterrupt:
        print("Processo interrompido. Salvando progresso intermediário...")
        df.to_parquet(TEMP_PARQUET_FILE)
        print(f"Progresso salvo em {TEMP_PARQUET_FILE}. Rode novamente para continuar.")

if UPDATE_DATAFRAME["COMPLETE_DATAFRAME"]["FILTER_DATAFRAME"]:
    df = read_dataframe(f"{DATA_PATH}/DATAS/ANALISYS_DATABASE", "CONSULT_API_DATAFRAME")
    df = filter_dataframe(df)
    load_files(df, FINAL_PATH, FINAL_FILE_NAME)

if any(EXECUTE_PREVISION.values()):
    if EXECUTE_PREVISION["LINEAR_REGRESSOR"]:
        df = df.drop(['Cast_3', 'Cast_2', 'Title'], axis=1, inplace=False) #Removidas por alta multicolinearidade
        runLinearRegressor(df)

    if EXECUTE_PREVISION["RANDON_FOREST"]:
        runLinearRegressor(df)