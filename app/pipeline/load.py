import json
import os
from typing import List

import pandas as pd
# from db.db import Base, SessionLocal, engine
# from db.models.db_model import Movie
# from db.schema.db_schema import MovieSchema

# Base.metadata.create_all(bind=engine)

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

# def load_json(content: List[str], output_path: str, filename: str) -> str:
#     """
#     Recebe uma Lista de nome de arquivos e transforma em um arquivo json

#     args:
#         content (List[str]): Lista de nome de arquivos a ser convertido em JSON
#         output_path (str): caminho onde será salvo o arquivo
#         filename (str): nome do arquivo a ser salvo

#     return: "Arquivo salvo com sucesso"
#     """
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)

#     if os.path.exists(f"{output_path}/{filename}.json"):
#         os.remove(f"{output_path}/{filename}.json")

#     data = {}
#     for string in content:
#         data[string] = True

#     with open(f"{output_path}/{filename}.json", "w") as arquivo:
#         json.dump(data, arquivo, indent=4)

#     return "Arquivo JSON criado com sucesso"

# def transform_dataframe_to_db(data_frame: pd.DataFrame) -> Movie:
#     """
#     Recebe uma dataframe e o transforma e cria um banco de Dados com uma tabela
#     com os dados do dataframe

#     args:
#         data_frame(pd.DataFrame): Dataframe a ser transformado em tabela de um BD
        
#     """
#     for index, row in data_frame.iterrows():
#         _insert_movie_in_database(
#             MovieSchema(
#                 id_movie_tmdb = row["id"],
#                 Year=row["Year"],
#                 Title=row["Title"],
#                 WorldwideBox_office=row["WorldwideBox_office"],
#                 DomesticBox_office=row["DomesticBox_office"],
#                 InternationalBox_office=row["InternationalBox_office"],
#                 production_cost=row["production_cost"],
#                 Runtime=row["Runtime"],
#                 release_date=row["release_date"],
#                 Genre_1 = row['Genre_1'],
#                 Genre_2 = str(row['Genre_2']) if row["Cast_2"] != "nan" else None,
#                 Genre_3 = str(row['Genre_3']) if row["Cast_3"] != "nan" else None,
#                 original_language = row["original_language"],
#                 popularity = row["popularity"],
#                 Production_Companies = row["Production_Companies"],
#                 vote_average = row["vote_average"],
#                 vote_count = row["vote_count"],
#                 Cast_1=row["Cast_1"],
#                 Cast_2 = str(row["Cast_2"] if row["Cast_2"] != "nan" else None),
#                 Cast_3= str(row["Cast_3"]) if row["Cast_3"] != "nan" else None,
#                 Director_1=row["Director_1"],
#                 IMDB_Rating=row["IMDB_Rating"],
#                 Metascore=row["Metascore"],
#             )
#         )


# def _insert_movie_in_database(movie: MovieSchema) -> Movie:
#     with SessionLocal() as db:
#         existing_movie = db.query(Movie).filter(Movie.id_movie_tmdb == int(movie.id_movie_tmdb)).first()
#         if not existing_movie:
#             movie = Movie(
#                 id_movie_tmdb = movie.id_movie_tmdb,
#                 Year=movie.Year,
#                 Title=movie.Title,
#                 WorldwideBox_office=movie.WorldwideBox_office,
#                 DomesticBox_office=movie.DomesticBox_office,
#                 InternationalBox_office=movie.InternationalBox_office,
#                 production_cost=movie.production_cost,
#                 Runtime=movie.Runtime,
#                 release_date=movie.release_date,
#                 Genre_1 = movie.Genre_1,
#                 Genre_2 = movie.Genre_2,
#                 Genre_3 = movie.Genre_3,
#                 original_language = movie.original_language,
#                 popularity = movie.popularity,
#                 Production_Companies = movie.Production_Companies,
#                 vote_average = movie.vote_average,
#                 vote_count = movie.vote_count,
#                 Cast_1 = movie.Cast_1,
#                 Cast_2 = movie.Cast_2,
#                 Cast_3 = movie.Cast_3,
#                 Director_1 = movie.Director_1,
#                 IMDB_Rating=movie.IMDB_Rating,
#                 Metascore=movie.Metascore,
#             )
#             db.add(movie)
#             db.commit()
#             db.refresh(movie)
