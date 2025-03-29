import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import json
import re
from datetime import datetime
from typing import Dict, List, Union
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
import time

import pandas as pd
import requests
from api.classes.consult_class import MovieAPI
from api.schema.omdb_schema import OmdbSchema
from tqdm import tqdm


class OMDBMovieAPI(MovieAPI):
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.current_key_index = 0  
        self.request_count = 0

    def get_current_key(self):
        """Retorna a chave da API atual com base no contador de requisições."""
        return self.api_keys[self.current_key_index]

    def switch_key(self):
        """Troca para a próxima chave após 1000 requisições."""
        self.request_count = 0
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)        

    def search_movie(self, row: Dict, data_cache: Dict = None) -> json:
        """
        Função buscar os dados básicos de um filme de acordo com o parâmetro
        query (EX: Titulo do filme)

        args: Row: Linha do dataframe com as informações para serem utilizadas
              na busca

        return: json
        """
        imdb_id = row["imdb_id"] if row["imdb_id"] is not None and pd.notna(row["imdb_id"]) else None
        BASE_URL: str = f"http://www.omdbapi.com/?apikey={self.get_current_key()}&"
        params = {"i": imdb_id, "type": "movie"}

        try:
            response = requests.get(BASE_URL, params=params)
            data = response.json()

            if response.status_code == 200 and 'Response' in data and data['Response'] == "True":
                self.request_count += 1
                # Troca de chave após 1000 requisições
                if self.request_count >= 1000:
                    self.switch_key()
                return self.build_object(data)
            else:
                return None
        except:
            return None

    def build_object(self, api_response: json) -> OmdbSchema:
        """
        Função para transformar o retorno da api em um objeto do tipo OmdbSchema

        args: api_response(json): Json com o retorno da AOU e informações a serem
                                  transformadas em um objeto do tipo OmdbSchema

        return: OmdbSchema
        """
        row_list = self.get_api_dict()
        for key, value in api_response.items():
            if key in row_list:
                if value is not None:
                    row_value = row_list[key]
                    if "filter" in row_value:
                        filter_code = row_value["filter"]
                        api_response[key] = eval(filter_code)
                    else:
                        api_response[key] = value
                else:
                    return None

        return OmdbSchema(
            imdbRating=api_response["imdbRating"],
            Metascore=api_response["Metascore"],
            Director=api_response["Director"],
            Actors=api_response["Actors"],
            Rated=api_response["Rated"],
        )

    def get_api_dict(self):
        return {
            "Actors": {
                "row": "Cast",
                "filter": "value if value and value != 'N/A' and value != 'NaN' and value != 'nan' else None"
            },
            "Director": {
                "row": "Director",
                "filter": "value if value and value != 'N/A' and value != 'NaN' else None"
            },
            "imdbRating": {
                "row": "IMDB_Rating",
                "filter": 'float(value.replace(".", "")) if value and value != "N/A" else None',
            },
            "Metascore": {
                "row": "Metascore",
                "filter": '(float(value.replace(".", "")) / 10) if (value and value != "N/A" and float(value.replace(",", "")) > 100) else (float(value.replace(",", "")) if (value and value != "N/A") else None)',
            },
            "Rated": {
                "row": "Rated",
            },
        }
    
# class OMDBMovieAPI(MovieAPI):
#     def __init__(self, api_key: str):
#         self.api_key = api_key

#     def search_movie(self, row: Dict, data_cache: Dict = None) -> json:
#         """
#         Função buscar os dados básicos de um filme de acordo com o parâmetro
#         query (EX: Titulo do filme)

#         args: Row: Linha do dataframe com as informações para serem utilizadas
#               na busca

#         return: json
#         """
#         # title = row["Title"] if row["Title"] is not None and pd.notna(row["Title"]) else None
#         # release_date = pd.to_datetime(row["release_date"], format="%d/%m/%Y %H:%M", errors='coerce')
#         # year = release_date.year if pd.notna(release_date) else None
#         # if not title or not year: 
#         #     return None
#         # params = {"t": title, "type": "movie"}

#         # if year:
#         #     params["y"] = year
#         imdb_id = row["imdb_id"] if row["imdb_id"] is not None and pd.notna(row["imdb_id"]) else None
#         BASE_URL: str = f"http://www.omdbapi.com/?apikey={self.api_key}&"
#         params = {"i": imdb_id, "type": "movie"}

#         try:
#             response = requests.get(BASE_URL, params=params)
#             data = response.json()
#             with open("OMDB_response.json", "w") as f:
#                     json.dump(data, f)
#             if response.status_code == 200 and 'Response' in data and data['Response'] == "True":
#                 return self.build_object(data)
#             else:
#                 return None
#         except:
#             return None

#     def build_object(self, api_response: json) -> OmdbSchema:
#         """
#         Função para transformar o retorno da api em um objeto do tipo OmdbSchema

#         args: api_response(json): Json com o retorno da AOU e informações a serem
#                                   transformadas em um objeto do tipo OmdbSchema

#         return: OmdbSchema
#         """
#         row_list = self.get_api_dict()
#         for key, value in api_response.items():
#             if key in row_list:
#                 if value is not None:
#                     row_value = row_list[key]
#                     if "filter" in row_value:
#                         filter_code = row_value["filter"]
#                         api_response[key] = eval(filter_code)
#                     else:
#                         api_response[key] = value
#                 else:
#                     return None

#         return OmdbSchema(
#             imdbRating=api_response["imdbRating"],
#             Metascore=api_response["Metascore"],
#             Director=api_response["Director"],
#             Actors=api_response["Actors"],
#             Rated=api_response["Rated"],
#         )

#     def get_api_dict(self):
#         return {
#             "Actors": {
#                 "row": "Cast",
#                 "filter": "value if value and value != 'N/A' and value != 'NaN' and value != 'nan' else None"
#             },
#             "Director": {
#                 "row": "Director",
#                 "filter": "value if value and value != 'N/A' and value != 'NaN' else None"
#             },
#             "imdbRating": {
#                 "row": "IMDB_Rating",
#                 "filter": 'float(value.replace(".", "")) if value and value != "N/A" else None',
#             },
#             "Metascore": {
#                 "row": "Metascore",
#                 "filter": '(float(value.replace(".", "")) / 10) if (value and value != "N/A" and float(value.replace(",", "")) > 100) else (float(value.replace(",", "")) if (value and value != "N/A") else None)',
#             },
#             "Rated": {
#                 "row": "Rated",
#             },
#         }

if __name__ == "__main__":
    import os
    import sys
    from dotenv import load_dotenv
    from pipeline.extract import (
        read_data_csv,
        exists_database,
        read_dataframe
    )
    project_root = 'C:/Users/JPedr/OneDrive/Documentos/TCC/Previsao_Espectadores'
    sys.path.append(project_root)
    load_dotenv(dotenv_path="env/.env")

    DATA_PATH       = "data/output"
    FINAL_PATH      = f"{DATA_PATH}/DATAS/ANALISYS_DATABASE"
    FINAL_FILE_NAME = "COMPLETE_DATABASE_FOR_ANALYSIS"
    
    api_key = os.getenv("OMDB_KEY")
    omdb_api = OMDBMovieAPI(api_key)

    df = read_dataframe(FINAL_PATH, FINAL_FILE_NAME)
    df = df.head(5)
    for index, row in df.iterrows():
        # print(f"Linha {index}: {row.to_dict()}")
        response = omdb_api.search_movie(row)
        print(f"Resposta da API: {response}")