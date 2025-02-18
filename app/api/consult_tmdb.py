import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Union
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests
from api.classes.consult_class import MovieAPI
from api.schema.tmdb_schema import TmdbSchema


class TMDBMovieAPI(MovieAPI):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def search_movie_id(self, row: Dict) -> json:
        """
        Função buscar o id de um filme de um filme noo tmdb de acordo com o parâmetro
        query (EX: Titulo do filme)

        args: Row: Linha do dataframe com as informações para serem utilizadas
              na busca

        return: json
        """
        BASE_URL: str = 'https://api.themoviedb.org/3/search/movie'
        existing_params = {
            "include_adult": "false",
            "language": "en-US",
            "page": "1"
        }
        new_params = {"query": row["Title"]}
        # if "Year" in row and pd.notna(row["Year"]):
        #     new_params["year"] = pd.to_datetime(row["Year"], errors='coerce').year 
        all_params = {**existing_params, **new_params}
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        try:
            response = requests.get(BASE_URL, headers=headers, params= all_params)
            if response.status_code == 200:
                return response.json()["results"]
            else:
                return None
        except:
            return None
        
    def search_movie(self, row: Dict, data_cache: Dict = None) -> json:
        """
        Função buscar os dados básicos de um filme de acordo com o parâmetro
        query (EX: Titulo do filme)

        args: Row: Linha do dataframe com as informações para serem utilizadas
              na busca

        return: json
        """
        movie_id = None
        if "id" in row:
            movie_id = row["id"]
        else:
            responseId = self.search_movie_id(row)
            if len(responseId) > 0 and "id" in responseId[0]:
                movie_id = responseId[0]["id"]

        
        BASE_URL: str = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {
            "language": "en-US",
        }

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            response = requests.get(BASE_URL, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                with open("TMDB_response.json", "w") as f:
                        json.dump(data, f)
                return self.build_object(data)
            else:
                return None
        except:
            return None

    def build_object(self, api_response: json) -> TmdbSchema:
        """
        Função para transformar o retorno da api em um objeto do tipo TmdbSchema

        args: api_response(json): Json com o retorno da AOU e informações a serem
                                  transformadas em um objeto do tipo TmdbSchema

        return: TmdbSchema
        """
        row_list = self.get_api_dict()
        for key, value in list(api_response.items()):
            if key in row_list:
                if value is not None:
                    row_value = row_list[key]
                    rows = row_value["row"] if isinstance(row_value["row"], list) else [row_value["row"]]
                    for row in rows:
                        if "filter" in row_value:
                            filter_code = row_value["filter"]
                            api_response[row] = eval(filter_code)
                        else:
                            api_response[row] = value
                else:
                    return None
                
        return TmdbSchema(
            budget = api_response["budget"],
            runtime = api_response["runtime"],
            release_date = api_response["release_date"],
            vote_average = api_response["vote_average"],
            vote_count = api_response["vote_count"],
            id = api_response["id"],
            imdb_id = api_response["imdb_id"],
            Genre_1 = api_response["Genre_1"],
            Genre_2 = api_response["Genre_2"],
            Genre_3 = api_response["Genre_3"],
            popularity = api_response["popularity"],
            Production_Companies = api_response["Production_Companies"],
            Title = api_response["title"],
        )
    
    def get_api_dict(self):
        return {
            "budget": {
                "row": "production_cost",
            },
            "runtime": {
                "row": "Runtime",
            },
            "release_date": {
                "row": "release_date",
                "filter": 'datetime.strptime(value, "%Y-%m-%d")',
            },
            "vote_average": {
                "row": "vote_average",
            },
            "vote_count": {
                "row": "vote_count",
            },
            "id": {
                "row": 'id'
            },
            "imdb_id":{
                "row": "imdb_id"
            },
            "genres": {
                "row": ['Genre_1', 'Genre_2', 'Genre_3'],
                "filter": "value[rows.index(row)]['name'] if rows.index(row) < len(value) else None"
            },
            "Genre_1": {
                "row": "Genre_1",
            },
            "Genre_2": {
                "row": "Genre_2",
            },
            "Genre_3": {
                "row": "Genre_3",
            },
            "popularity": {
                "row": 'popularity',
            },
            "Production_Companies": {
                "row": 'Production_Companies',
            },
            "production_companies": {
                "row": "Production_Companies",
                "filter": "value[0]['name'] if len(value) and value is not None else None"
            },
            "release_date": {
                "row": "release_date",
            },
            "title":{
                "row": "Title"
            },
            "Title":{
                "row": "Title"
            },
        }


if __name__ == "__main__":
    import os
    import sys
    from dotenv import load_dotenv

    project_root = 'C:/Users/JPedr/OneDrive/Documentos/TCC/Projeto_2'
    sys.path.append(project_root)
    
    load_dotenv(dotenv_path="env/.env")
    api_key = os.getenv("TMDB_KEY")

    tmdb_api = TMDBMovieAPI(api_key)

    response = tmdb_api.search_movie(title="LEGENDS OF OZ: DOROTHY'S RETURN", year=2013)
    movie_id = tmdb_api.search_movie_id(query = "10 Cloverfield Lane", year=2016)[0]["id"]
    print(movie_id)
    response = tmdb_api.search_movie(row={"id":movie_id})
    print (response)