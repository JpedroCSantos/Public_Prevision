import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests
from api.classes.consult_class import MovieAPI
from api.schema.tmdb_schema import TmdbSchema


class TMDBMovieAPI(MovieAPI):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def search_movie_id(self, title: str) -> Optional[List[Dict[str, Any]]]:
        """
        Função buscar o id de um filme de um filme noo tmdb de acordo com o parâmetro
        query (EX: Titulo do filme)
        """
        BASE_URL: str = 'https://api.themoviedb.org/3/search/movie'
        params = {
            "query": title,
            "include_adult": "false",
            "language": "en-US",
            "page": "1"
        }
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        try:
            response = requests.get(BASE_URL, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                return data.get("results")
            return None
        except requests.exceptions.RequestException:
            return None
        
    def search_movie(self, row: Dict[str, Any], data_cache: Optional[Dict[str, Any]] = None) -> Optional[TmdbSchema]:
        """
        Função buscar os dados básicos de um filme de acordo com o parâmetro
        query (EX: Titulo do filme)
        """
        movie_id = None
        # Prioritize using the movie ID if it exists and is valid
        if "id" in row and pd.notna(row["id"]) and row["id"] != 0:
            movie_id = int(row["id"])
        else:
            # First, search by original title
            if "Title" in row and pd.notna(row["Title"]):
                search_results = self.search_movie_id(row["Title"])
                if search_results:
                    movie_id = search_results[0].get("id")

            # If not found, search by Brazilian title
            if not movie_id and "Brazilian_Title" in row and pd.notna(row["Brazilian_Title"]):
                search_results = self.search_movie_id(row["Brazilian_Title"])
                if search_results:
                    movie_id = search_results[0].get("id")

        if not movie_id:
            return None

        BASE_URL: str = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {"language": "en-US"}
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        try:
            response = requests.get(BASE_URL, headers=headers, params=params)
            if response.status_code == 200:
                return self.build_object(response.json())
            return None
        except requests.exceptions.RequestException as e:
            print(f"Erro na requisição TMDB: {e}")
            return None

    def build_object(self, api_response: Dict[str, Any]) -> Optional[TmdbSchema]:
        """
        Função para transformar o retorno da api em um objeto do tipo TmdbSchema
        """
        row_list = self.get_api_dict()
        processed_data = {}

        for api_key, config in row_list.items():
            target_keys = config["row"] if isinstance(config["row"], list) else [config["row"]]
            value = api_response.get(api_key)

            for target_key in target_keys:
                if "filter" in config and value is not None:
                    try:
                        processed_data[target_key] = eval(config["filter"], {"value": value, "rows": target_keys, "row": target_key, "datetime": datetime})
                    except Exception:
                        processed_data[target_key] = None
                else:
                    processed_data[target_key] = value
        
        try:
            return TmdbSchema(
                budget=processed_data.get("budget"),
                runtime=processed_data.get("Runtime"),
                release_date=processed_data.get("release_date"),
                vote_average=processed_data.get("vote_average"),
                vote_count=processed_data.get("vote_count"),
                id=processed_data.get("id"),
                imdb_id=processed_data.get("imdb_id"),
                Genre_1=processed_data.get("Genre_1"),
                Genre_2=processed_data.get("Genre_2"),
                Genre_3=processed_data.get("Genre_3"),
                popularity=processed_data.get("popularity"),
                Production_Companies=processed_data.get("Production_Companies"),
                Title=api_response.get("title"), # Get original title directly
                belongs_to_collection=processed_data.get("belongs_to_collection")
            )
        except Exception as e:
            print(f"Erro ao criar o objeto TmdbSchema: {e}")
            return None

    def get_api_dict(self):
        return {
            "budget": {"row": "budget"},
            "runtime": {"row": "Runtime"},
            "release_date": {
                "row": "release_date",
                "filter": 'datetime.strptime(value, "%Y-%m-%d") if value else None',
            },
            "vote_average": {"row": "vote_average"},
            "vote_count": {"row": "vote_count"},
            "id": {"row": 'id'},
            "imdb_id": {"row": "imdb_id"},
            "genres": {
                "row": ['Genre_1', 'Genre_2', 'Genre_3'],
                "filter": "value[rows.index(row)]['name'] if rows.index(row) < len(value) else None"
            },
            "popularity": {"row": 'popularity'},
            "production_companies": {
                "row": "Production_Companies",
                "filter": "value[0]['name'] if value and len(value) > 0 else None"
            },
            "title": {"row": "Title"},
            "belongs_to_collection": {
                "row": "belongs_to_collection",
                "filter": "1 if value else 0"
            }
        }

if __name__ == "__main__":
    import os
    import sys
    import pandas as pd
    from dotenv import load_dotenv
    from pipeline.transform import correct_movie_titles

    project_root = 'C:/Users/JPedr/OneDrive/Documentos/TCC/Previsao_Espectadores'
    sys.path.append(project_root)
    
    load_dotenv(dotenv_path="env/.env")
    api_key = os.getenv("TMDB_KEY")
    
    if not api_key:
        print("Chave da API do TMDB não encontrada. Verifique seu arquivo .env")
    else:
        tmdb_api = TMDBMovieAPI(api_key)
        DATA_PATH = "data/output"
        try:
            df = pd.read_parquet(f"{DATA_PATH}/DATAS/ANALISYS_DATABASE/PUBLIC_ANALISYS_DATABASE.parquet")
            df = df.rename(columns={'Title_first': 'Title', 'Brazilian_Title_first': 'Brazilian_Title'})
            df = correct_movie_titles(df, 'Title')
            df = df.head(20)
            
            print("Executando testes com as seguintes 20 linhas:")
            print(df[['Title', 'Brazilian_Title']])
            
            for index, row in df.iterrows():
                print(f"--- Buscando filme: {row['Title']} ---")
                response = tmdb_api.search_movie(row=row.to_dict())
                print(response)
        except FileNotFoundError:
            print(f"Arquivo de teste não encontrado em: {DATA_PATH}/DATAS/ANALISYS_DATABASE/PUBLIC_ANALISYS_DATABASE.parquet")
        except Exception as e:
            print(f"Ocorreu um erro durante o teste: {e}")
