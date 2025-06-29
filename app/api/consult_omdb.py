import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
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

    def get_current_key(self) -> Optional[str]:
        """Retorna a chave da API atual ou None se não houver chaves."""
        if not self.api_keys:
            return None
        return self.api_keys[self.current_key_index]

    def switch_key(self):
        """Troca para a próxima chave da API."""
        if len(self.api_keys) > 1:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)        

    def search_movie(self, row: Dict[str, Any], data_cache: Optional[Dict[str, Any]] = None) -> Optional[OmdbSchema]:
        """
        Busca os dados de um filme usando seu IMDB ID.
        """
        imdb_id = row.get("imdb_id")
        api_key = self.get_current_key()

        if not imdb_id or pd.isna(imdb_id) or not api_key:
            return None

        BASE_URL = f"http://www.omdbapi.com/?apikey={api_key}&"
        params = {"i": imdb_id, "type": "movie"}

        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()  # Lança exceção para status de erro HTTP
            data = response.json()

            if data.get('Response') == "True":
                self.request_count += 1
                if self.request_count >= 999:  # Troca antes de atingir o limite
                    self.switch_key()
                return self.build_object(data)
            return None
        except requests.exceptions.RequestException as e:
            print(f"Erro de requisição ao consultar o OMDB: {e}")
            return None

    def build_object(self, api_response: Dict[str, Any]) -> Optional[OmdbSchema]:
        """
        Transforma a resposta da API em um objeto OmdbSchema.
        """
        row_list = self.get_api_dict()
        processed_data = {}

        for api_key, config in row_list.items():
            value = api_response.get(api_key)
            target_key = config["row"]
            
            if "filter" in config and value is not None:
                try:
                    processed_data[target_key] = eval(config["filter"], {"value": value})
                except (ValueError, TypeError):
                    processed_data[target_key] = None
            else:
                processed_data[target_key] = value if value not in ["N/A", "NaN"] else None
        
        try:
            return OmdbSchema(
                imdbRating=processed_data.get("IMDB_Rating"),
                Metascore=processed_data.get("Metascore"),
                Director=processed_data.get("Director"),
                Actors=processed_data.get("Cast"),
                Rated=processed_data.get("Rated"),
            )
        except Exception as e:
            print(f"Erro ao criar o objeto OmdbSchema para a resposta: {api_response}. Erro: {e}")
            return None

    def get_api_dict(self):
        return {
            "Actors": {
                "row": "Cast",
            },
            "Director": {
                "row": "Director",
            },
            "imdbRating": {
                "row": "IMDB_Rating",
                "filter": 'float(value) if value and value not in ["N/A", "NaN"] else None',
            },
            "Metascore": {
                "row": "Metascore",
                "filter": 'float(value) if value and value not in ["N/A", "NaN"] else None',
            },
            "Rated": {
                "row": "Rated",
            },
        }

if __name__ == "__main__":
    import os
    import sys
    import pandas as pd
    from dotenv import load_dotenv
    from pipeline.extract import read_dataframe

    project_root = 'C:/Users/JPedr/OneDrive/Documentos/TCC/Previsao_Espectadores'
    sys.path.append(project_root)
    load_dotenv(dotenv_path="env/.env")

    DATA_PATH = "data/output"
    DB_PATH = f"{DATA_PATH}/DATAS/ANALISYS_DATABASE"
    FILE_NAME = "PUBLIC_ANALISYS_DATABASE"
    
    api_key_premium = os.getenv("OMDB_KEY_PREMIUM")
    
    if not api_key_premium:
        print("Chave da API OMDB Premium não encontrada. Verifique o arquivo .env")
    else:
        omdb_api = OMDBMovieAPI([api_key_premium])
        try:
            df = read_dataframe(DB_PATH, FILE_NAME)
            
            # Seleciona um intervalo de linhas para teste (ex: 5 linhas a partir da linha 10)
            df_test = df.iloc[3630:].copy()
            
            print(f"Executando teste com {len(df_test)} linhas do DataFrame.")
            print(df_test[['Title', 'imdb_id']].to_string())

            for index, row in df_test.iterrows():
                print(f"--- Consultando filme: {row['Title']} (IMDB: {row['imdb_id']}) ---")
                response = omdb_api.search_movie(row.to_dict())
                print(f"Resposta da API: {response}")
        except FileNotFoundError:
            print(f"Arquivo de teste não encontrado em: {DB_PATH}/{FILE_NAME}.parquet")
        except Exception as e:
            print(f"Ocorreu um erro durante o teste: {e}")