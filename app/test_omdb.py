import os
import sys
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Adiciona o diretório raiz do projeto ao sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from api.consult_omdb import OMDBMovieAPI
from pipeline.extract import read_dataframe
from pipeline.load import load_files

def main():
    """
    Script para testar e corrigir as consultas à API do OMDB de forma isolada.
    """
    # Carrega as variáveis de ambiente
    load_dotenv(dotenv_path="env/.env")
    
    # Define os caminhos dos arquivos
    DATA_PATH = "data/output"
    ORIGINAL_DB_PATH = f"{DATA_PATH}/DATAS/ANALISYS_DATABASE"
    ORIGINAL_FILE_NAME = "PUBLIC_ANALISYS_DATABASE"
    
    INTERMEDIATE_PATH = f"{DATA_PATH}/DATAS/INTERMEDIUM_DATABASE"
    INTERMEDIATE_FILE_NAME = "INTERMEDIUM_FULL_DATABASE"

    # 1. Carrega o banco de dados original
    print("Lendo o banco de dados original...")
    original_df = read_dataframe(ORIGINAL_DB_PATH, ORIGINAL_FILE_NAME)
    if original_df is None or original_df.empty:
        print("Não foi possível carregar o banco de dados original. Encerrando.")
        return

    # 2. Filtra as linhas para a consulta
    start_index = 3600
    print(f"Filtrando as linhas para a consulta (a partir do índice {start_index})...")
    df_to_query = original_df.iloc[start_index:].copy()
    print(f"Total de {len(df_to_query)} linhas para consultar.")

    # 3. Consulta a API do OMDB
    print("Iniciando a consulta ao OMDB...")
    api_key_premium = os.getenv("OMDB_KEY_PREMIUM")
    if not api_key_premium:
        print("Chave da API OMDB Premium não encontrada no arquivo .env.")
        return
        
    omdb_api = OMDBMovieAPI([api_key_premium])
    
    results = []
    # Itera sobre as linhas com uma barra de progresso
    for index, row in tqdm(df_to_query.iterrows(), total=df_to_query.shape[0], desc="Consultando OMDB"):
        api_response = omdb_api.search_movie(row.to_dict())
        if api_response:
            # Converte o objeto OmdbSchema para um dicionário com os nomes das colunas do DataFrame
            result_dict = {
                'IMDB_Rating': api_response.imdbRating,
                'Metascore': api_response.Metascore,
                'Director': api_response.Director,
                'Cast': api_response.Actors,
                'Rated': api_response.Rated
            }
            result_dict['index'] = index 
            results.append(result_dict)

    if not results:
        print("Nenhum resultado foi obtido da consulta ao OMDB.")
        return
        
    print(f"Consulta finalizada. {len(results)} respostas válidas foram recebidas.")
    
    # Cria um DataFrame com os resultados
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index('index')

    # 4. Salva os resultados intermediários
    print(f"Salvando os resultados intermediários em: {INTERMEDIATE_PATH}")
    load_files(results_df, INTERMEDIATE_PATH, INTERMEDIATE_FILE_NAME)

    # 5. Atualiza o DataFrame original com os novos resultados
    print("Atualizando o banco de dados original com os novos resultados...")
    original_df.update(results_df)

    # 6. Salva o DataFrame original atualizado
    print(f"Salvando o banco de dados original atualizado em: {ORIGINAL_DB_PATH}")
    load_files(original_df, ORIGINAL_DB_PATH, ORIGINAL_FILE_NAME)
    
    print("Processo concluído com sucesso!")

if __name__ == "__main__":
    main()
