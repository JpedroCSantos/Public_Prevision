# Previsão de Público no Cinema Brasileiro

## Visão Geral
Este projeto tem como objetivo desenvolver um modelo de previsão de público para filmes exibidos no Brasil, utilizando técnicas de aprendizado de máquina e análise estatística. A modelagem é baseada em múltiplos fatores, como características dos filmes, informações sobre elenco e produção, datas de lançamento e avaliações do público. O trabalho está embasado em literatura acadêmica e modelos preditivos consolidados.


## Funcionalidades Principais
- **Importação de conjuntos de dados:** Importação e junção do conjunto de dados CSV contendo informações sobre filmes.
- **Montagem da base de dados:** Consulta à API do TMDB e OMDB para obter informações adicionais sobre os filmes.
- **Pré-processamento de dados:** Remoção de outliers, normalização de variáveis numéricas e transformação de variáveis categóricas.
- **Detecção de Multicolinearidade:** Aplicação do Variance Inflation Factor (VIF) para evitar redundância entre variáveis preditoras.
- **Análise de Necessidade de Transformação Polinomial:** Avaliação da relação entre variáveis independentes e a variável alvo para possíveis ajustes de não linearidade.
- **Validação Cruzada e Diagnóstico de Resíduos:** Implementação de validação cruzada para avaliar a robustez do modelo e análise de resíduos para verificar suposições estatísticas.
- **Treinamento de Modelo:** Implementação de regressão linear e possíveis extensões, como PCA e combinação de variáveis altamente correlacionadas.

## Importação de Conjuntos de Dados

O projeto utiliza um conjunto de dados público disponibilizado pelo governo brasileiro, contendo informações sobre bilheteria diária de filmes exibidos no país. Os dados são obtidos do portal **[dados.gov.br](https://dados.gov.br/dados/conjuntos-dados/relatorio-de-bilheteria-diaria-de-obras-informadas-pelas-exibidoras)** e incluem detalhes como:

- Nome do filme
- Data de exibição
- Público total
- Receita de bilheteria
- Distribuidora responsável
- Número de salas exibidoras

Após a importação, os dados são processados e integrados ao pipeline de análise e modelagem preditiva. Esse processo inclui a junção das informações da base governamental com outras fontes, como **OMDb** e **TMDb**, enriquecendo os atributos disponíveis para previsão de público.

O processamento dos dados segue as seguintes etapas:
1. **Coleta:** Download e leitura do arquivo disponibilizado no formato **CSV**.
2. **Limpeza e Padronização:** Remoção de registros inconsistentes e padronização de campos para compatibilidade com outras bases.
3. **Junção com Outras Fontes:** Cruzamento de informações da bilheteria com metadados dos filmes obtidos via APIs externas.
4. **Armazenamento:** Persistência dos dados processados no banco **SQLite (`MOVIES.db`)** para acesso eficiente durante a modelagem.

Esse conjunto de dados desempenha um papel fundamental na construção do modelo de previsão, garantindo que o treinamento seja baseado em dados reais do mercado cinematográfico brasileiro.

## Pré-requisitos

Para executar este projeto corretamente, é necessário ter os seguintes componentes instalados:

### 1. **Python**
- **Versão recomendada**: `Python 3.12.2`
- Recomenda-se o uso do **pyenv** para gerenciar versões do Python.

### 2. **Gerenciador de Dependências**
- **Poetry**: Utilizado para gerenciar as dependências do projeto.

### 3. **Bibliotecas Principais**
As seguintes bibliotecas devem ser instaladas no ambiente do projeto:

| Biblioteca        | Versão Recomendada  |
|------------------|-------------------|
| Pandas          | `>=2.2.3,<3.0.0`   |
| Requests        | `>=2.32.3,<3.0.0`  |
| Pyarrow        | `>=19.0.0,<20.0.0` |
| Fastparquet     | `>=2024.11.0,<2025.0.0` |
| SQLAlchemy      | `>=2.0.38,<3.0.0`  |
| Scikit-Learn    | `>=1.6.1,<2.0.0`   |
| Matplotlib      | `>=3.10.0,<4.0.0`  |
| Seaborn        | `>=0.13.2,<0.14.0`  |
| Pydantic       | `>=2.10.6,<3.0.0`  |
| Python-dotenv   | `>=1.0.1,<2.0.0`  |

Essas bibliotecas principais garantirão que todas as dependências secundárias sejam instaladas automaticamente.

## Instalação/Uso

1. Clone o repositório do FilmeBox para o seu ambiente local: 
 ```bash
    git clone https://github.com/JpedroCSantos/tcc_boxOffice.git
```
2. Definir a versao do Python usando o `pyenv local 3.12.2`
2. `poetry env use 3.12.1`, `poetry install` e `poetry lock`.
3. Para iniciar o ambiente virtual utilize o comando `source .venv/Scripts/activate`
4. Criar uma [chave de API do TMDB](https://developer.themoviedb.org/reference/intro/getting-started) e uma [chave de API do OMDB](https://www.omdbapi.com/apikey.aspx).
5. Execute o comando `python app/database.py` para gerar o arquivo csv da base de dados.
6. Certifique-se de instalar as versões especificadas das bibliotecas Pandas, Requests
7. Execute os scripts `python src/main.py` através de um terminal ou ambiente de desenvolvimento que suporte Python.

## Estrutura do Projeto
```plaintext
├── app/                       # Código-fonte principal
│   ├── api/                   # Implementação da API
│   │   ├── schema/            # Schemas de requisição e resposta da API
│   │   │   ├── consult_omdb.py
│   │   │   ├── consult_tmdb.py
│   │   │   ├── consult.py
│   │   │   ├── __init__.py
│   ├── db/                    # Configuração do banco de dados
│   │   ├── models/            # Modelos de banco de dados
│   │   ├── schema/            # Estruturas de tabelas
│   │   ├── db.py              # Conexão com SQLite
│   │   ├── __init__.py
│   ├── pipeline/              # Pipeline de ETL (Extração, Transformação e Carga)
│   │   ├── extract.py         # Extração de dados
│   │   ├── transform.py       # Transformação de dados
│   │   ├── load.py            # Carga dos dados no banco
│   │   ├── __init__.py
│   ├── prediction/            # Modelos de previsão de público
│   │   ├── linear_regressor.py # Modelo de Regressão Linear
│   │   ├── random_forest_regressor.py # Modelo de Random Forest
│   │   ├── process.py         # Processamento de variáveis preditoras
│   │   ├── transform.py       # Transformação de dados para modelagem
│   │   ├── main.py            # Script principal de execução dos modelos
│   │   ├── __init__.py
├── data/                      # Diretório de dados
│   ├── input/                 # Dados brutos
│   ├── output/                # Dados processados
│   ├── __init__.py
├── README.md                   # Este arquivo
├── requirements.txt            # Lista de dependências do projeto
├── poetry.lock                 # Gerenciador de pacotes Poetry
├── pyproject.toml              # Configuração do projeto Python
```

## Contribuição
Contribuições são bem-vindas! Se você deseja contribuir para o projeto, siga estas etapas:

1. Fork o projeto.
2. Crie uma branch para sua feature (git checkout -b feature/NomeDaFeature).
3. Commit suas mudanças (git commit -am 'Adicionando uma nova feature').
4. Push para a branch (git push origin feature/NomeDaFeature).
5. Crie um novo Pull Request.

## Autores
* [João Pedro Santos](https://www.linkedin.com/in/jpedro-santos/)