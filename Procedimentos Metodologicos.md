# 1. Procedimentos Metodológicos

## 1.1 Análise Exploratória de Dados (EDA)

Inicialmente, foi realizada uma análise exploratória de dados (EDA) para compreender a distribuição das variáveis e identificar possíveis outliers. O histograma da variável Total de Público revelou a presença de outliers significativos, que foram tratados utilizando o intervalo interquartil (IQR).

**Figura 2 - Distribuição dos atributos não categóricos**
![Figura 2 - Distribuição dos atributos não categóricos](https://storage.googleapis.com/generativeai-downloads/images/d378297b5e40632a0c7c0be509176313)
Fonte: Próprio Autor, 2025.

**Figura 3 - Histograma de Público Total (Antes da Remoção de Outliers)**
![Figura 3 - Histograma de Público Total (Antes da Remoção de Outliers)](https://storage.googleapis.com/generativeai-downloads/images/5f483c6b2fd53c481dd9d7c0401d4a0a)
Fonte: Próprio Autor, 2025.

**Figura 4 - Histograma de Público Total (Depois da Remoção de Outliers)**
![Figura 4 - Histograma de Público Total (Depois da Remoção de Outliers)](https://storage.googleapis.com/generativeai-downloads/images/42e8d35ca24843b01850785f2cc81c5d)
Fonte: Próprio Autor, 2025.

## 1.1.2 Transformação de Dados

As variáveis numéricas passaram por normalização através da utilização do método MinMaxScaler, que aplica uma transformação linear que escala cada característica para um intervalo específico, geralmente entre 0 e 1, utilizando a seguinte equação 1.

$X_{scaled}=\frac{X-X_{min}}{X_{max}-X_{min}}$ (1)

garantindo que todas as variáveis ficassem no intervalo de 0 a 1, conforme recomenda (HAIR et al., 2009). As transformações de dados realizadas foram:

* Seno e Cosseno para Dia da Semana (captura de padrões cíclicos).
* Remoção de variáveis altamente correlacionadas para evitar multicolinearidade.

## 1.1.3 Codificação de Variáveis Categóricas

Segundo LI e LIU (2022) a transformação de variáveis categóricas em variáveis numéricas é uma etapa essencial para melhorar a precisão de modelos preditivos em aprendizado de máquina, especialmente no contexto de dados relacionados à indústria cinematográfica.

Nesse sentido, o método Target Encoding é uma solução eficiente para lidar com variáveis categóricas associadas ao elenco e à produção de filmes. Essa técnica aplica a equação 2 que transforma as categorias em valores numéricos contínuos, substituindo cada categoria pela média da variável-alvo correspondente. Essa abordagem é particularmente útil em modelos lineares, pois reduz a dimensionalidade do conjunto de dados sem perda significativa de informações, minimizando também o risco de super ajuste associado a outros métodos de codificação.

$Categoria\_Codificada = \frac{\sum{y}}{\text{número de ocorrências da categoria}}$ (2)

## 1.2 Detecção de Multicolinearidade

A multicolinearidade ocorre quando duas ou mais variáveis independentes em um modelo de regressão estão altamente correlacionadas, o que pode comprometer a precisão das estimativas dos coeficientes e dificultar a interpretação dos resultados (HAIR et al., 2009). Para identificar a presença de multicolinearidade, foi utilizada a estatística Variance Inflation Factor (VIF), conforme demonstrado na Equação 3:

$VIF(X_{i})=\frac{1}{1-R_{i}^{2}}$ (3)

Onde $R_{i}^{2}$ representa o coeficiente de determinação da regressão da variável $X_{i}$ sobre as demais variáveis explicativas. Valores de VIF muito acima de 10 indicam uma possível multicolinearidade problemática, enquanto valores próximos de 1 sugerem baixa correlação com outras variáveis (HAIR et al., 2009).

A tabela 1 apresenta os valores iniciais de VIF para cada variável:

**Tabela 1 - Variance Inflation Factor por indicador**

| Indicador | VIF |
| :--- | :--- |
| País de produção | 7,50 |
| Dias em exibição | 9,13 |
| Tempo de execução | 39,13 |
| Votação média | 37,24 |
| Gênero | 11,01 |
| Empresas Produtoras | 76,76 |
| Classificação IMDB | 13,79 |
| Ator 1 | 124,31 |
| Ator 2 | 211,35 |
| Ator 3 | 176,47 |
| Diretor | 177,45 |
| Mês de lançamento | 48,49 |
| Dia da Semana sin | infinito |
| Dia da Semana cos | infinito |

Fonte: Próprio Autor, 2025.

As variáveis Ator 1, Ator 2, Ator 3, Diretor e Empresas Produtoras apresentaram VIFs muito elevados, indicando uma elevada correlação uma vez que todas estão relacionadas ao elenco. Da mesma forma as variáveis Dia da Semana sin e Dia da Semana cos também apresentaram valores elevados intuindo-se que esta correlação ocorre pois se referem à mesma variável de medida. Visando melhorar a acurácia, foram mantidas as variáveis Ator 1 e Dia da Semana sin, As demais variáveis correlacionadas foram removidas buscando ampliar a estabilidade das estimativas e reduzir os efeitos negativos da multicolinearidade.

Após remoção os seguintes resultados foram obtidos:

**Tabela 2 - Variance Inflation Factor por indicador pós remoção**

| Indicador | VIF |
| :--- | :--- |
| Pais de produção | 7,50 |
| Dias em exibição | 8,99 |
| Tempo de execução | 39,05 |
| Avaliação de Críticos | 37,24 |
| Gênero | 10,97 |
| Classificação IMDB | 13,73 |
| Ator 1 | 29,22 |
| Mês de lançamento | 48,18 |
| Dia da Semana sin | 23,22 |

Fonte: Próprio Autor, 2025.

Pode-se notar uma melhora nos indicadores VIF, com nenhuma variável ultrapassando agora o valor de 50. Embora algumas variáveis como mês de lançamento, avaliação de críticos e tempo de execução apresentam os maiores valores de VIF, elas foram mantidas no modelo devido à sua relevância teórica e impacto significativo na previsão do público total.

## 4.2. Verificação das Premissas da Regressão Linear

Para garantir a validade dos resultados, foram verificadas as seguintes premissas estatísticas:

* Normalidade das variáveis preditoras
* Normalidade dos resíduos
* Homocedasticidade dos resíduos

### 4.2.1. Normalidade das Variáveis

A normalidade das variáveis foi avaliada utilizando o teste de Shapiro-Wilk, um dos mais robustos para pequenas amostras (HAIR et al., 2009 apud SHAPIRO & WILK, 1965). Complementarmente, gráficos Q-Q Plots e Histogramas com Kernel Density Estimation (KDE) foram utilizados para análise visual.

![Gráficos de Normalidade das Variáveis](https://storage.googleapis.com/generativeai-downloads/images/5d311956795022881b238d38865672ab)

Os resultados iniciais indicaram que algumas variáveis apresentaram assimetria positiva e negativa, o que poderia comprometer a inferência estatística da regressão. Para corrigir esse problema, foram aplicadas transformações conforme a recomendação de BOX & COX (HAIR et al., 2009 apud BOX & COX 1964):

* **Tempo de Execução:** Aplicada a Transformação Box-Cox, pois a variável era estritamente positiva e apresentava leve assimetria.
* **Avaliação do Público:** Aplicada a Transformação Log-Invertida (log(1+ (máximo - x))) para corrigir a assimetria negativa.
* **Dias em Exibição:** Aplicada a Transformação Logarítmica (log1p), pois a variável apresentava forte assimetria positiva.
* **Avaliação dos Críticos:** Aplicada a Transformação Box-Cox, pois a variável apresentava leve desvio da normalidade.

Após a aplicação das transformações, novas análises demonstraram que as distribuições estavam próximas da normalidade.

![Gráficos de Normalidade das Variáveis Pós-transformação](https://storage.googleapis.com/generativeai-downloads/images/507e11f185ef35b2e557291a27e7d9b9)
![Gráficos de Normalidade das Variáveis Pós-transformação](https://storage.googleapis.com/generativeai-downloads/images/0b171852c009d17ed6877c8e88e9389f)

### 4.2.2. Verificação da Homocedasticidade

A homocedasticidade dos resíduos foi analisada utilizando o teste de Breusch-Pagan (HAIR et al., 2009 apud BREUSCH & PAGAN, 1979) e gráficos Resíduos vs Valores Preditos. Inicialmente, os resíduos apresentaram heterocedasticidade, caracterizada por um padrão de funil. Para corrigir esse problema, a variável dependente "Público Total" foi transformada usando log1p, conforme recomendado por WOOLDRIDGE (2013). Após essa transformação, a heterocedasticidade foi significativamente reduzida.

![Gráficos de Homocedasticidade](https://storage.googleapis.com/generativeai-downloads/images/157d6b83f0c37f14b60e6bb69c3caefc)

# 1.3 Modelagem

A modelagem foi realizada utilizando a Regressão Linear Múltipla. O modelo foi treinado e avaliado utilizando o conjunto de dados dividido em 80% para treino e 20% para teste.

# 2. Resultados e Discussões

Os seguintes indicadores foram utilizados para avaliar o desempenho do modelo:

* **Erro Quadrático Médio (Mean Squared Error) (MSE):** 0.01
* **Coeficiente de Determinação ($R^{2}$):** 0.79
* **Média do $R^{2}$ na Validação Cruzada:** 0.751

**Equação da Regressão Linear:**

`Public_Total = 8.5133 + (0.0000 * Prodution_country) + (0.1128 * Days_in_exibithion) + (-0.0018 * Runtime) + (-0.0521 * Vote_Average) + (0.000 * Genre_1) + (-0.0389 * IMDB_Rating) + (0.0001 * Cast_1) + (0.0000 * Month) + (0.0000 * Day_of_Week_sin)`

## 2.1.1 Validação Cruzada

A validação cruzada é uma técnica utilizada para avaliar o desempenho de modelos preditivos, garantindo que os mesmos generalizem bem para novos dados. Em vez de dividir os dados em um único conjunto de treino e teste, a validação cruzada particiona os dados em múltiplos subconjuntos (chamados de partições ou folds) e treina o modelo diversas vezes, alternando os conjuntos de treino e teste. No estudo realizado, utilizou-se a validação cruzada com cinco partições (5-fold cross-validation), resultando em um $R^{2}$ médio de 0,751, o que indica um bom poder preditivo do modelo.

## 2.1.2 Diagnóstico de Resíduos

O diagnóstico de resíduos é um conjunto de análises aplicadas após a modelagem para verificar se as suposições do modelo de regressão linear foram atendidas. Em termos simples, os resíduos são as diferenças entre os valores reais e os valores preditos pelo modelo. Se o modelo for adequado, os resíduos devem seguir uma distribuição aleatória, sem padrões sistemáticos.

No estudo, para verificar a adequação do modelo, foi realizado um diagnóstico de resíduos. O gráfico de dispersão entre os resíduos e os valores previstos está presente na figura 5. Os gráficos Resíduos vs Valores Reais e Distribuição dos Resíduos indicaram que a transformação logarítmica ajudou a aproximar os resíduos de uma distribuição normal. No entanto, um leve padrão ainda era perceptível, sugerindo a presença de fatores não capturados pelo modelo.

**Figura 5 - Diagnóstico de resíduos**
![Figura 5 - Diagnóstico de resíduos](https://storage.googleapis.com/generativeai-downloads/images/5f79599d14619d77f248d21a2c3327d5)
Fonte: Próprio Autor, 2025.

# 3. Conclusão

O presente estudo teve como objetivo desenvolver um modelo de previsão de público para filmes, utilizando técnicas de aprendizado de máquina e análise estatística. Para isso, foi realizada uma extensa etapa de processamento e seleção de atributos relevantes, a fim de construir um modelo preditivo capaz de capturar os padrões que influenciam o número de expectadores de um filme.

A validação cruzada demonstrou que o modelo de regressão linear foi capaz de capturar de maneira satisfatória os padrões dos dados, obtendo um coeficiente de determinação ($R^{2}$) médio de 0,87. A análise de diagnóstico de resíduos, por sua vez, indicou que, apesar do bom desempenho geral do modelo, ainda há a presença de pequenas discrepâncias na distribuição dos erros, sugerindo a possibilidade de melhorias por meio do uso de modelos mais robustos ou transformações adicionais nas variáveis.

Os resultados obtidos reforçam a importância de uma abordagem criteriosa na seleção e tratamento das variáveis para previsão de bilheteria. Conforme discutido em estudos anteriores, como (HE; HU, 2021) e (HENNIG-THURAU et al., 2006), fatores como a presença de grandes estúdios, o envolvimento de diretores renomados e o gênero cinematográfico são determinantes na atratividade de um filme. Neste sentido, o modelo desenvolvido corrobora com a literatura ao demonstrar que tais atributos são fundamentais na previsão de audiência.

Como sugestão para trabalhos futuros, sugere-se a incorporação de modelos não lineares, como Random Forest Regressor e Redes Neurais, que podem capturar relações mais complexas entre as variáveis. Além disso, a inclusão de novas fontes de dados, como tendências de redes sociais e volume de buscas na internet, pode melhorar ainda mais a acurácia do modelo. Finalmente, uma abordagem de aprendizado de transferência, que permita utilizar dados de mercados internacionais para prever o desempenho de filmes no Brasil, também pode ser explorada para expandir a aplicabilidade deste estudo.

Assim, este trabalho contribui para a compreensão dos fatores que influenciam o sucesso comercial dos filmes e serve como base para pesquisas futuras na área de previsão de bilheteria, auxiliando tanto pesquisadores quanto profissionais da indústria cinematográfica na tomada de decisões estratégicas.