## Análise do Modelo de Regressão Múltipla para Previsão de Bilheteria

Nesta seção, detalhamos a construção e a avaliação do modelo inicial de regressão linear múltipla, cujo objetivo é identificar os fatores determinantes para o público total (bilheteria) de uma obra cinematográfica. A metodologia adotada segue os preceitos da análise multivariada de dados, com especial referência a Hair et al. (2009), garantindo um processo de modelagem robusto, desde a especificação inicial até a validação e o diagnóstico das premissas.

### 1. Especificação e Análise do Modelo Inicial

O primeiro modelo foi especificado utilizando um conjunto abrangente de variáveis preditoras, incluindo características da obra (`Runtime`, `Genre_1`, `Vote_Average`, `IMDB_Rating`), informações de lançamento (`Days_in_exibithion`, `Month`, `Day_of_Week_sin`), e fatores de produção (`Prodution_country`, `Production_Companies`, `Director_1`, `Cast_1`, `Cast_2`, `Cast_3`). A variável dependente, `Public_Total`, foi transformada logaritmicamente (log(x+1)) para estabilizar a variância e normalizar sua distribuição.

A avaliação do modelo seguiu três estágios críticos recomendados por Hair et al. (2009): avaliação do ajuste geral, exame dos preditores individuais e diagnóstico das premissas.

#### 1.1. Avaliação do Ajuste Geral e Diagnóstico de Superajuste (Overfitting)

A análise inicial indicou um modelo com um poder explicativo aparentemente excepcional nos dados de treino. O **R-quadrado (R²)** foi de **0.949**, e a média do R² na validação cruzada (`Cross-Validation`) foi de **0.947**, sugerindo que o modelo explicava aproximadamente 95% da variabilidade na bilheteria do conjunto de treinamento. O **Teste F** apresentou um valor de **3304.0** com uma probabilidade de **0.00**, confirmando a significância estatística global do modelo.

Contudo, a etapa de validação, considerada por Hair et al. (2009) como o teste definitivo da viabilidade de um modelo, revelou uma falha crítica. Ao aplicar o modelo a um conjunto de dados de teste (dados não vistos durante o treinamento), o **R² despencou para 0.09**. Essa discrepância drástica entre o desempenho no treino (94.7%) e no teste (9%) é um sintoma inequívoco de **superajuste (overfitting)**. O modelo demonstrou ter "decorado" as particularidades e o ruído dos dados de treino, em vez de aprender as relações subjacentes e generalizáveis, tornando-o ineficaz para previsões no mundo real.

A causa raiz do overfitting foi atribuída à combinação do codificador `TargetEncoder` com variáveis categóricas de alta cardinalidade, como o elenco (`Cast_1`, `Cast_2`, `Cast_3`) e o diretor (`Director_1`). Essa abordagem, embora poderosa, incentivou o modelo a memorizar os resultados para combinações específicas de ator-filme presentes no treino, em vez de generalizar o conceito de "força do elenco".

#### 1.2. Avaliação dos Preditores Individuais e da Multicolinearidade

A análise dos coeficientes individuais, através dos **p-valores (P>|t|)**, permitiu identificar as variáveis com contribuição estatisticamente significativa para o modelo. As seguintes variáveis apresentaram p-valores maiores que o nível de significância de 0.05, sendo consideradas não significativas:

*   `Runtime` (p=0.512)
*   `Vote_Average` (p=0.680)
*   `IMDB_Rating` (p=0.144)
*   `Month` (p=0.643)
*   `Day_of_Week_sin` (p=0.750)

A presença dessas variáveis, conforme Hair et al. (2009), introduz ruído e complexidade desnecessária, podendo instabilizar os coeficientes das demais.

Adicionalmente, o **Fator de Inflação de Variância (VIF)** foi calculado para diagnosticar a multicolinearidade. Embora a maioria das variáveis tenha apresentado VIF baixo, as variáveis de elenco exibiram valores elevados (`Cast_2`=8.04, `Cast_3`=8.19), aproximando-se do limiar problemático de 10, o que indica uma sobreposição de informações e dificulta a interpretação isolada do impacto de cada ator.

#### 1.3. Diagnóstico das Premissas Estatísticas

A análise dos resíduos revelou que, embora a premissa de ausência de autocorrelação tenha sido satisfeita (**Durbin-Watson = 1.952**), a premissa de normalidade dos resíduos foi violada, conforme indicado pelos testes **Omnibus** e **Jarque-Bera (Prob(JB) = 0.00)**. Essa violação é, frequentemente, uma consequência de um modelo mal especificado, como o que foi identificado aqui devido ao overfitting.

### 2. Construção e Análise do Modelo Base (Iteração 2)

Com base na análise diagnóstica, ficou evidente que o modelo inicial não era robusto. Para corrigir suas deficiências, procedeu-se à construção de um **modelo base**, mais enxuto e parcimonioso.

#### 2.1. Refinamento Metodológico

O refinamento seguiu duas diretrizes principais, pautadas pelo **Princípio da Parcimônia** (Hair et al., 2009):

1.  **Simplificação do Modelo:** Foram removidas todas as variáveis que não apresentaram significância estatística no modelo inicial (`Runtime`, `Vote_Average`, `IMDB_Rating`, `Month`, `Day_of_Week_sin`).
2.  **Controle do Overfitting:** Para tratar a causa principal do superajuste, as variáveis de alta cardinalidade (`Cast_1`, `Cast_2`, `Cast_3`, `Director_1`) foram removidas.
3.  **Ajuste Conceitual:** A variável `budget` (orçamento), embora estatisticamente significativa, foi removida. A decisão foi pautada por uma questão conceitual: como o objetivo do modelo é a *previsão* de público, o orçamento de um filme é um investimento inicial e não um fator causal direto da escolha do espectador. Sua inclusão poderia levar a um modelo explicativo, mas não a um modelo preditivo puro.

Com isso, o modelo base foi especificado com apenas quatro variáveis preditoras: `Days_in_exibithion`, `Prodution_country`, `Number_of_exhibition_rooms`, e `Belongs_to_collection`.

#### 2.2. Análise dos Resultados do Modelo Base

A avaliação do modelo base demonstrou uma melhora substancial em sua validade e robustez.

*   **Validação e Generalização:** O critério de sucesso para esta iteração foi a convergência entre o desempenho do modelo nos dados de treino e teste. Os resultados foram:
    *   **Média do R² na Validação Cruzada (treino): 0.644**
    *   **R² no conjunto de Teste: 0.640**
    
    A proximidade entre os valores (64.4% e 64.0%) confirma que **o overfitting foi eliminado com sucesso**. O modelo agora possui um poder de generalização confiável, e seu R² de 64% pode ser considerado uma medida honesta e substancial de seu poder preditivo.

*   **Análise dos Preditores e Multicolinearidade:** Todas as quatro variáveis do modelo se mostraram **altamente significativas (p < 0.001)**. Além disso, a análise do VIF revelou que todos os preditores possuíam valores extremamente baixos (VIF < 1.7), indicando ausência de multicolinearidade e, portanto, coeficientes estáveis e interpretáveis. A variável com maior impacto foi `Number_of_exhibition_rooms`, destacando a importância da estratégia de distribuição para o sucesso de um filme.

*   **Diagnóstico de Premissas:** Embora a premissa de normalidade dos resíduos ainda seja violada (devido à assimetria), a **Curtose (2.931)** atingiu um valor quase ideal. Dado o grande tamanho da amostra, a violação remanescente é considerada leve, e o modelo, robusto o suficiente para ser considerado válido.

### 3. Conclusão da Iteração 2 e Próximos Passos

A segunda iteração resultou em um modelo de regressão metodologicamente sólido, generalizável e com um poder explicativo de 64%. Este modelo servirá como a fundação (baseline) para futuras melhorias. A próxima fase da pesquisa se concentrará em reintroduzir, de forma controlada e conceitualmente robusta, o impacto de fatores como o diretor e o elenco, através de técnicas de engenharia de features que evitem o risco de overfitting.

### 4. Plano de Ação Experimental e Análise das Iterações

Para incrementar o poder preditivo do modelo base, foi adotado um fluxo de trabalho experimental, testando hipóteses de forma isolada para medir o impacto individual de novas variáveis.

#### 4.1. Iteração 3: Teste da Hipótese da Popularidade do Diretor

*   **Objetivo:** Isolar e medir o impacto preditivo do renome de um diretor.
*   **Metodologia:** Foi criada uma nova variável, `Director_Popularity`, cujo valor representava a bilheteria média histórica de um diretor, calculada de forma segura (sem data leakage) a partir do conjunto de treino.
*   **Resultados:**
    *   **Média do R² na Validação Cruzada (treino): 0.862**
    *   **R² no conjunto de Teste: 0.380**
*   **Análise e Decisão:** A introdução da feature `Director_Popularity` aumentou drasticamente o R² no conjunto de treino, mas falhou em generalizar para o conjunto de teste, reintroduzindo o problema do overfitting. A discrepância entre 86.2% e 38.0% indica que a medida, por ser baseada na média de bilheteria, é muito sensível a sucessos pontuais (outliers) e leva o modelo a "memorizar" em vez de aprender. **Conclusão: a hipótese de que a popularidade do diretor, medida desta forma, melhora o modelo preditivo foi rejeitada.** O modelo base da Iteração 2 continua sendo superior em termos de robustez.

#### 4.2. Iteração 4: Teste da Hipótese da "Força do Elenco"

*   **Objetivo:** Isolar e medir o impacto preditivo do elenco com uma metodologia mais robusta, aprendendo com a falha da Iteração 3.
*   **Metodologia Adaptada:** Para evitar a sensibilidade a outliers, a "força" do elenco não foi medida pela bilheteria média. Em vez disso, foi utilizada uma abordagem baseada em **frequência**. Foi criada a variável `Cast_Power`, que categoriza os atores em "níveis" (tiers) com base no número de vezes que aparecem no conjunto de treino. Esta medida representa a "presença de mercado" de um ator, sendo menos volátil.
*   **Resultados:**
    *   **Média do R² na Validação Cruzada (treino): 0.648**
    *   **R² no conjunto de Teste: 0.630**
*   **Análise e Decisão:** **A hipótese foi validada com sucesso.** A introdução da `Cast_Power` melhorou ligeiramente o poder preditivo do modelo, mas, crucialmente, o fez sem causar overfitting. A proximidade entre os R² de treino (64.8%) e teste (63.0%) demonstra que a feature generaliza bem. A variável se mostrou estatisticamente significativa e com um coeficiente positivo, alinhado à teoria. **Conclusão: o modelo base foi aprimorado com a inclusão da `Cast_Power`, tornando-se o nosso novo modelo de referência.**

#### 4.3. Iteração 5: Teste da Hipótese da "Força do Diretor" (Revisada)

*   **Objetivo:** Verificar se a metodologia de "tiering" por frequência, bem-sucedida para o elenco, também se aplicaria aos diretores.
*   **Metodologia:** Foi criada a variável `Director_Power` usando a mesma lógica da `Cast_Power` (tiers baseados em frequência no treino).
*   **Resultados:**
    *   **Média do R² na Validação Cruzada (treino): 0.648**
    *   **R² no conjunto de Teste: 0.640**
*   **Análise e Decisão:** Este experimento gerou um aprendizado valioso. A feature `Director_Power` se mostrou estatisticamente significativa (p = 0.023) e não causou overfitting. Contudo, seu **coeficiente foi negativo (-0.0877)**, sugerindo que diretores mais frequentes estão associados a um público menor. Essa conclusão, embora estatisticamente válida, é **teoricamente contra-intuitiva**. Um modelo preditivo deve ter, além de acurácia, interpretabilidade. Manter uma variável que contradiz a lógica do fenômeno estudado enfraquece o modelo. **Conclusão: a hipótese foi rejeitada.** A frequência não é um bom proxy para o apelo popular de um diretor nos dados. A decisão metodológica correta é **descartar esta feature** e manter o modelo da Iteração 4.

### 5. Modelo Final Aprimorado e Próximos Passos

O processo iterativo e experimental nos levou a um modelo final robusto, parcimonioso e teoricamente coerente. O modelo é composto pelas variáveis do modelo base acrescidas da feature `Cast_Power`. Este modelo (R² ~0.64) representa a conclusão da nossa busca por uma representação da influência do elenco e diretores.

A próxima fase da pesquisa pode explorar outras fontes de variabilidade, como o **gênero do filme**, para verificar se é possível incrementar ainda mais o poder preditivo do nosso já sólido modelo final.

=== PLANO DE AÇÃO (ATUALIZAÇÃO PÓS-CORREÇÃO DA BASE DE DADOS) ===

**Data da Atualização:** [Inserir Data]

**Contexto:** O usuário identificou e corrigiu uma falha na base de dados final, que não continha as colunas `Genre_1` e `production_companies`. Esta correção é um passo fundamental para a robustez do modelo.

**Impacto da Correção:**
1.  **Metodologia Preservada:** Nossos aprendizados sobre os perigos do `TargetEncoder` e a importância da validação cruzada continuam válidos. A metodologia de teste de hipóteses em branches isoladas está correta.
2.  **Resultados Defasados:** Os resultados numéricos (R²) de todos os experimentos anteriores (Baseline, Hipótese do Diretor, Hipótese do Elenco) estão defasados, pois foram calculados sobre uma base de dados incompleta.
3.  **Necessidade de Recalibração:** Precisamos reestabelecer nossa linha de base (benchmark) e reavaliar as features de engenharia de variáveis (`Cast_Power`) com a base de dados correta.

**Plano de Ação Futuro:**

1.  **FASE 1: Estabelecer a Nova Linha de Base (feature/genre)**
    *   **Objetivo:** Criar um novo modelo de referência (benchmark) robusto.
    *   **Ação:** Modificar o script `linear_regressor.py` para incluir as variáveis `Genre_1` e `production_companies` no pré-processamento (provavelmente usando `OneHotEncoder`).
    *   **Variáveis:** O modelo incluirá as features numéricas já validadas + `Genre_1` + `production_companies`.
    *   **Resultado Esperado:** Um R² de treino e teste equilibrados, que servirá como nosso novo ponto de partida para comparação.

2.  **FASE 2: Reavaliar a Hipótese `Cast_Power`**
    *   **Objetivo:** Medir o ganho de performance real da feature `Cast_Power` sobre a nova e mais forte linha de base.
    *   **Ação:** Em uma nova branch, reintroduzir a lógica da `Cast_Power` ao modelo da Fase 1.
    *   **Resultado Esperado:** Comparar o R² do modelo (Fase 2) vs (Fase 1). Se houver um incremento significativo sem causar overfitting, a feature será considerada um sucesso e incorporada.

3.  **FASE 3: Nova Hipótese - `Production_Company_Power`**
    *   **Objetivo:** Testar se a popularidade/frequência da produtora pode melhorar o modelo.
    *   **Ação:** Aplicar uma lógica de engenharia de features similar à `Cast_Power` para a coluna `production_companies`.
    *   **Resultado Esperado:** Avaliar o impacto da nova feature `Production_Company_Power` no desempenho do modelo.

Vamos focar na Fase 1.

--- ANÁLISE DOS EXPERIMENTOS ---

**Benchmark Inicial (Baseado em Hipótese Incorreta)**
*   **Descrição:** Primeira versão estável do código, mas que ainda continha a variável `Director_1` sendo processada por `TargetEncoder`.
*   **Resultados:** R² de Teste ~0.64.
*   **Conclusão:** O resultado estava artificialmente inflado pelo vazamento de dados do `TargetEncoder` em uma variável de alta cardinalidade. Serviu para aprendermos a importância de usar `OneHotEncoder` para baselines.

**Experimento 1: A Maldição da Dimensionalidade**
*   **Descrição:** Inclusão de `Genre_1`, `Prodution_country` e `Production_Companies` via `OneHotEncoder`.
*   **Resultados:** R² de Teste ~0.21.
*   **Conclusão:** FALHA. A criação de centenas de colunas para países e produtoras com poucos dados destruiu a capacidade de generalização do modelo. Lição: variáveis categóricas de alta cardinalidade são perigosas e exigem tratamento especial.

**Experimento 2: A Verdadeira Baseline (Teste de Ablação)**
*   **Descrição:** Modelo apenas com as 3 variáveis numéricas mais fortes (`Days_in_exibithion`, `Number_of_exhibition_rooms`, `Belongs_to_collection`).
*   **Resultados:** R² de Teste **0.611** | R² de Treino (CV) **0.619**.
*   **Conclusão:** SUCESSO. Modelo extremamente estável, sem overfitting. Este é o nosso benchmark confiável.

**Experimento 3: Adicionando Gênero**
*   **Descrição:** Adição da variável `Genre_1` (baixa cardinalidade) ao modelo do Experimento 2.
*   **Resultados:** R² de Teste **0.621** | R² de Treino (CV) **0.627**.
*   **Conclusão:** SUCESSO. Houve um ganho de performance de 1 p.p. de forma estável. A hipótese de que o gênero adiciona poder preditivo foi validada. O modelo atual (Experimento 3) é nossa nova baseline.

**Experimento 4: Adicionando "Poder do Elenco" (Cast_Power)**
*   **Descrição:** Adição da feature `Cast_Power` (engenharia de variável baseada na frequência dos atores) ao modelo do Experimento 3.
*   **Resultados:** R² de Teste **0.629** | R² de Treino (CV) **0.634**.
*   **Conclusão:** SUCESSO. A feature `Cast_Power` se mostrou altamente significativa (p-valor = 0.000) e adicionou mais 1 p.p. de performance de forma estável. A hipótese foi validada.

**Próximo Passo:** Aplicar uma engenharia de feature similar (baseada em frequência) para a variável `Production_Companies` para criar uma feature `Production_Company_Power`.

### Experimento 5: Reavaliação de Variáveis Removidas (`Runtime` e Notas de Avaliação)

*   **Objetivo:** Após a construção de um modelo base robusto, revisitamos a hipótese inicial de que as variáveis `Runtime`, `IMDB_Rating` e `Vote_Average` poderiam, de fato, contribuir para o modelo preditivo. O objetivo era verificar se, na presença de preditores mais bem especificados, essas variáveis ainda se mostrariam relevantes.
*   **Metodologia (Parte 1 - Multicolinearidade):** As três variáveis foram reintroduzidas no modelo mais completo até então (incluindo `Genre` e `Cast_Power`). A análise do sumário estatístico imediatamente revelou:
    1.  **Significância Estatística:** Todas as três variáveis se mostraram individualmente significativas (p-valor < 0.05).
    2.  **Multicolinearidade:** O modelo apresentou um coeficiente negativo e significativo para `IMDB_Rating`, o que é teoricamente contraintuitivo. O diagnóstico do `statsmodels` (com um "Condition Number" altíssimo de `1.69e+15`) confirmou a presença de forte multicolinearidade. As variáveis `IMDB_Rating` e `Vote_Average` medem o mesmo conceito latente ("qualidade do filme") e sua presença simultânea estava inflando os erros padrão e tornando os coeficientes ininterpretáveis.

*   **Metodologia (Parte 2 - Engenharia de Feature `Quality_Score`):** Para resolver a multicolinearidade, seguimos a abordagem recomendada por Hair et al. (2009): combinar as variáveis colineares. Criamos uma nova feature, `Quality_Score`, como a média simples de `IMDB_Rating` e `Vote_Average`. As variáveis originais foram então removidas.

*   **Resultados Finais:**
    *   **Média do R² na Validação Cruzada (treino): 0.645**
    *   **R² no conjunto de Teste: 0.638**
    *   O aviso de multicolinearidade desapareceu (Cond. No. caiu para `96.7`).
    *   A variável `Quality_Score` se mostrou estatisticamente significativa (p=0.000).

*   **Análise e Decisão:** Embora o modelo tenha se tornado estatisticamente mais robusto, a análise revelou duas conclusões críticas:
    1.  **Ausência de Ganho Preditivo:** Os valores de R² (`~0.64`) foram virtualmente idênticos aos do modelo sem essas variáveis. A inclusão delas não melhorou a capacidade de previsão do modelo.
    2.  **Coeficiente Contraintuitivo:** A `Quality_Score` apresentou um coeficiente negativo. Isso ocorre porque, após controlar pelo efeito dominante do `Number_of_exhibition_rooms`, o modelo associa notas de qualidade muito altas a filmes de nicho/arte, que possuem bilheteria menor que blockbusters de nota "apenas" boa.

*   **Conclusão Final do Experimento:** Com base no **Princípio da Parcimônia**, a decisão correta é **não incluir `Runtime` e a `Quality_Score` no modelo final**. Elas adicionam complexidade e geram coeficientes de difícil interpretação sem fornecer um ganho real no poder preditivo. O modelo anterior, sem elas, é superior por ser mais simples e teoricamente mais coerente. 