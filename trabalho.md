## Análise do Modelo de Regressão Múltipla para Previsão de Bilheteria

Nesta seção, detalhamos a construção e a avaliação do modelo de regressão linear múltipla, cujo objetivo é identificar os fatores determinantes para o público total (bilheteria) de uma obra cinematográfica. A metodologia adotada segue os preceitos da análise multivariada de dados, com especial referência a Hair et al. (2009), garantindo um processo de modelagem robusto, desde a especificação inicial até a validação e o diagnóstico das premissas.

### 1. Fundamentos Metodológicos

Antes de detalhar o processo de construção do modelo, é crucial estabelecer dois conceitos metodológicos que nortearam nossas decisões.

#### 1.1 O Papel das Variáveis no Modelo: Preditoras vs. Dependente

Em qualquer modelo de regressão, é fundamental distinguir o papel de cada variável. No nosso projeto, elas se dividem em duas categorias:

**1. Variável Dependente (Y):**

*   **O que é:** É a variável que queremos prever ou explicar. É o "efeito" ou o resultado que estamos estudando.
*   **No nosso modelo:** A variável dependente é `Public_Total`. Nosso objetivo principal é construir um modelo que consiga estimar o público total de um filme com a maior acurácia possível.
*   **Transformação:** Nós aplicamos uma transformação logarítmica (`np.log1p`) nesta variável. Fizemos isso porque a distribuição original do público é extremamente assimétrica à direita (poucos filmes com bilheteria astronômica e muitos com bilheteria modesta). A transformação ajuda a linearizar a relação entre o público e nossos preditores, além de estabilizar a variância dos resíduos, melhorando o ajuste e a validade do modelo. Todas as previsões e análises de resíduos são feitas na escala logarítmica e, quando precisamos interpretar os resultados em termos de público real, revertemos a transformação (`np.expm1`).

**2. Variáveis Preditoras (ou Independentes) (X):**

*   **O que são:** São as variáveis que usamos para prever a variável dependente. São as "causas" ou os fatores que acreditamos influenciar o resultado.
*   **No nosso modelo:** São todas as outras colunas que alimentam o modelo após o pré-processamento.
O objetivo da regressão linear é encontrar a melhor combinação linear ponderada dessas variáveis preditoras para explicar a variação na variável dependente.

#### 1.2 A Premissa-Chave: Normalidade dos Resíduos, não das Variáveis

Uma das premissas fundamentais da regressão linear múltipla, especialmente para garantir a validade dos testes de hipótese (testes-t, teste-F) e dos intervalos de confiança, é que os **resíduos do modelo** (os erros de previsão) sigam uma distribuição normal.

É um equívoco comum pensar que as *variáveis preditoras* ou a *variável dependente* precisam ser normalmente distribuídas. O que realmente importa é a distribuição da diferença entre o valor real (`y`) e o valor previsto pelo modelo (`ŷ`). A teoria estatística por trás dos testes de significância assume que esses erros, em média, são zero e se distribuem como uma curva de sino (normal) em torno desse zero.

Quando analisamos o "Q-Q Plot" e o histograma dos resíduos, estamos verificando diretamente esta premissa crucial. Se os resíduos não são normais, isso não invalida os coeficientes do modelo, mas nos alerta de que os *p-valores* e os *intervalos de confiança* podem não ser totalmente precisos.

### 2. O Processo de Construção do Modelo: Uma Abordagem Iterativa

O desenvolvimento do modelo seguiu um processo iterativo, começando com um modelo abrangente e refinando-o com base em diagnósticos estatísticos rigorosos, sempre guiado pelo princípio da parcimônia.

#### 2.1. O Modelo Inicial e o Diagnóstico de Superajuste (Overfitting)

O primeiro modelo foi especificado utilizando um conjunto amplo de variáveis. As variáveis numéricas (`Runtime`, `Vote_Average`, etc.) foram padronizadas com `StandardScaler`. Para as variáveis categóricas, foi utilizado o `TargetEncoder`, que calcula a média da variável alvo para cada categoria. A análise inicial indicou um poder explicativo aparentemente excepcional (R² de ~0.95 no treino), mas a validação em dados de teste revelou uma falha crítica: o R² despencou para **0.09**. Essa discrepância é um sintoma inequívoco de **superajuste (overfitting)**. A causa raiz foi atribuída ao uso do `TargetEncoder` em variáveis de alta cardinalidade (elenco, diretor), que levou o modelo a "memorizar" os dados.

#### 2.2. Experimentos de Refinamento e Engenharia de Features

Para corrigir o overfitting e construir um modelo robusto, uma série de experimentos controlados foi executada. Em todos os experimentos, as variáveis numéricas foram padronizadas com `StandardScaler` e a variável alvo (`Public_Total`) foi transformada com `np.log1p`.

*   **Experimento 1: A Maldição da Dimensionalidade:**
    *   **Metodologia:** A primeira tentativa utilizou `OneHotEncoder` para todas as variáveis categóricas, incluindo as de alta cardinalidade como `Prodution_country` e `Production_Companies`. Este método cria uma nova coluna binária para cada categoria.
    *   **Resultado:** R² de teste de ~0.21. A criação de centenas de colunas com poucos dados destruiu a capacidade de generalização do modelo, ensinando a lição de que `OneHotEncoder` não é adequado para variáveis de alta cardinalidade.

*   **Experimento 2: A Verdadeira Baseline (Teste de Ablação):**
    *   **Metodologia:** Um modelo minimalista foi treinado apenas com as 3 variáveis mais robustas: as numéricas `Days_in_exibithion` e `Number_of_exhibition_rooms` (padronizadas com `StandardScaler`), e a binária `Belongs_to_collection`.
    *   **Resultado:** R² de teste de **0.611** e R² de treino de **0.619**. Este modelo extremamente estável e sem overfitting se tornou nosso benchmark confiável.

*   **Experimento 3: Adicionando Gênero:**
    *   **Metodologia:** A variável `Genre_1`, de baixa cardinalidade, foi adicionada ao modelo anterior e processada com `OneHotEncoder`.
    *   **Resultado:** O R² de teste subiu para **0.621** e o de treino para **0.627**, validando que a adição de gênero com a técnica correta melhora o modelo de forma estável.

*   **Experimento 4: Adicionando "Poder do Elenco" (Cast_Power):**
    *   **Metodologia:** Foi criada a feature `Cast_Power` através de engenharia de variáveis. A frequência de aparição de cada ator foi contada no conjunto de treino e, com base em quantis, os atores foram classificados em "níveis" (tiers). Essa nova variável ordinal foi então adicionada ao modelo.
    *   **Resultado:** O R² de teste alcançou **0.629** e o de treino **0.634**. A feature se mostrou altamente significativa (p-valor < 0.001) e adicionou poder preditivo sem overfitting.

*   **Experimento 5: Adicionando "Poder do Diretor" (Director_Power):**
    *   **Metodologia:** Seguindo o sucesso do `Cast_Power`, a mesma técnica de engenharia de features baseada em frequência foi aplicada à variável `Director_1`, criando a feature `Director_Power`.
    *   **Resultado:** O R² de teste subiu para **0.635** e o de treino para **0.641**. A adição se mostrou estável e a variável, significativa, melhorando incrementalmente o modelo.

*   **Experimento 6: Agrupando Países em "Tier de Mercado" (Market_Tier):**
    *   **Metodologia:** Para tratar a alta cardinalidade de `Prodution_country`, foi criada a feature `Market_Tier`. Os países foram agrupados em três níveis com base no volume de produção de filmes no dataset de treino ('Tier 1: USA', 'Tier 2: Outros Mercados Grandes', 'Tier 3: Resto do Mundo'). Essa nova variável categórica foi processada com `OneHotEncoder`.
    *   **Resultado:** O R² de teste foi para **0.638** e o de treino para **0.644**. A feature se mostrou significativa, validando que a origem da produção, quando agrupada de forma inteligente, contribui para o modelo. Este passou a ser o nosso modelo final.

*   **Experimentos Adicionais (Infrutíferos):**
    *   **Reintrodução de `Runtime` e `Quality_Score`:** Foi testada a reintrodução do `Runtime` (padronizado) e de uma `Quality_Score` (média padronizada do `IMDB_Rating` e `Vote_Average`). Essas variáveis não adicionaram poder preditivo e, em alguns casos, apresentaram coeficientes contraintuitivos.
    *   **Teste do `Production_Company_Power`:** Uma feature de "poder da produtora", criada com a mesma lógica de frequência do `Cast_Power`, se mostrou estatisticamente não significativa.
    *   **Conclusão:** Guiados pelo **Princípio da Parcimônia**, decidimos por não incluir estas variáveis no modelo final, que se consolidou como o do Experimento 6.

### 3. Análise e Validação do Modelo Final

O processo iterativo nos levou a um modelo final robusto, parcimonioso e teoricamente coerente. Antes de discutir suas implicações, uma última validação metodológica foi necessária.

#### 3.1. Validação Estatística: Correção de Heterocedasticidade com Erros-Padrão Robustos

*   **Contexto:** Após a análise dos gráficos de diagnóstico, foi identificado um problema metodológico crítico que, se não tratado, invalidaria as conclusões do modelo: a **heterocedasticidade**. O gráfico de "Resíduos vs. Valores Previstos" exibia um claro padrão de funil, indicando que a variância dos erros do modelo não era constante.
*   **Problema:** A heterocedasticidade viola uma das premissas fundamentais da Regressão Linear (OLS). Embora não afete os coeficientes, ela torna os erros-padrão, os testes-t e, consequentemente, os **p-valores** não confiáveis. Isso nos impediria de afirmar com segurança quais variáveis são estatisticamente significativas.
*   **Ação Realizada:** Seguindo a recomendação da literatura econométrica e estatística, o modelo foi reajustado utilizando **erros-padrão robustos à heterocedasticidade (erros de Huber-White)**. A implementação foi feita no `statsmodels` alterando a chamada `.fit()` para `.fit(cov_type='HC3')`.
*   **Resultado:** O sumário do modelo agora é estatisticamente válido, e as inferências (p-valores e intervalos de confiança) são confiáveis, mesmo na presença de heterocedasticidade.
*   **Conclusão da Etapa:** Esta foi a etapa final de refinamento do modelo. O modelo preditivo está agora completo e metodologicamente robusto, pronto para a interpretação e discussão final.

#### 3.2. Interpretação dos Coeficientes do Modelo Final

Com um modelo estatisticamente validado, podemos transformar os resultados numéricos em uma narrativa analítica.

*   **a) O Coração do Modelo: As Variáveis Estruturais**
    *   **`Number_of_exhibition_rooms` e `Days_in_exibithion`:** Estas são as variáveis mais fortes e a "prova de sanidade" do modelo. A interpretação é direta: o potencial de um filme fazer público está diretamente e fortemente ligado à sua distribuição (mais salas) e sua permanência em cartaz. Isso confirma a hipótese mais básica do mercado cinematográfico.
    *   **`Belongs_to_collection` (Coeficiente Negativo):** Este é um achado interessante. Pode-se argumentar que essa variável captura o efeito de sequências de menor sucesso que "pegam carona" na fama do original, ou que franquias, em média, têm um desempenho inferior a sucessos originais e inesperados quando controlamos por outros fatores.

*   **b) O Fator Humano: O Poder de Estrelas e Diretores**
    *   **`Cast_Power` (Positivo e Significativo):** A hipótese foi confirmada. Atores com maior frequência no dataset (um proxy para popularidade/experiência) estão associados a um maior público. Isso valida a estratégia dos estúdios de escalar "estrelas" para atrair espectadores.
    *   **`Director_Power` (Negativo e Significativo):** Este é talvez o achado mais rico para discussão, por ser contraintuitivo. A hipótese é que diretores de "blockbuster" fazem poucos filmes, enquanto diretores de nicho são mais prolíficos. Portanto, a alta frequência (maior `Director_Power`) pode ser um proxy para um tipo de cinema que, em média, atrai um público menor. Isso não significa que bons diretores afastam o público, mas que o *tipo* de diretor que filma com mais frequência pode estar associado a um tipo de filme de menor apelo de massa.

*   **c) A Geografia do Sucesso: O Impacto do Mercado**
    *   **`Market_Tier` (Positivo e Significativo):** A engenharia de features funcionou. O modelo mostra que a origem da produção importa. Filmes produzidos nos EUA (Tier 3) têm uma vantagem estatística sobre os de grandes mercados (Tier 2), que por sua vez superam os de mercados locais (Tier 1). Isso reflete o poder de marketing global e a influência cultural da indústria de Hollywood.

*   **d) A Nuance dos Gêneros**
    *   Com p-valores agora confiáveis, a análise pode ser mais específica. Por que `Horror` (p=0.006) e `History` (p=0.024) são significativos? O terror muitas vezes tem um público cativo e pode ser muito lucrativo com baixo orçamento. Filmes históricos podem ser eventos culturais. Por que `Comedy` e `Drama` não são significativos? Talvez a variabilidade de sucesso dentro desses gêneros seja tão imensa que, na média, eles não se distinguem da base de comparação (neste caso, "Ação") quando controlamos por outros fatores. 