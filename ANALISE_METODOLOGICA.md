# Análise Metodológica do Modelo de Regressão

Este documento serve como um registro das principais decisões metodológicas, conceitos teóricos e aprendizados durante o desenvolvimento do modelo de previsão de público.

---

### A Premissa-Chave: Normalidade dos Resíduos, não das Variáveis

Uma das premissas fundamentais da regressão linear múltipla, especialmente para garantir a validade dos testes de hipótese (testes-t, teste-F) e dos intervalos de confiança, é que os **resíduos do modelo** (os erros de previsão) sigam uma distribuição normal.

É um equívoco comum pensar que as *variáveis preditoras* ou a *variável dependente* precisam ser normalmente distribuídas. Elas não precisam.

-   **Variáveis Preditoras (X):** Podem ter qualquer distribuição (uniforme, bimodal, assimétrica, etc.). O modelo linear não faz nenhuma suposição sobre sua distribuição. De fato, no nosso projeto, temos preditores como `Cast_Power` (ordinal) e `Genre_1` (categórica, transformada em dummies), que estão longe de ser normais.
-   **Variável Dependente (Y):** Embora a normalidade da variável dependente possa ajudar, não é uma exigência estrita. Se a variável dependente for muito assimétrica, como é o caso da nossa variável `Public_Total` (bilheteria), aplicar uma transformação (como o logaritmo) pode ajudar o modelo a estabelecer uma relação mais linear com os preditores e, como consequência, a produzir resíduos que se aproximam mais da normalidade. Foi exatamente o que fizemos.

**O Foco nos Resíduos:** O que realmente importa é a distribuição da diferença entre o valor real (`y`) e o valor previsto pelo modelo (`ŷ`). A teoria estatística por trás dos testes de significância assume que esses erros, em média, são zero e se distribuem como uma curva de sino (normal) em torno desse zero.

Quando analisamos o "Q-Q Plot" e o histograma dos resíduos, estamos verificando diretamente esta premissa crucial. Se os resíduos não são normais, isso não invalida os coeficientes do modelo (eles continuam sendo as melhores estimativas lineares não-viesadas), mas nos alerta de que os *p-valores* e os *intervalos de confiança* podem não ser totalmente precisos. A nossa análise recente que revelou uma cauda esquerda nos resíduos é um exemplo prático dessa verificação.

---

### O Papel das Variáveis no Modelo: Preditoras vs. Dependente

Em qualquer modelo de regressão, é fundamental distinguir o papel de cada variável. No nosso projeto, elas se dividem em duas categorias:

**1. Variável Dependente (Y):**

-   **O que é:** É a variável que queremos prever ou explicar. É o "efeito" ou o resultado que estamos estudando.
-   **No nosso modelo:** A variável dependente é `Public_Total`. Nosso objetivo principal é construir um modelo que consiga estimar o público total de um filme com a maior acurácia possível.
-   **Transformação:** Nós aplicamos uma transformação logarítmica (`np.log1p`) nesta variável. Fizemos isso porque a distribuição original do público é extremamente assimétrica à direita (poucos filmes com bilheteria astronômica e muitos com bilheteria modesta). A transformação ajuda a linearizar a relação entre o público e nossos preditores, além de estabilizar a variância dos resíduos, melhorando o ajuste e a validade do modelo. Todas as previsões e análises de resíduos são feitas na escala logarítmica e, quando precisamos interpretar os resultados em termos de público real, revertemos a transformação (`np.expm1`).

**2. Variáveis Preditoras (ou Independentes) (X):**

-   **O que são:** São as variáveis que usamos para prever a variável dependente. São as "causas" ou os fatores que acreditamos influenciar o resultado.
-   **No nosso modelo:** São todas as outras colunas que alimentam o modelo após o pré-processamento. Exemplos atuais incluem:
    -   `Budget_log`: O orçamento do filme (também transformado em log).
    -   `Cast_Power`: Nossa feature de engenharia que mede a "força" do elenco.
    -   `Genre_1` (codificado): O gênero principal do filme.
    -   `Production_Company_Power`: Nossa feature que mede a "força" da produtora.

O objetivo da regressão linear é encontrar a melhor combinação linear ponderada dessas variáveis preditoras para explicar a variação na variável dependente. A equação do nosso modelo, em essência, é:

`Log(Público) ≈ ß₀ + ß₁*Budget_log + ß₂*Cast_Power + ß₃*Gênero_Ação + ... + ε`

Onde os `ß` (betas) são os coeficientes que o modelo calcula, representando a magnitude e a direção do efeito de cada preditor.

---
### **Iteração 7: Validação Estatística do Modelo Final com Erros-Padrão Robustos (29/06/2025)**

*   **Contexto:** Após a análise dos gráficos de diagnóstico (Iteração 6), foi identificado um problema metodológico crítico que, se não tratado, invalidaria as conclusões do modelo: a **heterocedasticidade**. O gráfico de "Resíduos vs. Valores Previstos" exibia um claro padrão de funil, indicando que a variância dos erros do modelo não era constante.
*   **Problema:** A heterocedasticidade viola uma das premissas fundamentais da Regressão Linear (OLS). Embora não afete os coeficientes, ela torna os erros-padrão, os testes-t e, consequentemente, os **p-valores** não confiáveis. Isso nos impediria de afirmar com segurança quais variáveis são estatisticamente significativas.
*   **Ação Realizada:** Seguindo a recomendação da literatura econométrica e estatística, o modelo foi reajustado utilizando **erros-padrão robustos à heterocedasticidade (erros de Huber-White)**. A implementação foi feita no `statsmodels` alterando a chamada `.fit()` para `.fit(cov_type='HC3')`.
*   **Resultado:** O sumário do modelo agora é estatisticamente válido, e as inferências (p-valores e intervalos de confiança) são confiáveis, mesmo na presença de heterocedasticidade. Os coeficientes e o R² não se alteram, mas agora podemos interpretar a significância das variáveis com rigor acadêmico.
*   **Conclusão da Iteração:** Esta foi a etapa final de refinamento do modelo. O modelo preditivo está agora completo e metodologicamente robusto, pronto para a interpretação e discussão final no TCC.

---
### **Sugestões para Interpretação e Discussão dos Resultados**

Aqui estão alguns pontos que podem ser explorados na escrita do TCC, transformando os resultados numéricos em uma narrativa analítica.

#### a) O Coração do Modelo: As Variáveis Estruturais

*   **`Number_of_exhibition_rooms` e `Days_in_exibithion`:** Estas são as variáveis mais fortes e a "prova de sanidade" do modelo. A interpretação é direta: o potencial de um filme fazer público está diretamente e fortemente ligado à sua distribuição (mais salas) e sua permanência em cartaz. Isso confirma a hipótese mais básica do mercado cinematográfico.
*   **`Belongs_to_collection` (Coeficiente Negativo):** Este é um achado interessante. Por que pertencer a uma coleção/franquia tem um impacto *negativo* no público, quando controlamos pelos outros fatores?
    *   *Hipótese para Discussão:* Pode ser que franquias, embora populares, tenham um desempenho médio inferior a sucessos originais e inesperados? Ou talvez filmes de franquia tenham orçamentos tão altos que a relação com o público seja diferente? Pode-se argumentar que essa variável captura o efeito de sequências de menor sucesso que "pegam carona" na fama do original.

#### b) O Fator Humano: O Poder de Estrelas e Diretores

*   **`Cast_Power` (Positivo e Significativo):** A hipótese foi confirmada. Atores com maior frequência no dataset (um proxy para popularidade/experiência) estão associados a um maior público. Isso valida a estratégia dos estúdios de escalar "estrelas" para atrair espectadores.
*   **`Director_Power` (Negativo e Significativo):** Este é talvez o achado mais rico para discussão, por ser contraintuitivo.
    *   *Hipótese Principal para Discussão:* Diretores de "blockbuster" (ex: James Cameron, Christopher Nolan) fazem poucos filmes, com grandes intervalos. Diretores de filmes de nicho, "de arte" ou dramas europeus podem ser muito mais prolíficos, fazendo um filme a cada um ou dois anos. Portanto, a alta frequência pode ser um proxy para um tipo de cinema que, em média, atrai um público menor. O modelo pode ter capturado essa dinâmica. Isso não significa que bons diretores afastam o público, but que o *tipo* de diretor que filma com mais frequência pode estar associado a um tipo de filme de menor apelo de massa.

#### c) A Geografia do Sucesso: O Impacto do Mercado

*   **`Market_Tier` (Positivo e Significativo):** A engenharia de features funcionou. O modelo mostra que a origem da produção importa. Filmes produzidos nos EUA (Tier 3) têm uma vantagem estatística sobre os de grandes mercados (Tier 2), que por sua vez superam os de mercados locais (Tier 1).
    *   *Discussão:* Isso reflete o poder de marketing global, a distribuição e a influência cultural da indústria de Hollywood.

#### d) A Nuance dos Gêneros

*   Em vez de dizer "o gênero importa", a análise pode ser mais específica. O modelo, com p-valores agora confiáveis, mostra que certos gêneros se destacam.
    *   *Discussão:* Por que `Horror` (p=0.006) e `History` (p=0.024) são significativos? O terror muitas vezes tem um público cativo e pode ser muito lucrativo com baixo orçamento. Filmes históricos podem ser eventos culturais. Por que `Comedy` e `Drama` não são significativos? Talvez a variabilidade de sucesso dentro desses gêneros seja tão imensa que, na média, eles não se distinguem da base de comparação (neste caso, "Ação") quando controlamos por fatores como elenco e distribuição. 