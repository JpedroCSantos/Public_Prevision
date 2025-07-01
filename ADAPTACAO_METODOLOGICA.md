# Guia para Adaptação da Metodologia do TCC

Este documento serve como um guia para atualizar o texto da dissertação, comparando a metodologia descrita na versão antiga (`Procedimentos Metodologicos.md`) com o modelo final e robusto que desenvolvemos.

---

### **1. Aprofundamento da Revisão de Literatura**

**Status: APROFUNDAR**

*   **Foco Antigo:** Utilização de referências seminais sobre previsão de bilheteria.
*   **Novo Foco:**
    1.  **Expandir a Base Teórica:** Ir além dos artigos já selecionados (ex: Hennig-Thurau et al.), explorando com maior profundidade os estudos sobre os fatores que influenciam o comportamento do consumidor no cinema.
    2.  **Contextualizar para o Mercado Local:** Buscar ativamente por referências e dados que abordem as especificidades do **mercado cinematográfico brasileiro**.
    3.  **Incorporar a Perspectiva do Exibidor:** Incluir na revisão estudos ou artigos que analisem o ponto de vista das **empresas exibidoras**, fortalecendo a relevância prática do modelo.
    4.  **Objetivo:** Criar uma revisão de literatura que não apenas fundamente o modelo, mas que também o posicione dentro do contexto específico do mercado nacional, aumentando a originalidade e a contribuição do trabalho.

---

### **2. Justificativa da Escolha Metodológica**

**Status: ROBUSTECER**

*   **Foco Antigo:** Justificativa simples da regressão múltipla.
*   **Novo Foco:**
    1.  **Detalhar o Processo de Decisão (Hair et al.):** Não apenas citar o fluxograma de Hair, mas descrever o raciocínio. Explicar que, dado que a variável dependente é métrica (`Public_Total`) e o objetivo primário é **entender a relação entre múltiplas variáveis independentes e a variável dependente**, a regressão múltipla é a técnica indicada pelo framework.
    2.  **Análise Comparativa (Por que não outras técnicas?):** Justificar a escolha da regressão em detrimento de outros algoritmos mencionados na literatura (e.g., Random Forest, Redes Neurais).
        *   **Foco em Interpretabilidade vs. Previsão Pura:** Argumentar que, embora modelos como `Random Forest` possam ter um poder preditivo superior, o objetivo central deste TCC é a **inferência**, ou seja, entender e interpretar o impacto individual de cada variável (ex: "Qual o efeito do poder do elenco na bilheteria?"). A natureza de "caixa-preta" (black-box) desses modelos os torna menos adequados para este objetivo analítico.
        *   **Complexidade e Parcimônia:** Posicionar a regressão linear como uma escolha alinhada ao **princípio da parcimônia**. Ela oferece um modelo robusto, interpretável e adequado ao tamanho do dataset, sem a complexidade e a necessidade de grandes volumes de dados que técnicas como Redes Neurais exigiriam.

---

### **3. Seção de Transformação de Dados**

**Status: REESCREVER**

*   **Foco Antigo:** Normalização de preditores com `MinMaxScaler`.
*   **Novo Foco:**
    1.  **Justificar a transformação `log1p` na variável dependente (`Public_Total`):** Explicar que o objetivo foi tratar a forte assimetria da variável, estabilizar a variância e linearizar a relação com os preditores, sendo o primeiro passo para mitigar a heterocedasticidade.
    2.  **Corrigir a padronização dos preditores:** Mencionar o uso de `StandardScaler` (e não `MinMaxScaler`) e explicar seu propósito (colocar variáveis em escala comum para que o modelo não atribua importância indevida baseada apenas na magnitude).
    3.  **Remover menções a transformações cíclicas (sin/cos) e outras transformações nos preditores (Box-Cox, etc.).**

---

### **4. Seção de Codificação de Variáveis Categóricas**

**Status: REESCREVER COMPLETAMENTE**

*   **Foco Antigo:** `Target Encoding`.
*   **Novo Foco (muito mais robusto):**
    1.  **Descartar totalmente a explicação sobre `Target Encoding`**.
    2.  **Criar uma seção de "Engenharia de Features"**: Detalhar a criação das variáveis ordinais `Cast_Power`, `Director_Power`, e `Market_Tier`. Justificar como uma abordagem superior para capturar o impacto não-linear da popularidade e relevância de mercado, ao mesmo tempo que resolve problemas de cardinalidade.
    3.  **Atualizar a codificação:** Explicar que para a variável `Genre_1`, de baixa cardinalidade, foi utilizado o `One-Hot Encoding` para criar variáveis dummy, evitando a criação de uma relação ordinal artificial.

---

### **5. Seção de Multicolinearidade (VIF)**

**Status: ATUALIZAR NARRATIVA (Abordagem Integrada)**

*   **Foco Antigo:** Usar VIF para remover variáveis.
*   **Novo Foco (Integrado na Jornada do Modelo):** Em vez de uma seção isolada sobre VIF, a discussão deve ser tecida ao longo da metodologia, mostrando como o diagnóstico de multicolinearidade informou as decisões em cada etapa.
    1.  **No capítulo do "Modelo Inicial":** Apresentar a primeira tabela de VIF, destacando os valores elevados para as variáveis de elenco (`Cast_2`, `Cast_3`). Usar isso, junto com o overfitting e os p-valores altos, como **uma das principais justificativas para a simplificação do modelo** e a necessidade de uma abordagem mais inteligente.
    2.  **No capítulo de "Engenharia de Features":** Ao introduzir a criação da `Cast_Power`, explicar que essa técnica não só visava capturar a "força do elenco" de forma mais robusta, mas também serviu como uma **solução elegante para o problema de multicolinearidade** identificado anteriormente, ao consolidar múltiplas variáveis correlacionadas em uma única.
    3.  **No capítulo do "Modelo Final":** Apresentar a tabela de VIF final do modelo validado (que terá todos os valores baixos). Usar esta tabela como **prova final de que a premissa de ausência de multicolinearidade foi satisfeita**, reforçando a robustez e a confiabilidade dos coeficientes do modelo.

Essa abordagem conta uma história completa sobre o problema: como ele foi identificado, como foi resolvido e como a solução foi validada.

---

### **6. Seção sobre Premissas da Regressão**

**Status: CORRIGIR E REESCREVER**

*   **Normalidade das Variáveis:**
    *   **Foco Antigo:** Tentativa de normalizar as variáveis preditoras.
    *   **Novo Foco:** Corrigir o conceito. Afirmar que a premissa de normalidade **se aplica aos resíduos, não aos preditores**. Remover toda a discussão e os gráficos sobre a normalização dos preditores.

*   **Homocedasticidade:**
    *   **Foco Antigo:** Alegação de que `log1p` resolveu o problema.
    *   **Novo Foco:** Apresentar a narrativa completa e mais sofisticada:
        1.  A transformação `log1p` foi aplicada como um **primeiro passo** para mitigar a heterocedasticidade.
        2.  No entanto, a análise dos gráficos de diagnóstico do modelo final revelou que a heterocedasticidade **ainda estava presente**.
        3.  Para garantir a validade das inferências estatísticas, a solução definitiva foi ajustar o modelo utilizando **erros-padrão robustos à heterocedasticidade (Huber-White, `HC3`)**. Isso demonstra um tratamento metodológico completo e correto do problema.

---

### **7. Seções de Resultados e Conclusão**

**Status: SUBSTITUIR COMPLETAMENTE**

*   **Foco Antigo:** Resultados de um modelo desatualizado.
*   **Novo Foco:**
    1.  **Resultados:** Apresentar a **tabela de resultados final do `statsmodels`**, destacando que ela utiliza erros-padrão robustos. Incluir as métricas finais e corretas de R² e MSE.
    2.  **Equação:** Escrever a equação do modelo final com as variáveis e coeficientes corretos.
    3.  **Diagnóstico:** Apresentar e interpretar os gráficos de diagnóstico finais. Reconhecer as limitações (ex: resíduos não perfeitamente normais) e reforçar como a principal ameaça à validade (heterocedasticidade) foi tratada estatisticamente.
    4.  **Conclusão e Discussão:** Reescrever totalmente a conclusão para refletir os achados do modelo final. Discutir as implicações dos coeficientes (ex: o `Director_Power` negativo, a importância do `Market_Tier`, etc.), conforme sugerido anteriormente. 