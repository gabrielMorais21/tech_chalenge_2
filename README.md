# Tech Challenge Fase 2: Otimização de Modelos de Diagnóstico de Câncer de Mama

**Nome:** Gabriel Morais Dias

---

## 1. Introdução e Objetivo
O presente projeto tem como objetivo otimizar a precisão e a eficiência de modelos de Machine Learning utilizados no diagnóstico de câncer de mama. Para isso, implementamos um **Algoritmo Genético (AG)** para a busca dos melhores hiperparâmetros de um modelo `RandomForestClassifier`. Adicionalmente, integramos recursos avançados de Nuvem (Azure ML e MLflow) para MLOps e Processamento de Linguagem Natural (Azure OpenAI) para traduzir as predições matemáticas em laudos médicos interpretáveis.

---

## 2. Arquitetura da Solução em Nuvem
O projeto foi totalmente desenvolvido e provisionado no ecossistema Microsoft Azure, garantindo escalabilidade e rastreabilidade.
* **Compute Instance:** Treinamento realizado em ambiente gerenciado do Azure Machine Learning.
* **Monitoramento e Logging:** Utilização do `MLflow` para versionamento de experimentos, rastreamento de métricas e registro de artefatos (Model Registry).
* **LLM Integration:** Uso do modelo `gpt-4o` via Azure OpenAI Service para Geração de Texto.

  ![azure_diagram](https://github.com/user-attachments/assets/29ff7842-7e99-4bdb-a2f7-b07bc46cbcdc)

---

## 3. Otimização via Algoritmos Genéticos
Em vez de utilizar buscas exaustivas (como GridSearch), modelamos o problema biologicamente para explorar o espaço de busca de forma mais eficiente:
* **Representação dos Genes (Codificação):** Cada indivíduo é composto por três genes que representam os hiperparâmetros da Floresta Aleatória: `n_estimators` (int), `max_depth` (int) e `min_samples_split` (float).
* **Função Fitness:** Utilizamos a validação cruzada (3-fold cross-validation) otimizando o **F1-Score**. A escolha do F1-Score foi uma decisão clínica: no diagnóstico de câncer, é vital punir tanto falsos positivos quanto falsos negativos.
* **Operadores:** Implementamos torneio para *Seleção*, cruzamento de ponto único (*Crossover*) e *Mutação* tipada (garantindo a integridade dos tipos de dados exigidos pelo Scikit-Learn).

---

## 4. Experimentos e Resultados
Foram realizados 3 experimentos distintos para analisar o comportamento da evolução:

1. **Exp 1: Conservador** (Pop: 10, Mutação: 5%)
2. **Exp 2: Exploratório** (Pop: 15, Mutação: 40%)
3. **Exp 3: Robusto** (Pop: 40, Mutação: 10%)

O melhor modelo foi obtido no **Experimento 3 (Robusto)**, alcançando um F1-Score de **0.9395515**. Ao comparar o modelo otimizado com a baseline (Random Forest com parâmetros padrão), observamos que o Algoritmo Genético conseguiu manter o alto recall ao mesmo tempo em que calibrou a complexidade da árvore.

---

## 5. Integração com LLMs (Processamento de Linguagem Natural)
Para resolver o desafio da interpretabilidade "caixa-preta" dos modelos de IA na saúde, integramos o `Azure OpenAI`. 
Aplicamos técnicas de **Prompt Engineering** estruturando o modelo com a persona de um *"assistente médico especialista em oncologia"*. O LLM recebe a predição binária, a probabilidade e os hiperparâmetros matemáticos, e gera automaticamente:
1. A explicação do diagnóstico em linguagem natural.
2. Insights acionáveis para a equipe médica.
3. Recomendações de próximos passos clínicos (ex: biópsia, exames de imagem).

---

## 6. Qualidade e Boas Práticas (Engenharia de Software)
* O código foi modularizado, isolando as funções matemáticas no arquivo `ag_functions.py`.
* Foram implementados **Testes Automatizados** utilizando a biblioteca `pytest` no arquivo `test_ag.py` para garantir a integridade dos operadores genéticos antes do treinamento.
* O modelo vencedor foi registrado no catálogo de Modelos da Azure, pronto para deploy.
