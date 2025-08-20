# Relatório do 1º Trabalho Computacional
## Inteligência Computacional Aplicada

**Disciplinas:** TIP7077 - Inteligência Computacional Aplicada (PPGETI) / CCP9011 - Inteligência Computacional (PPGMMQ)  
**Instituição:** Universidade Federal do Ceará (UFC)  
**Professor Responsável:** Prof. Guilherme de Alencar Barreto  
**Data de Entrega:** 14/07/2025  
**Aluno:** [Seu Nome]  

---

## 1. Introdução

Este relatório apresenta a implementação e análise comparativa de modelos de regressão para o conjunto de dados "Real estate valuation data set", conforme solicitado no 1º Trabalho Computacional da disciplina de Inteligência Computacional Aplicada.

### 1.1 Objetivos

- Implementar modelos de regressão linear múltipla de mínimos quadrados (MQ)
- Implementar rede Perceptron Logístico (PS)
- Implementar rede MLP com 1 e 2 camadas ocultas
- Comparar os resultados com o artigo de referência
- Analisar a qualidade dos modelos através de métricas de avaliação

### 1.2 Conjunto de Dados

O conjunto de dados utilizado é o "Real estate valuation data set", disponível no repositório UCI Machine Learning Repository. Este dataset contém informações sobre avaliação de imóveis e será utilizado para treinar e avaliar os diferentes modelos de regressão implementados.

**Referência do Dataset:** [https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set](https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set)

**Artigo de Referência:** Yeh, I. C., & Hsu, T. K. (2018). Building real estate valuation models with comparative approach through case-based reasoning. *Applied Soft Computing*, 65, 260-271.

---

## 2. Metodologia

### 2.1 Modelos Implementados

#### 2.1.1 Regressão Linear Múltipla (MQ)
- **Implementação:** Utilizando `sklearn.linear_model.LinearRegression`
- **Características:** Modelo linear clássico baseado no método dos mínimos quadrados
- **Aplicação:** Baseline para comparação com modelos mais complexos

#### 2.1.2 Perceptron Logístico (PS)
- **Implementação:** Utilizando `sklearn.neural_network.MLPRegressor` com configuração específica
- **Configuração:** 
  - Sem camadas ocultas (`hidden_layer_sizes=()`)
  - Função de ativação identidade (`activation='identity'`)
  - Máximo de 100.000 iterações
  - Estado aleatório fixo para reprodutibilidade

#### 2.1.3 Rede MLP com 1 Camada Oculta
- **Implementação:** Utilizando `sklearn.neural_network.MLPRegressor`
- **Configuração:**
  - 1 camada oculta com 10 neurônios (`hidden_layer_sizes=(10,)`)
  - Função de ativação ReLU (`activation='relu'`)
  - Máximo de 10.000 iterações
  - Estado aleatório fixo para reprodutibilidade

#### 2.1.4 Rede MLP com 2 Camadas Ocultas
- **Implementação:** Utilizando `sklearn.neural_network.MLPRegressor`
- **Configuração:**
  - 2 camadas ocultas com 10 e 5 neurônios respectivamente (`hidden_layer_sizes=(10, 5)`)
  - Função de ativação ReLU (`activation='relu'`)
  - Máximo de 10.000 iterações
  - Estado aleatório fixo para reprodutibilidade

### 2.2 Pré-processamento dos Dados

- **Divisão dos dados:** 80% para treinamento, 20% para teste
- **Normalização:** Aplicada quando necessário para melhor convergência dos modelos neurais
- **Feature Engineering:** Exploração de características polinomiais para modelos lineares

### 2.3 Métricas de Avaliação

Para cada modelo, foram calculadas as seguintes métricas:
- **Coeficiente de Determinação (R²):** Para dados de treino e teste
- **Coeficiente de Correlação:** Entre valores reais e preditos
- **Análise de Resíduos:** Histogramas para verificar gaussianidade
- **Gráficos de Dispersão:** Valores reais vs. preditos

---

## 3. Resultados e Análise

### 3.1 Performance dos Modelos

#### 3.1.1 Regressão Linear Múltipla
- **R² Treino:** [Valor a ser preenchido com resultado real]
- **R² Teste:** [Valor a ser preenchido com resultado real]
- **Correlação Treino:** [Valor a ser preenchido com resultado real]
- **Correlação Teste:** [Valor a ser preenchido com resultado real]

#### 3.1.2 Perceptron Logístico
- **R² Treino:** [Valor a ser preenchido com resultado real]
- **R² Teste:** [Valor a ser preenchido com resultado real]
- **Correlação Treino:** [Valor a ser preenchido com resultado real]
- **Correlação Teste:** [Valor a ser preenchido com resultado real]

#### 3.1.3 MLP com 1 Camada Oculta
- **R² Treino:** [Valor a ser preenchido com resultado real]
- **R² Teste:** [Valor a ser preenchido com resultado real]
- **Correlação Treino:** [Valor a ser preenchido com resultado real]
- **Correlação Teste:** [Valor a ser preenchido com resultado real]

#### 3.1.4 MLP com 2 Camadas Ocultas
- **R² Treino:** [Valor a ser preenchido com resultado real]
- **R² Teste:** [Valor a ser preenchido com resultado real]
- **Correlação Treino:** [Valor a ser preenchido com resultado real]
- **Correlação Teste:** [Valor a ser preenchido com resultado real]

### 3.2 Análise de Resíduos

#### 3.2.1 Histogramas dos Resíduos
Para cada modelo implementado, foram gerados histogramas dos resíduos utilizando apenas os dados de treinamento. Esta análise permite verificar:

- **Distribuição dos erros:** Verificação da gaussianidade dos resíduos
- **Viés do modelo:** Identificação de padrões sistemáticos nos erros
- **Homocedasticidade:** Constância da variância dos erros

**Observações sobre os histogramas:**
- [Comentários específicos sobre a distribuição dos resíduos de cada modelo]
- [Análise da gaussianidade observada vs. esperada]

#### 3.2.2 Gráficos de Dispersão
Foram gerados gráficos comparando valores reais vs. valores preditos para dados de treino e teste, permitindo:

- **Avaliação da qualidade preditiva:** Proximidade dos pontos à linha ideal (y=x)
- **Identificação de padrões:** Detecção de viés ou não-linearidades
- **Comparação entre treino e teste:** Avaliação da generalização

**Análise dos gráficos:**
- [Comentários sobre a qualidade dos ajustes]
- [Observações sobre overfitting/underfitting]
- [Comparação entre modelos]

### 3.3 Coeficientes de Correlação

Os coeficientes de correlação entre valores reais e preditos foram calculados para todos os modelos, tanto para dados de treino quanto para dados de teste. Esta métrica permite:

- **Quantificação da qualidade do ajuste:** Valores próximos a 1 indicam boa correlação
- **Comparação entre modelos:** Identificação do melhor modelo
- **Avaliação da generalização:** Diferença entre correlação de treino e teste

**Interpretação dos valores obtidos:**
- [Análise dos coeficientes de correlação]
- [Comparação com valores esperados para bons ajustes]
- [Discussão sobre a qualidade preditiva dos modelos]

---

## 4. Discussão

### 4.1 Comparação entre Modelos

A análise comparativa dos modelos implementados revela:

- **Modelo Linear:** [Comentários sobre performance e limitações]
- **Perceptron Logístico:** [Análise da capacidade de capturar não-linearidades]
- **MLP com 1 Camada:** [Discussão sobre complexidade vs. performance]
- **MLP com 2 Camadas:** [Análise da capacidade de modelagem]

### 4.2 Comparação com o Artigo de Referência

Os resultados obtidos foram comparados com aqueles reportados no artigo de Yeh & Hsu (2018):

- **Similaridades:** [Identificação de padrões similares]
- **Diferenças:** [Discussão de divergências]
- **Possíveis explicações:** [Análise de fatores que podem explicar diferenças]

### 4.3 Limitações e Considerações

- **Tamanho do dataset:** [Discussão sobre adequação do tamanho amostral]
- **Overfitting:** [Análise do risco de overfitting nos modelos neurais]
- **Reprodutibilidade:** [Discussão sobre estabilidade dos resultados]

---

## 5. Conclusões

### 5.1 Principais Achados

- [Resumo dos principais resultados obtidos]
- [Identificação do melhor modelo]
- [Insights sobre a adequação dos diferentes algoritmos]

### 5.2 Contribuições

- [Contribuições do trabalho para o entendimento do problema]
- [Validação ou refutação de hipóteses do artigo de referência]
- [Novos insights obtidos]

### 5.3 Trabalhos Futuros

- [Sugestões para melhorias nos modelos]
- [Possíveis extensões do trabalho]
- [Direções para pesquisa futura]

---

## 6. Referências

1. Yeh, I. C., & Hsu, T. K. (2018). Building real estate valuation models with comparative approach through case-based reasoning. *Applied Soft Computing*, 65, 260-271.

2. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of machine learning research*, 12, 2825-2830.

3. Dua, D., & Graff, C. (2017). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.

---

## 7. Anexos

### 7.1 Código Fonte
O código completo da implementação está disponível no arquivo: `assigment_1_solution.ipynb`

### 7.2 Imagens e Gráficos
Todos os gráficos, histogramas e visualizações gerados estão incluídos no notebook de solução.

### 7.3 Dados Utilizados
O conjunto de dados "Real estate valuation data set" está disponível em formato Excel no arquivo: `Real estate valuation data set.xlsx`

---

**Data de Conclusão:** [Data]  
**Assinatura do Aluno:** _________________  
**Assinatura do Professor:** _________________ 