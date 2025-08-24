# Relatório do 1º Trabalho Computacional
## Inteligência Computacional Aplicada  

**Disciplinas:** TIP7077 - Inteligência Computacional Aplicada (PPGETI) / CCP9011 - Inteligência Computacional (PPGMMQ)
**Instituição:** Universidade Federal do Ceará (UFC)
**Professor Responsável:** Prof. Guilherme de Alencar Barreto
**Data de Entrega:** 28/08/2025
**Aluno:** Guilherme Lawrence Rebouças Oliveira
**Matrícula:** 586718

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
### 2.1 Análise exploratória

Observando-se a distribuição dos dados e a correlação entre eles, decidiu-se que todas, exceto a variável 'No', seriam utilizadas nos dados de treinamento dos modelos. Adicionalmente, a variável 'X3 distance to the nearest MRT station' foi escolhida para definição de um modelo inicial de comparação uma vez que possui as maiores correlações entre as demais e foi apontada no texto de referência como o tipo de variável com maior influência no valor do imóvel.

#### Distribuição dos dados numéricos
![[Pasted image 20250820084959.png]]
#### Matriz de correlação
![[Pasted image 20250820085058.png]]

### 2.2 Modelos Implementados
Para cada modelo foi avaliada a adequação dos resíduos à normalidade, a dispersão dos valores de saída, bem como a correlação entre os valores de treino e teste. Os resultados de cada tipo de modelo são abordados de forma gráfica e textual. Os valores de correlação (r) e o coeficiente de determinação (R²) são interpretados de acordo com a seguinte tabela:

| Faixa de valores | Interpretação (r) – Schober et al. (2018) | Interpretação (R²) – Chicco et al. (2021) |
| ---------------- | ----------------------------------------- | ----------------------------------------- |
| 0.00 – 0.10      | Desprezível                               | Muito fraco                               |
| 0.10 – 0.39      | Fraca                                     | Fraco                                     |
| 0.40 – 0.69      | Moderada                                  | Moderado                                  |
| 0.70 – 0.89      | Forte                                     | Moderadamente forte                       |
| 0.90 – 1.00      | Muito forte                               | Forte a muito forte                       |

#### 2.2.1 Regressão Linear Simples
- **Implementação:** Utilizando `sklearn.linear_model.LinearRegression`
- **Características:** Modelo linear clássico baseado no método dos mínimos quadrados utilizando-se apenas de uma variável preditora 'X3 distance to the nearest MRT station'.
- **Resultados**: 
	- *Métricas:* Coeficiente de correlação de Pearson (Treino): 0.6741 R² (Treino): 0.4545 Coeficiente de correlação de Pearson (Teste): 0.6789 R² (Teste): 0.4435
	- *Gaussianidade:* O histograma se mostra de forma levemente assimétrica, devido ao poder preditivo limitado do modelo que se utiliza apenas de uma variável e não é capaz de se adequar a não linearidades.
	- *Correlação e R²*: Ambos moderados, funcionando melhor para valores inferiores. Isso dá ao fato de o modelo se adequar melhor para valores inferiores de preço, mas ser incapaz de identificar valores mais altos com base apenas na variável escolhida.

![[Pasted image 20250820090446.png]]

#### 2.2.2 Regressão Linear Múltipla 
- **Implementação:** Utilizando `sklearn.linear_model.LinearRegression`
- **Características:** Modelo linear clássico baseado no método dos mínimos quadrados utilizando as múltiplas variáveis disponíveis.
- **Resultados**: 
	- *Métricas:* Coeficiente de correlação de Pearson (Treino): 0.7620 R² (Treino): 0.5806 Coeficiente de correlação de Pearson (Teste): 0.7683 R² (Teste): 0.5800
	- *Gaussianidade:* O histograma apresenta dois picos de distribuição, o que revela a limitação de um modelo linear na captação do comportamento da variável de estudo.
	- *Correlação e R²*: Com a utilização de outras variáveis, a principal limitação do modelo anterior é vencida, gerando um valor de correlação forte, mas um R² ainda moderado pela incapacidade de se ajustar a valores mais extremos provenientes de relações não lineares.

![[Pasted image 20250820094545.png]]

#### 2.2.3 Regressão Polinomial 
- **Implementação:** Utilizando `sklearn.linear_model.PolynomialFeatures` e `sklearn.preprocessing.PolynomialFeatures`
- **Características:** Modelo que captura relações não-lineares através da criação de features polinomiais das variáveis originais. Permite modelar curvas e superfícies complexas que não podem ser capturadas por modelos lineares simples.
- **Resultados**: 
	- *Métricas (Grau 2):* Coeficiente de correlação de Pearson (Treino): 0.8422 R² (Treino): 0.7092 Coeficiente de correlação de Pearson (Teste): 0.8136 R² (Teste): 0.6582
	* *Métricas (Grau 4):* Coeficiente de correlação de Pearson (Treino): 0.8843 R² (Treino): 0.7820 Coeficiente de correlação de Pearson (Teste): 0.5914 R² (Teste): 0.1320
	- *Gaussianidade:* Os histogramas já não apresentam mais de um pico nem uma grande assimetria como observada nos modelos anteriores, devido à inserção de múltiplas variáveis preditores e à capacidade de se adequar a não linearidades, o que diminui o erro para valores extremos.
	- *Correlação e R²*:  As métricas já passam a apresentar um valor que pode ser considerado alto, contudo, o R² ainda é limitado pela presença de valores muito extremos. Também observa-se que para o grau 4, o modelo já começa a apresentar sinais de *overfitting*, que se acentuam para graus maiores.
![[Pasted image 20250820090848.png]]

![[Pasted image 20250820090840.png]]

![[Pasted image 20250820091413.png]]

#### 2.2.4 Perceptron Logístico
- **Implementação:** Utilizando `sklearn.neural_network.MLPRegressor`
- **Características:** Rede neural artificial sem camadas ocultas, implementada como um modelo de regressão linear através de uma arquitetura neural. Equivalente matemático à regressão linear múltipla, mas com estrutura de rede neural. Configurada com função de ativação identidade, máximo de 100.000 iterações e estado aleatório fixo para reprodutibilidade.
-  **Resultados**: 
	* *Métricas:* Coeficiente de correlação de Pearson (Treino): 0.7307 R² (Treino): 0.5337 Coeficiente de correlação de Pearson (Teste): 0.7421 R² (Teste): 0.5402
	- *Gaussianidade:* As métricas voltam a apresentar dois picos de distribuição, uma vez que, como o modelo de regressão apresentado anteriormente, possui apenas componentes lineares, dessa forma apresentando diferenças similares em relação à curva Gaussiana.
	- *Correlação e R²*:  Similar às métricas resultantes da regressão linear múltipla, o valor de correlação é forte, mas o R² ainda é moderado pela incapacidade de se ajustar a valores mais extremos provenientes de relações não lineares.

![[Pasted image 20250820091143.png]]

#### 2.2.4 Rede MLP
- **Implementação:** Utilizando `sklearn.neural_network.MLPRegressor`
- **Características:** Rede neural artificial multicamadas (Multi-Layer Perceptron) com arquiteturas de 1 e 2 camadas ocultas. Utiliza função de ativação ReLU e é capaz de capturar relações não-lineares complexas através de transformações não-lineares das variáveis de entrada. Configurada com 1 camada (10 neurônios) e 2 camadas (10 e 5 neurônios), máximo de 10.000 iterações e estado aleatório fixo para reprodutibilidade.
-  **Resultados**: 
	* *Métricas (1 camada oculta):* Coeficiente de correlação de Pearson (Treino): 0.7733 R² (Treino): 0.5970 Coeficiente de correlação de Pearson (Teste): 0.7962 R² (Teste): 0.6248
	* * *Métricas (2 camada oculta):* Coeficiente de correlação de Pearson (Treino): 0.7633 R² (Treino): 0.5825 Coeficiente de correlação de Pearson (Teste): 0.7777 R² (Teste): 0.5938
	- *Gaussianidade:* Em ambos os modelos se observa uma melhoria na adequação à curva Gaussiana pela capacidade dos modelos de se adequar a não-linearidades principalmente nos intervalos com maior densidade de observações. Contudo, ainda são apresentados mais de um pico de densidade, possivelmente, devido à limitação no tamanho dos dados de treinamento.
	- *Correlação e R²*:  As duas métricas são similares para as duas arquiteturas, sendo superiores aos demais modelos implementados com exceção do modelo polinomial.
	
![[Pasted image 20250820091214.png]]

![[Pasted image 20250820091254.png]]

## 3 Discussão dos resultados
#### Correlação dos Modelos (Treino/Teste)
![[Pasted image 20250820103309.png]]

#### R² dos Modelos (Treino/Teste)
![[Pasted image 20250820103315.png]]



Ao contrário do relatado no **artigo de referência** (Yeh; Hsu, 2018), os modelos baseados em redes neurais testados neste trabalho **não apresentaram desempenho consistentemente superior** aos métodos de regressão. Entre os modelos avaliados, a **regressão polinomial múltipla de grau 2** destacou-se como a que obteve os melhores resultados tanto no coeficiente de correlação (r) quanto no coeficiente de determinação (R²).

Esse comportamento pode ser explicado **pelo tamanho reduzido do dataset** utilizado. Como argumentam Brigato e Iocchi (2020), em cenários com amostras limitadas, modelos de baixa complexidade tendem a ser mais robustos ao sobreajuste, podendo inclusive superar arquiteturas mais profundas. 

Portanto, pode-se concluir que, neste conjunto de dados específico, **modelos regressivos** se mostraram mais adequados, equilibrando precisão e generalização. 


## 4 Referências
* SCHOBER, P.; BOER, C.; SCHWARTE, L. A. Correlation coefficients: appropriate use and interpretation.*Anesthesia & Analgesia*, v. 126, n. 5, p. 1763-1768, 2018.

* CHICCO, D.; WARRENS, M. J.; JURMAN, G. The coefficient of determination R² and adjusted R² in regression.*Psychological Methods*, v. 26, n. 4, p. 612-635, 2021.

* BRIGATO, L.; IOCCHI, L. A close look at deep learning with small data.*Pattern Recognition Letters*, v. 135, p. 96-104, 2020.