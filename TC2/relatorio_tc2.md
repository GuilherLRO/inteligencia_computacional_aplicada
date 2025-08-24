# Relatório do 2º Trabalho Computacional
## Inteligência Computacional Aplicada  

**Disciplinas:** TIP7077 - Inteligência Computacional Aplicada (PPGETI) / CCP9011 - Inteligência Computacional (PPGMMQ)
**Instituição:** Universidade Federal do Ceará (UFC)
**Professor Responsável:** Prof. Guilherme de Alencar Barreto
**Data de Entrega:** [DATA]
**Aluno:** Guilherme Lawrence Rebouças Oliveira
**Matrícula:** 586718

---
## 1. Introdução

Este relatório apresenta a implementação e análise comparativa de classificadores para reconhecimento de faces utilizando o conjunto de dados Yale A, conforme solicitado no 2º Trabalho Computacional da disciplina de Inteligência Computacional Aplicada.

### 1.1 Objetivos
- Implementar classificadores de padrões para reconhecimento de faces
- Aplicar técnicas de extração de atributos via PCA
- Analisar o impacto da redução de dimensionalidade no desempenho dos classificadores
- Implementar transformações adicionais (Box-Cox) e avaliar seus efeitos
- Desenvolver sistema de controle de acesso com detecção de intrusos

### 1.2 Conjunto de Dados
O conjunto de dados utilizado é o Yale A, contendo imagens de faces de 15 indivíduos em diferentes condições de iluminação e expressões faciais.

---
## 2. Metodologia

### 2.1 Pré-processamento das Imagens
- Redimensionamento das imagens para dimensões específicas
- Vetorização das imagens para geração de vetores de atributos
- Aplicação de PCA com e sem redução de dimensionalidade

### 2.2 Classificadores Implementados
- **MQ:** Classificador Linear de Mínimos Quadrados
- **PL:** Perceptron Logístico
- **MLP-1H:** Perceptron Multicamadas com 1 camada oculta
- **MLP-2H:** Perceptron Multicamadas com 2 camadas ocultas

### 2.3 Configurações de Teste
- **Proporção de treinamento:** 80% (P_train = 80)
- **Número de repetições:** 50 (N_r = 50)
- **Diferentes normalizações:** Sem normalização, z-score, [0,+1], [-1,+1]
- **Funções de ativação:** Sigmoidais, ReLU
- **Métodos de otimização:** Variações do gradiente descendente

---
## 3. Resultados e Análise

### 3.1 Classificadores sem PCA
**Tabela 1: Resultados sem aplicação de PCA**

| Classificador | Média | Mínimo | Máximo | Mediana | Desvio Padrão | Tempo de execução |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MQ | | | | | | |
| PL | | | | | | |
| MLP-1H | | | | | | |
| MLP-2H | | | | | | |

**Questão 1:** [RESPOSTA]
**Questão 2:** [RESPOSTA]

### 3.2 Classificadores com PCA (sem redução de dimensionalidade)
**Tabela 2: Resultados com PCA sem redução de dimensionalidade**

| Classificador | Média | Mínimo | Máximo | Mediana | Desvio Padrão | Tempo de execução |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MQ | | | | | | |
| PL | | | | | | |
| MLP-1H | | | | | | |
| MLP-2H | | | | | | |

**Questão 4:** [RESPOSTA]

### 3.3 Classificadores com PCA (com redução de dimensionalidade)
**Dimensão escolhida para preservar 98% da variância:** q = [VALOR]

**Tabela 3: Resultados com PCA com redução de dimensionalidade**

| Classificador | Média | Mínimo | Máximo | Mediana | Desvio Padrão | Tempo de execução |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MQ | | | | | | |
| PL | | | | | | |
| MLP-1H | | | | | | |
| MLP-2H | | | | | | |

**Questão 6:** [RESPOSTA]

### 3.4 Aplicação da Transformação de Box-Cox
**Questão 7:** [RESPOSTA]

### 3.5 Sistema de Controle de Acesso
**Resultados com detecção de intrusos (11 imagens adicionais):**

| Classificador | Acurácia | Taxa Falsos Negativos | Taxa Falsos Positivos | Sensibilidade | Precisão |
| :--- | :--- | :--- | :--- | :--- | :--- |
| MQ | | | | | |
| PL | | | | | |
| MLP-1H | | | | | |
| MLP-2H | | | | | |

**Questão 8:** [RESPOSTA]

---
## 4. Discussão dos Resultados

### 4.1 Comparação entre Classificadores
[ANÁLISE COMPARATIVA]

### 4.2 Impacto do PCA
[ANÁLISE DO IMPACTO DO PCA]

### 4.3 Efeitos da Redução de Dimensionalidade
[ANÁLISE DOS EFEITOS DA REDUÇÃO]

### 4.4 Transformações Adicionais
[ANÁLISE DAS TRANSFORMAÇÕES BOX-COX]

### 4.5 Desempenho no Sistema de Controle de Acesso
[ANÁLISE DO SISTEMA DE CONTROLE]

---
## 5. Conclusões

[CONCLUSÕES PRINCIPAIS]

---
## 6. Referências

[LISTA DE REFERÊNCIAS UTILIZADAS]
