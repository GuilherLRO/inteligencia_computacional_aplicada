### PROJETO 2: Reconhecimento de Faces

**Prof. Dr. Guilherme de Alencar Barreto**
$14/Julho/2025$
Departamento de Engenharia de Teleinformática (DETI)
Disciplinas: TIP7077 - Inteligência Computacional Aplicada
CCP9011 - Inteligência Computacional
Universidade Federal do Ceará (UFC), Campus do Pici, Fortaleza-CE

**Objetivo**
[cite_start]Desenvolver as habilidades de síntese de classificadores de padrões estudados na disciplina em problemas de processamento de imagens, extração de atributos via PCA e reconhecimento de pessoas a partir de imagens da face[cite: 5].

**Material fornecido:**
[cite_start]Kit de projeto com arquivos de imagens da face de 15 indivíduos (Yale A), código Matlab/Octave para processamento das imagens e geração do banco de amostras de treinamento e teste, bem como código PCA para redução de dimensionalidade das amostras de treinamento/teste[cite: 6].

---

### **1 Sequência de Atividades**

#### **• Atividade 1: Pré-processamento sem PCA**
[cite_start]Abrir e executar o arquivo `face_preprocessing_column.m` sem aplicação do PCA, comentando as linhas 56-60[cite: 8].
[cite_start]Escolha as dimensões para redução das imagens na linha 37[cite: 9]. [cite_start]Note que quanto maior os valores da redução, maior será a dimensão dos vetores de atributos após a vetorização das imagens e, obviamente, maior será o tempo de treinamento/teste dos classificadores[cite: 9].
[cite_start]**Exemplos:** $[20~20] \Rightarrow 20*20=400$[cite: 10]; [cite_start]$[30~30] \Rightarrow 30*30=900$[cite: 10].

#### **• Atividade 2: Execução dos Classificadores sem PCA**
[cite_start]Abrir e executar o arquivo `compara_todos.m` usando $P_{train}=80$, ou seja, 80% dos vetores de atributos serão usados para treinar os classificadores[cite: 11, 12]. [cite_start]Faça também $N_{r}=50$ (número de repetições independentes de treino/teste)[cite: 13].
[cite_start]Preencha a tabela de estatísticas de desempenho abaixo[cite: 13]. [cite_start]A figura de mérito é a taxa de acerto do classificador, com estatísticas descritivas (valor médio, desvio padrão, valores mínimo/máximo e mediana) determinadas ao final das 50 rodadas independentes[cite: 14].
[cite_start]**Classificadores a serem implementados:** Classificador Linear de Mínimos Quadrados (MQ), Perceptron Logístico (PL) e Perceptron Multicamadas com uma (MLP-1H) e duas camadas (MLP-2H)[cite: 15].
[cite_start]**OBS-2:** Antes de preencher a tabela, teste os classificadores para diferentes tipos de normalização dos atributos (sem normalização, com normalização z-score e normalização por mudança de escala [0,+1] ou [-1,+1])[cite: 23].
[cite_start]Teste também diferentes funções de ativação (sigmoidais, ReLu, etc.) e diferentes variações do método do gradiente descendente[cite: 24]. [cite_start]Inclua na tabela apenas o resultado da versão que deu melhor resultado[cite: 25].

[cite_start]**Tabela 1: Tabela de resultados sem a aplicação de PCA[cite: 21].**

| Classificador | Média | Mínimo | Máximo | Mediana | Desvio Padrão | Tempo de execução |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MQ | | | | | | |
| PL | | | | | | |
| MLP-1H | | | | | | |
| MLP-2H | | | | | | |

[cite_start]**Questão 1:** O que se pode concluir sobre os desempenhos dos classificadores avaliados? [cite: 26]
**Questão 2:** Qual deles teve o melhor desempenho em relação à taxa de acerto? [cite_start]E em relação ao tempo? [cite: 27]

---

### **2 Aplicação de PCA**

#### **• Atividade 3: Pré-processamento com PCA (sem redução de dimensionalidade)**
[cite_start]Executar o arquivo `face_preprocessing_column.m` com aplicação do PCA, ou seja, descomentar as linhas 56-60[cite: 28].
[cite_start]Faça $q=400$ ou $q=900$ na linha 57, a depender do redimensionamento das imagens escolhido na Atividade 1[cite: 29]. [cite_start]Note que, para este valor de q, a aplicação de PCA não conduz a uma redução da dimensionalidade, mas apenas promove a diagonalização da matriz de covariância dos dados transformados[cite: 29]. [cite_start]Em outras palavras, os atributos do novo conjunto de dados Z são descorrelacionados entre si[cite: 30].

#### **• Atividade 4: Execução dos Classificadores com PCA (sem redução de dimensionalidade)**
[cite_start]Executar novamente a Atividade 2, preenchendo a tabela de desempenho abaixo[cite: 31, 32].

[cite_start]**Tabela 2: Tabela de resultados com a aplicação de PCA sem redução de dimensionalidade[cite: 33].**

| Classificador | Média | Mínimo | Máximo | Mediana | Desvio Padrão | Tempo de execução |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MQ | | | | | | |
| PL | | | | | | |
| MLP-1H | | | | | | |
| MLP-2H | | | | | | |

[cite_start]**Questão 4:** O que se pode concluir sobre os desempenhos dos classificadores avaliados? [cite: 34] [cite_start]Houve alguma mudança (melhora ou piora) nos desempenhos em relação à tabela anterior? [cite: 35]

#### **• Atividade 5: Pré-processamento com PCA (com redução de dimensionalidade)**
[cite_start]Com base na figura de variância explicada acumulada, que mostra a variância explicada acumulada em função do número de componentes considerado, escolher um valor para `q` que preserve pelo menos 98% da informação (i.e., variância) dos dados originais[cite: 36]. [cite_start]O valor de `q` pode ser escolhido visualizando o conteúdo do vetor VEq, como sendo aquela componente cujo valor é maior que 98%[cite: 37].
[cite_start]Execute o arquivo `face_preprocessing_column.m` com PCA para o `q` escolhido[cite: 38]. [cite_start]Note que, para este valor de `q`, a aplicação de PCA conduz a uma redução da dimensionalidade, além de promover a descorrelação dos atributos dos dados transformados[cite: 39].

[cite_start]**Questão 5:** Qual foi a dimensão de redução `q` escolhida, de modo a preservar 98% da informação do conjunto de dados original? [cite: 40]

#### **• Atividade 6: Execução dos Classificadores com PCA (com redução de dimensionalidade)**
[cite_start]Com base no valor de `q` escolhido na Atividade 5, e no conjunto de dados gerados correspondente, treine os modelos e preencha a tabela abaixo com os resultados de desempenho com os dados de teste[cite: 41, 42].

[cite_start]**Tabela 3: Tabela de resultados com a aplicação de PCA com redução de dimensionalidade[cite: 43].**

| Classificador | Média | Mínimo | Máximo | Mediana | Desvio Padrão | Tempo de execução |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MQ | | | | | | |
| PL | | | | | | |
| MLP-1H | | | | | | |
| MLP-2H | | | | | | |

[cite_start]**Questão 6:** O que se pode concluir sobre os desempenhos dos classificadores avaliados com a realização da redução de dimensionalidade via PCA? [cite: 44] [cite_start]Houve alguma mudança (melhora ou piora) nos desempenhos em relação à tabela anterior? [cite: 45] [cite_start]Quais classificadores pioraram/melhoraram de desempenho com a redução de dimensionalidade via PCA? [cite: 46]

---

### **3 Transformações e Aplicações Adicionais**

#### **• Atividade 7: Aplicação da Transformação de Box-Cox**
[cite_start]Repita a Atividade 6, porém aplicando a transformação de Box-Cox aos dados transformados após a aplicação de PCA[cite: 47]. [cite_start]Em seguida, aplique a normalização z-score aos atributos dos dados transformados[cite: 48].

[cite_start]**Questão 7:** Houve alguma mudança (melhora ou piora) nos desempenhos dos classificadores em relação aos resultados da Atividade 6? [cite: 49] [cite_start]Quais classificadores pioraram/melhoraram com a aplicação da transformação Box-Cox juntamente com PCA? [cite: 50]

#### **• Atividade 8: Aplicações de Controle de Acesso**
[cite_start]Use a seguinte sequência de ações para projeto dos classificadores: Imagens vetorizadas + PCA + Box-Cox + normalização z-escore + Classificador[cite: 51, 52]. [cite_start]Adicione 11 imagens próprias ao conjunto de dados para atuar como "intruso" [cite: 53][cite_start], ou seja, indivíduo ao qual não deve ser dado acesso[cite: 54].

[cite_start]**Questão 8:** Calcule os seguintes índices de desempenho para os classificadores implementados: acurácia, taxa de falsos negativos (proporção de pessoas às quais acesso foi permitido incorretamente) e taxa de falsos positivos (pessoas às quais acesso não foi permitido incorretamente), sensibilidade e precisão[cite: 55]. [cite_start]Os valores devem ser médios com inclusão de medida de dispersão (e.g., desvio padrão) para 50 rodadas[cite: 56].

---

[cite_start]BOA SORTE!! [cite: 57]