# Parte 1: Análise Exploratória Completa dos Dados

## 📊 O que é Análise Exploratória de Dados (EDA)?

A Análise Exploratória de Dados é o primeiro passo fundamental em qualquer projeto de machine learning. Ela nos permite:
- **Entender** a estrutura dos dados
- **Identificar** problemas nos dados
- **Descobrir** padrões e relacionamentos
- **Tomar decisões** sobre pré-processamento

## 🚀 Passo a Passo da Implementação

### 1. Importação das Bibliotecas Necessárias

```python
# Bibliotecas para manipulação e análise de dados
import pandas as pd
import numpy as np

# Bibliotecas para visualização
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações para melhor visualização
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Para exibir todas as colunas do DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
```

### 2. Carregamento e Primeira Visualização dos Dados

```python
# Carregar o dataset
df = pd.read_excel('Real estate valuation data set.xlsx')

# Visualizar as primeiras linhas
print("=== PRIMEIRAS 5 LINHAS ===")
print(df.head())

# Visualizar as últimas linhas
print("\n=== ÚLTIMAS 5 LINHAS ===")
print(df.tail())

# Informações básicas sobre o dataset
print("\n=== INFORMAÇÕES BÁSICAS ===")
print(f"Forma do dataset: {df.shape}")
print(f"Número de linhas: {df.shape[0]}")
print(f"Número de colunas: {df.shape[1]}")
```

### 3. Análise da Estrutura dos Dados

```python
# Informações detalhadas sobre as colunas
print("=== INFORMAÇÕES DETALHADAS DAS COLUNAS ===")
print(df.info())

# Tipos de dados de cada coluna
print("\n=== TIPOS DE DADOS ===")
print(df.dtypes)

# Nomes das colunas
print("\n=== NOMES DAS COLUNAS ===")
for i, col in enumerate(df.columns):
    print(f"{i+1}. {col}")
```

### 4. Estatísticas Descritivas

```python
# Estatísticas descritivas básicas
print("=== ESTATÍSTICAS DESCRITIVAS ===")
print(df.describe())

# Estatísticas descritivas incluindo variáveis categóricas
print("\n=== ESTATÍSTICAS COMPLETAS ===")
print(df.describe(include='all'))

# Verificar se há valores únicos em cada coluna
print("\n=== VALORES ÚNICOS POR COLUNA ===")
for col in df.columns:
    unique_count = df[col].nunique()
    print(f"{col}: {unique_count} valores únicos")
```

### 5. Verificação de Valores Nulos/Missing

```python
# Verificar valores nulos
print("=== VERIFICAÇÃO DE VALORES NULOS ===")
null_counts = df.isnull().sum()
print("Valores nulos por coluna:")
print(null_counts)

# Porcentagem de valores nulos
print("\n=== PERCENTUAL DE VALORES NULOS ===")
null_percentages = (df.isnull().sum() / len(df)) * 100
for col, percentage in null_percentages.items():
    if percentage > 0:
        print(f"{col}: {percentage:.2f}%")
    else:
        print(f"{col}: 0% (sem valores nulos)")

# Visualização gráfica dos valores nulos
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
plt.title('Mapa de Valores Nulos no Dataset')
plt.tight_layout()
plt.show()
```

### 6. Análise das Variáveis Individuais

#### 6.1 Variáveis Numéricas
```python
# Selecionar apenas colunas numéricas
numeric_columns = df.select_dtypes(include=[np.number]).columns
print(f"Colunas numéricas: {list(numeric_columns)}")

# Histogramas para variáveis numéricas
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, col in enumerate(numeric_columns):
    if i < len(axes):
        axes[i].hist(df[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i].set_title(f'Distribuição de {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequência')
        axes[i].grid(True, alpha=0.3)

# Remover subplots vazios se houver
for i in range(len(numeric_columns), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

# Boxplots para identificar outliers
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, col in enumerate(numeric_columns):
    if i < len(axes):
        axes[i].boxplot(df[col])
        axes[i].set_title(f'Boxplot de {col}')
        axes[i].set_ylabel(col)
        axes[i].grid(True, alpha=0.3)

# Remover subplots vazios se houver
for i in range(len(numeric_columns), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()
```

#### 6.2 Análise de Outliers
```python
# Identificar outliers usando IQR (Interquartile Range)
print("=== ANÁLISE DE OUTLIERS (Método IQR) ===")
for col in numeric_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    print(f"\n{col}:")
    print(f"  Q1: {Q1:.2f}")
    print(f"  Q3: {Q3:.2f}")
    print(f"  IQR: {IQR:.2f}")
    print(f"  Limite inferior: {lower_bound:.2f}")
    print(f"  Limite superior: {upper_bound:.2f}")
    print(f"  Número de outliers: {len(outliers)}")
    print(f"  Percentual de outliers: {(len(outliers)/len(df)*100):.2f}%")
```

### 7. Análise de Correlações

```python
# Matriz de correlação
print("=== MATRIZ DE CORRELAÇÃO ===")
correlation_matrix = df[numeric_columns].corr()
print(correlation_matrix)

# Visualização da matriz de correlação
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, 
            mask=mask,
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.3f')
plt.title('Matriz de Correlação entre Variáveis Numéricas')
plt.tight_layout()
plt.show()

# Correlações com a variável target (assumindo que é a última coluna)
target_col = df.columns[-1]
print(f"\n=== CORRELAÇÕES COM A VARIÁVEL TARGET ({target_col}) ===")
target_correlations = correlation_matrix[target_col].sort_values(ascending=False)
print(target_correlations)
```

### 8. Análise de Distribuições Conjuntas

```python
# Scatter plots para variáveis mais correlacionadas com o target
top_correlations = target_correlations[1:4]  # Top 3 correlações (excluindo a própria variável)
print(f"Variáveis com maior correlação com {target_col}:")
print(top_correlations)

# Criar scatter plots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, (col, corr) in enumerate(top_correlations.items()):
    axes[i].scatter(df[col], df[target_col], alpha=0.6, color='blue')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel(target_col)
    axes[i].set_title(f'{col} vs {target_col}\nCorrelação: {corr:.3f}')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 9. Resumo da Análise Exploratória

```python
# Criar um resumo executivo
print("=== RESUMO EXECUTIVO DA ANÁLISE EXPLORATÓRIA ===")
print(f"Dataset: Real Estate Valuation")
print(f"Dimensões: {df.shape[0]} linhas × {df.shape[1]} colunas")
print(f"Variáveis numéricas: {len(numeric_columns)}")
print(f"Variáveis categóricas: {len(df.columns) - len(numeric_columns)}")
print(f"Valores nulos: {df.isnull().sum().sum()}")
print(f"Memória utilizada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print(f"\nVariável target: {target_col}")
print(f"Range da variável target: {df[target_col].min():.2f} - {df[target_col].max():.2f}")
print(f"Média da variável target: {df[target_col].mean():.2f}")
print(f"Desvio padrão da variável target: {df[target_col].std():.2f}")

print(f"\nTop 3 variáveis mais correlacionadas com {target_col}:")
for i, (col, corr) in enumerate(top_correlations.head(3).items()):
    print(f"  {i+1}. {col}: {corr:.3f}")
```

## 🎯 O que Aprendemos com Esta Análise?

### **Estrutura dos Dados:**
- Quantas observações e variáveis temos
- Quais são os tipos de dados
- Se há valores nulos ou problemas

### **Distribuições:**
- Como as variáveis se distribuem
- Se há outliers que podem afetar os modelos
- Se as variáveis seguem distribuições normais

### **Relacionamentos:**
- Quais variáveis estão mais correlacionadas com o target
- Se há multicolinearidade entre features
- Quais variáveis podem ser mais importantes para predição

### **Qualidade dos Dados:**
- Se precisamos tratar outliers
- Se precisamos normalizar variáveis
- Se há dados inconsistentes

## 🔍 Próximos Passos Após a EDA

1. **Pré-processamento dos dados** (se necessário)
2. **Divisão em treino/teste**
3. **Implementação dos modelos de regressão**
4. **Comparação de performance**

## 💡 Dicas Importantes

- **Sempre** comece com EDA antes de modelar
- **Documente** suas descobertas
- **Visualize** os dados de diferentes formas
- **Questionar** os dados: fazem sentido?
- **Identificar** padrões que podem ajudar na modelagem

Esta análise exploratória é fundamental para entender seus dados e tomar decisões informadas sobre como proceder com a modelagem! 