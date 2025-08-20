# 🚀 Guia Simples para Treinar os Modelos

## 📋 O que vamos fazer (em ordem simples):

1. **Preparar os dados** (dividir em treino/teste)
2. **Treinar 4 modelos** básicos
3. **Comparar resultados** simples
4. **Fazer os gráficos** obrigatórios

---

## 🔧 PASSO 1: Preparar os Dados

```python
# Importar bibliotecas básicas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Carregar dados
df = pd.read_excel('Real estate valuation data set.xlsx')

# Separar features (X) e target (Y)
X = df.iloc[:, :-1]  # Todas as colunas exceto a última
y = df.iloc[:, -1]   # Última coluna (preço)

# Dividir em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados (importante para redes neurais)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Treino: {X_train.shape}")
print(f"Teste: {X_test.shape}")
```

---

## 🎯 PASSO 2: Treinar os Modelos

### **Modelo 1: Regressão Linear (MQ)**

```python
from sklearn.linear_model import LinearRegression

# Treinar
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predições
lr_train_pred = lr_model.predict(X_train_scaled)
lr_test_pred = lr_model.predict(X_test_scaled)

# Métricas
lr_train_r2 = r2_score(y_train, lr_train_pred)
lr_test_r2 = r2_score(y_test, lr_test_pred)

print(f"Regressão Linear - Treino R²: {lr_train_r2:.3f}")
print(f"Regressão Linear - Teste R²: {lr_test_r2:.3f}")
```

### **Modelo 2: Perceptron Simples (PS)**

```python
from sklearn.neural_network import MLPRegressor

# Perceptron = MLP com 0 camadas ocultas
ps_model = MLPRegressor(hidden_layer_sizes=(), max_iter=1000, random_state=42)
ps_model.fit(X_train_scaled, y_train)

# Predições
ps_train_pred = ps_model.predict(X_train_scaled)
ps_test_pred = ps_model.predict(X_test_scaled)

# Métricas
ps_train_r2 = r2_score(y_train, ps_train_pred)
ps_test_r2 = r2_score(y_test, ps_test_pred)

print(f"Perceptron Simples - Treino R²: {ps_train_r2:.3f}")
print(f"Perceptron Simples - Teste R²: {ps_test_r2:.3f}")
```

### **Modelo 3: MLP com 1 Camada Oculta**

```python
# MLP com 1 camada oculta (10 neurônios)
mlp1_model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
mlp1_model.fit(X_train_scaled, y_train)

# Predições
mlp1_train_pred = mlp1_model.predict(X_train_scaled)
mlp1_test_pred = mlp1_model.predict(X_test_scaled)

# Métricas
mlp1_train_r2 = r2_score(y_train, mlp1_train_pred)
mlp1_test_r2 = r2_score(y_test, mlp1_test_pred)

print(f"MLP 1 Camada - Treino R²: {mlp1_train_r2:.3f}")
print(f"MLP 1 Camada - Teste R²: {mlp1_test_r2:.3f}")
```

### **Modelo 4: MLP com 2 Camadas Ocultas**

```python
# MLP com 2 camadas ocultas (10, 5 neurônios)
mlp2_model = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
mlp2_model.fit(X_train_scaled, y_train)

# Predições
mlp2_train_pred = mlp2_model.predict(X_train_scaled)
mlp2_test_pred = mlp2_model.predict(X_test_scaled)

# Métricas
mlp2_train_r2 = r2_score(y_train, mlp2_train_pred)
mlp2_test_r2 = r2_score(y_test, mlp2_test_pred)

print(f"MLP 2 Camadas - Treino R²: {mlp2_train_r2:.3f}")
print(f"MLP 2 Camadas - Teste R²: {mlp2_test_r2:.3f}")
```

---

## 📊 PASSO 3: Comparar Resultados

```python
# Criar tabela de comparação
import pandas as pd

resultados = pd.DataFrame({
    'Modelo': ['Regressão Linear', 'Perceptron Simples', 'MLP 1 Camada', 'MLP 2 Camadas'],
    'R² Treino': [lr_train_r2, ps_train_r2, mlp1_train_r2, mlp2_train_r2],
    'R² Teste': [lr_test_r2, ps_test_r2, mlp1_test_r2, mlp2_test_r2]
})

print("=== COMPARAÇÃO DOS MODELOS ===")
print(resultados.round(3))

# Encontrar o melhor modelo
melhor_teste = resultados.loc[resultados['R² Teste'].idxmax()]
print(f"\n🎯 MELHOR MODELO: {melhor_teste['Modelo']}")
print(f"   R² Teste: {melhor_teste['R² Teste']:.3f}")
```

---

## 📈 PASSO 4: Gráficos Obrigatórios

### **1. Histograma dos Resíduos (do melhor modelo)**

```python
import matplotlib.pyplot as plt

# Escolher o melhor modelo (exemplo: MLP 2 camadas)
melhor_pred_train = mlp2_train_pred
melhor_pred_test = mlp2_test_pred

# Calcular resíduos (erros) do treino
residuos_treino = y_train - melhor_pred_train

# Histograma dos resíduos
plt.figure(figsize=(10, 6))
plt.hist(residuos_treino, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Histograma dos Resíduos (Dados de Treino)')
plt.xlabel('Resíduos (Valor Real - Valor Predito)')
plt.ylabel('Frequência')
plt.grid(True, alpha=0.3)
plt.show()

print("📊 COMENTÁRIO SOBRE O HISTOGRAMA:")
print("Se os resíduos seguem uma distribuição normal (gaussiana),")
print("o modelo está bem ajustado aos dados.")
```

### **2. Gráficos de Dispersão (Real vs Predito)**

```python
# Gráfico para dados de TREINO
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train, melhor_pred_train, alpha=0.6, color='blue')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Valor Real')
plt.ylabel('Valor Predito')
plt.title('Treino: Real vs Predito')
plt.grid(True, alpha=0.3)

# Gráfico para dados de TESTE
plt.subplot(1, 2, 2)
plt.scatter(y_test, melhor_pred_test, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valor Real')
plt.ylabel('Valor Predito')
plt.title('Teste: Real vs Predito')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("📊 COMENTÁRIO SOBRE OS GRÁFICOS:")
print("Se os pontos ficam próximos da linha vermelha (y=x),")
print("o modelo está fazendo boas predições.")
```

### **3. Coeficientes de Correlação**

```python
# Calcular correlações
from scipy.stats import pearsonr

corr_treino, _ = pearsonr(y_train, melhor_pred_train)
corr_teste, _ = pearsonr(y_test, melhor_pred_test)

print("=== COEFICIENTES DE CORRELAÇÃO ===")
print(f"Treino: {corr_treino:.3f}")
print(f"Teste:  {corr_teste:.3f}")

print("\n📊 COMENTÁRIO SOBRE CORRELAÇÕES:")
print("Valores próximos de 1.0 indicam forte correlação")
print("entre valores reais e preditos (bom modelo).")
```

---

## 🎉 RESUMO FINAL

```python
print("=== RESUMO FINAL ===")
print(f"✅ Modelos treinados: 4")
print(f"✅ Melhor modelo: {melhor_teste['Modelo']}")
print(f"✅ R² no teste: {melhor_teste['R² Teste']:.3f}")
print(f"✅ Correlação teste: {corr_teste:.3f}")
print(f"✅ Gráficos criados: 3")
print("\n🎯 PRÓXIMO PASSO: Comparar com o artigo de Yeh & Hsu (2018)")
```

---

## 💡 **DICAS IMPORTANTES:**

1. **Execute o código na ordem** - não pule passos
2. **Se der erro** - verifique se instalou as bibliotecas:
   ```bash
   pip install scikit-learn scipy matplotlib pandas numpy openpyxl
   ```
3. **Se um modelo demorar** - é normal para redes neurais
4. **Se R² for baixo** - tente ajustar parâmetros das redes neurais
5. **Guarde os resultados** - você vai precisar para comparar com o artigo

---

## 🔍 **O que cada modelo faz:**

- **Regressão Linear**: Linha reta que melhor se ajusta aos dados
- **Perceptron Simples**: Rede neural básica (sem camadas ocultas)
- **MLP 1 Camada**: Rede neural com uma camada intermediária
- **MLP 2 Camadas**: Rede neural com duas camadas intermediárias

**Quanto mais camadas, mais complexo o modelo pode ser, mas cuidado com overfitting!** 