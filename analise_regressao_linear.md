# 📊 Análise Completa do Modelo de Regressão Linear

## 🎯 O que vamos implementar:

1. **Histograma dos resíduos** (dados de treinamento)
2. **Gráficos de dispersão** (real vs predito para treino e teste)
3. **Coeficientes de correlação** (treino e teste)
4. **Análise e comentários** sobre os resultados

---

## 🔧 IMPLEMENTAÇÃO COMPLETA

### **PASSO 1: Preparar e Treinar o Modelo**

```python
# Importar bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, shapiro
import warnings
warnings.filterwarnings('ignore')

# Configurações de visualização
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Carregar dados
df = pd.read_excel('Real estate valuation data set.xlsx')

# Separar features (X) e target (Y)
X = df.iloc[:, :-1]  # Todas as colunas exceto a última
y = df.iloc[:, -1]   # Última coluna (preço)

# Dividir em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("=== INFORMAÇÕES DOS DADOS ===")
print(f"Forma dos dados: {df.shape}")
print(f"Treino: {X_train.shape}")
print(f"Teste: {X_test.shape}")
print(f"Variável target: {y.name}")
print(f"Features: {list(X.columns)}")

# Treinar modelo de regressão linear
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Fazer predições
lr_train_pred = lr_model.predict(X_train_scaled)
lr_test_pred = lr_model.predict(X_test_scaled)

# Calcular métricas básicas
lr_train_r2 = r2_score(y_train, lr_train_pred)
lr_test_r2 = r2_score(y_test, lr_test_pred)

print(f"\n=== MÉTRICAS DO MODELO ===")
print(f"R² Treino: {lr_train_r2:.4f}")
print(f"R² Teste:  {lr_test_r2:.4f}")
```

---

## 📈 PASSO 2: Histograma dos Resíduos (Dados de Treinamento)

```python
# Calcular resíduos (erros) para dados de treinamento
residuos_treino = y_train - lr_train_pred

# Criar figura com subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Histograma dos resíduos
axes[0, 0].hist(residuos_treino, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
axes[0, 0].set_title('Histograma dos Resíduos (Dados de Treinamento)', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Resíduos (Valor Real - Valor Predito)')
axes[0, 0].set_ylabel('Densidade')
axes[0, 0].grid(True, alpha=0.3)

# 2. Adicionar curva normal teórica
from scipy.stats import norm
mu, sigma = norm.fit(residuos_treino)
x = np.linspace(residuos_treino.min(), residuos_treino.max(), 100)
y_normal = norm.pdf(x, mu, sigma)
axes[0, 0].plot(x, y_normal, 'r-', linewidth=2, label=f'Normal (μ={mu:.2f}, σ={sigma:.2f})')
axes[0, 0].legend()

# 3. QQ-Plot para verificar normalidade
from scipy.stats import probplot
probplot(residuos_treino, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot dos Resíduos', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 4. Gráfico de resíduos vs preditos
axes[1, 0].scatter(lr_train_pred, residuos_treino, alpha=0.6, color='green')
axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_title('Resíduos vs Valores Preditos', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Valores Preditos')
axes[1, 0].set_ylabel('Resíduos')
axes[1, 0].grid(True, alpha=0.3)

# 5. Boxplot dos resíduos
axes[1, 1].boxplot(residuos_treino, patch_artist=True, boxprops=dict(facecolor='lightblue'))
axes[1, 1].set_title('Boxplot dos Resíduos', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Resíduos')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Teste de normalidade (Shapiro-Wilk)
statistic, p_value = shapiro(residuos_treino)
print(f"\n=== TESTE DE NORMALIDADE (Shapiro-Wilk) ===")
print(f"Estatística: {statistic:.4f}")
print(f"P-valor: {p_value:.4f}")
print(f"Resíduos são normais? {'SIM' if p_value > 0.05 else 'NÃO'} (α=0.05)")

# Estatísticas descritivas dos resíduos
print(f"\n=== ESTATÍSTICAS DOS RESÍDUOS ===")
print(f"Média: {np.mean(residuos_treino):.4f}")
print(f"Desvio padrão: {np.std(residuos_treino):.4f}")
print(f"Mínimo: {np.min(residuos_treino):.4f}")
print(f"Máximo: {np.max(residuos_treino):.4f}")
print(f"Assimetria: {pd.Series(residuos_treino).skew():.4f}")
print(f"Curtose: {pd.Series(residuos_treino).kurtosis():.4f}")
```

---

## 📊 PASSO 3: Gráficos de Dispersão (Real vs Predito)

```python
# Criar gráficos de dispersão para treino e teste
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico para dados de TREINO
axes[0].scatter(y_train, lr_train_pred, alpha=0.6, color='blue', s=50)
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=3, label='Linha de Perfeita Predição')
axes[0].set_xlabel('Valor Real (Preço)', fontsize=12)
axes[0].set_ylabel('Valor Predito (Preço)', fontsize=12)
axes[0].set_title('TREINO: Real vs Predito\nRegressão Linear', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Adicionar R² no gráfico
axes[0].text(0.05, 0.95, f'R² = {lr_train_r2:.4f}', transform=axes[0].transAxes, 
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
              fontsize=12, verticalalignment='top')

# Gráfico para dados de TESTE
axes[1].scatter(y_test, lr_test_pred, alpha=0.6, color='green', s=50)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3, label='Linha de Perfeita Predição')
axes[1].set_xlabel('Valor Real (Preço)', fontsize=12)
axes[1].set_ylabel('Valor Predito (Preço)', fontsize=12)
axes[1].set_title('TESTE: Real vs Predito\nRegressão Linear', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Adicionar R² no gráfico
axes[1].text(0.05, 0.95, f'R² = {lr_test_r2:.4f}', transform=axes[1].transAxes, 
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
              fontsize=12, verticalalignment='top')

plt.tight_layout()
plt.show()

# Calcular erros para análise
mse_treino = mean_squared_error(y_train, lr_train_pred)
mse_teste = mean_squared_error(y_test, lr_test_pred)
rmse_treino = np.sqrt(mse_treino)
rmse_teste = np.sqrt(mse_teste)
mae_treino = mean_absolute_error(y_train, lr_train_pred)
mae_teste = mean_absolute_error(y_test, lr_test_pred)

print(f"\n=== MÉTRICAS DE ERRO ===")
print(f"Treino - MSE: {mse_treino:.4f}, RMSE: {rmse_treino:.4f}, MAE: {mae_treino:.4f}")
print(f"Teste  - MSE: {mse_teste:.4f}, RMSE: {rmse_teste:.4f}, MAE: {mae_teste:.4f}")
```

---

## 🔗 PASSO 4: Coeficientes de Correlação

```python
# Calcular correlações de Pearson
corr_treino, p_valor_treino = pearsonr(y_train, lr_train_pred)
corr_teste, p_valor_teste = pearsonr(y_test, lr_test_pred)

# Calcular correlação de Spearman (robusta a outliers)
from scipy.stats import spearmanr
corr_spearman_treino, p_spearman_treino = spearmanr(y_train, lr_train_pred)
corr_spearman_teste, p_spearman_teste = spearmanr(y_test, lr_test_pred)

# Criar tabela de correlações
print("=== COEFICIENTES DE CORRELAÇÃO ===")
print("=" * 60)
print(f"{'Métrica':<20} {'Treino':<15} {'Teste':<15}")
print("=" * 60)
print(f"{'Correlação Pearson':<20} {corr_treino:<15.4f} {corr_teste:<15.4f}")
print(f"{'P-valor Pearson':<20} {p_valor_treino:<15.4f} {p_valor_teste:<15.4f}")
print(f"{'Correlação Spearman':<20} {corr_spearman_treino:<15.4f} {corr_spearman_teste:<15.4f}")
print(f"{'P-valor Spearman':<20} {p_spearman_treino:<15.4f} {p_spearman_teste:<15.4f}")
print("=" * 60)

# Interpretação das correlações
print(f"\n=== INTERPRETAÇÃO DAS CORRELAÇÕES ===")
print(f"📊 Correlação Treino: {corr_treino:.4f}")
if corr_treino >= 0.9:
    print("   ✅ EXCELENTE: Correlação muito forte")
elif corr_treino >= 0.8:
    print("   ✅ MUITO BOM: Correlação forte")
elif corr_treino >= 0.7:
    print("   ✅ BOM: Correlação moderadamente forte")
elif corr_treino >= 0.5:
    print("   ⚠️  MODERADO: Correlação moderada")
else:
    print("   ❌ BAIXO: Correlação fraca")

print(f"\n📊 Correlação Teste: {corr_teste:.4f}")
if corr_teste >= 0.9:
    print("   ✅ EXCELENTE: Correlação muito forte")
elif corr_teste >= 0.8:
    print("   ✅ MUITO BOM: Correlação forte")
elif corr_teste >= 0.7:
    print("   ✅ BOM: Correlação moderadamente forte")
elif corr_teste >= 0.5:
    print("   ⚠️  MODERADO: Correlação moderada")
else:
    print("   ❌ BAIXO: Correlação fraca")
```

---

## 📝 PASSO 5: Análise e Comentários

```python
# Criar resumo executivo
print("\n" + "="*80)
print("📋 RESUMO EXECUTIVO - MODELO DE REGRESSÃO LINEAR")
print("="*80)

print(f"\n🎯 PERFORMANCE DO MODELO:")
print(f"   • R² Treino: {lr_train_r2:.4f}")
print(f"   • R² Teste:  {lr_test_r2:.4f}")
print(f"   • Diferença: {abs(lr_train_r2 - lr_test_r2):.4f}")

if abs(lr_train_r2 - lr_test_r2) < 0.05:
    print("   ✅ Modelo estável (pouca diferença treino/teste)")
else:
    print("   ⚠️  Possível overfitting (grande diferença treino/teste)")

print(f"\n📊 ANÁLISE DOS RESÍDUOS:")
print(f"   • Média dos resíduos: {np.mean(residuos_treino):.4f} (deve ser próxima de 0)")
print(f"   • Desvio padrão: {np.std(residuos_treino):.4f}")
print(f"   • Normalidade (Shapiro-Wilk): {'SIM' if p_value > 0.05 else 'NÃO'}")

print(f"\n🔗 CORRELAÇÕES:")
print(f"   • Treino: {corr_treino:.4f}")
print(f"   • Teste:  {corr_teste:.4f}")

print(f"\n💡 INTERPRETAÇÃO GERAL:")
if lr_test_r2 >= 0.8 and corr_teste >= 0.8:
    print("   ✅ EXCELENTE: Modelo muito bem ajustado aos dados")
elif lr_test_r2 >= 0.7 and corr_teste >= 0.7:
    print("   ✅ BOM: Modelo bem ajustado aos dados")
elif lr_test_r2 >= 0.5 and corr_teste >= 0.5:
    print("   ⚠️  MODERADO: Modelo com ajuste moderado")
else:
    print("   ❌ BAIXO: Modelo com ajuste insuficiente")

print("="*80)
```

---

## 🎯 **RESPOSTAS AOS REQUISITOS:**

### **1. Histograma dos Resíduos:**
- ✅ **Implementado** com curva normal teórica
- ✅ **Teste de normalidade** (Shapiro-Wilk)
- ✅ **QQ-Plot** para verificação visual
- ✅ **Análise de assimetria e curtose**

### **2. Gráficos de Dispersão:**
- ✅ **Treino vs Teste** lado a lado
- ✅ **Linha de perfeita predição** (y=x)
- ✅ **R²** exibido em cada gráfico
- ✅ **Métricas de erro** (MSE, RMSE, MAE)

### **3. Coeficientes de Correlação:**
- ✅ **Pearson e Spearman** para robustez
- ✅ **P-valores** para significância estatística
- ✅ **Interpretação automática** dos valores
- ✅ **Comparação treino vs teste**

### **4. Análise Completa:**
- ✅ **Resumo executivo** com interpretações
- ✅ **Detecção de overfitting**
- ✅ **Avaliação da qualidade do ajuste**
- ✅ **Comentários sobre gaussianidade**

---

## 💡 **COMO USAR:**

1. **Execute cada bloco** na ordem apresentada
2. **Analise os gráficos** gerados
3. **Leia os comentários** automáticos
4. **Use o resumo executivo** para sua análise

Este código fornece uma análise completa e profissional do modelo de regressão linear! 🚀 