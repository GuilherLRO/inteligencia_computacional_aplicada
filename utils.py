import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np
from scipy.stats import norm

def plot_regression_diagnostics(y_train, y_train_pred, y_test, y_test_pred, title):
    """
    Plota o histograma dos resíduos do treino com curva normal ajustada,
    gráficos de dispersão Real vs Predito para treino e teste,
    exibe o coeficiente de correlação de Pearson entre os valores reais e preditos,
    e mostra o QQ plot dos resíduos do treino.

    Parâmetros:
    - y_train: valores reais do treino
    - y_train_pred: valores preditos do treino
    - y_test: valores reais do teste
    - y_test_pred: valores preditos do teste
    """
    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr, norm, probplot

    # Coeficiente de correlação de Pearson
    corr_train, _ = pearsonr(y_train, y_train_pred)
    corr_test, _ = pearsonr(y_test, y_test_pred)

    r2_train = float(r2_score(y_train, y_train_pred))
    r2_test = float(r2_score(y_test, y_test_pred))

    print(f"Coeficiente de correlação de Pearson (Treino): {corr_train:.4f}")
    print(f"R² (Treino): {r2_train:.4f}")
    print(f"Coeficiente de correlação de Pearson (Teste): {corr_test:.4f}")
    print(f"R² (Teste): {r2_test:.4f}")

    residuos_treino = y_train - y_train_pred

    # 2x2 grid de subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Diagnóstico do Modelo - ' + title, fontsize=18, y=0.98)

    # 1. Histograma dos resíduos + curva normal (Treinamento)
    ax = axes[0, 0]
    ax.hist(residuos_treino, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    mu, sigma = norm.fit(residuos_treino)
    x = np.linspace(residuos_treino.min(), residuos_treino.max(), 100)
    y_norm = norm.pdf(x, mu, sigma)
    ax.plot(x, y_norm, 'r-', linewidth=2, label=f'Normal (μ={mu:.2f}, σ={sigma:.2f})')
    ax.set_xlabel('Resíduos (Valor Real - Valor Predito)')
    ax.set_ylabel('Densidade')
    ax.set_title('Histograma dos Resíduos (Treinamento)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. QQ plot dos resíduos do treino
    ax = axes[0, 1]
    probplot(residuos_treino, dist="norm", plot=ax)
    ax.set_title('QQ Plot dos Resíduos (Treinamento)')
    ax.grid(True, alpha=0.3)

    # 3. Dispersão: Valor Real vs Valor Predito (Treino)
    ax = axes[1, 0]
    ax.scatter(y_train, y_train_pred, alpha=0.6, color='blue', s=50)
    ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2, label='Linha de Perfeita Predição')
    ax.set_xlabel('Valor Real (Preço)')
    ax.set_ylabel('Valor Predito (Preço)')
    ax.set_title('TREINO: Real vs Predito')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.text(0.05, 0.95, f'R² = {r2_train:.4f}\nCorr = {corr_train:.4f}', transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            fontsize=12, verticalalignment='top')

    # 4. Dispersão: Valor Real vs Valor Predito (Teste)
    ax = axes[1, 1]
    ax.scatter(y_test, y_test_pred, alpha=0.6, color='green', s=50)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Linha de Perfeita Predição')
    ax.set_xlabel('Valor Real (Preço)')
    ax.set_ylabel('Valor Predito (Preço)')
    ax.set_title('TESTE: Real vs Predito')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.text(0.05, 0.95, f'R² = {r2_test:.4f}\nCorr = {corr_test:.4f}', transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            fontsize=12, verticalalignment='top')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


    return corr_train, corr_test, r2_train, r2_test