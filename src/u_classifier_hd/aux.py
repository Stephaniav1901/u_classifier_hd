import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.stats import norm, multivariate_normal, probplot
from typing import List, Dict, Any, Tuple

from .core import u_classifier_multisample

def create_ar1_covariance(p: int, rho: float) -> np.ndarray:
    """
    Cria uma matriz de covariância com estrutura autorregressiva de ordem 1 (AR(1)).
    A covariância entre as variáveis i e j é rho**|i-j|.
    
    Args:
        p (int): A dimensionalidade da matriz.
        rho (float): O coeficiente de correlação.

    Returns:
        np.ndarray: A matriz de covariância (p, p).
    """
    # Usa a função toeplitz para criar a matriz AR(1).
    return toeplitz(rho ** np.arange(p))

def create_unstructured_covariance(p: int) -> np.ndarray:
    """
    Cria a matriz de covariância não estruturada (UN) descrita na Seção 4.
    A diagonal é 1 e os elementos fora da diagonal são rho_ij = (i-1)/p.
    
    Args:
        p (int): A dimensionalidade da matriz.

    Returns:
        np.ndarray: A matriz de covariância (p, p).
    """
    cov_matrix = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            if i == j:
                cov_matrix[i, j] = 1.0
            else:
                val = max(i, j) / p 
                cov_matrix[i, j] = val
                cov_matrix[j, i] = val
    return cov_matrix

def generate_mean_vectors(p: int) -> tuple[np.ndarray, np.ndarray]:
    """Gera os vetores de médias mu1 e mu2."""
    mu1 = np.zeros(p)
    
    # Para mu2, floor(p/3) elementos são 0, o resto é 1.
    mu2 = np.zeros(p)
    num_zeros = int(np.floor(p / 3))
    mu2[num_zeros:] = 1.0
    
    return mu1, mu2

def generate_data(n: int, mu: np.ndarray, cov: np.ndarray, dist: str = 'normal', df: int = 10) -> np.ndarray:
    """
    Gera dados de uma distribuição Normal Multivariada ou t de Student.

    Args:
        n (int): Número de amostras a serem geradas.
        mu (np.ndarray): Vetor de médias (1D com p elementos).
        cov (np.ndarray): Matriz de covariância (p, p).
        dist (str): A distribuição a ser usada ('normal' ou 't').
        df (int): Graus de liberdade para a distribuição t.

    Returns:
        np.ndarray: A matriz de dados gerada (n, p).
    """
    if dist == 'normal':
        # Gera dados da Normal Multivariada.
        return np.random.multivariate_normal(mu, cov, n)
    elif dist == 't-student':
        # Gera dados da distribuição t de Student.
        y = np.random.chisquare(df, n) / df
        z = np.random.multivariate_normal(np.zeros(p), cov, n)
        return mu + z / np.sqrt(y[:, np.newaxis])
    else:
        raise ValueError("Distribuição deve ser 'normal' ou 't-student'.")
    
def calculate_theoretical_errors(mu1: np.ndarray, mu2: np.ndarray, Sigma1: np.ndarray, Sigma2: np.ndarray, n1: int, n2: int) -> tuple[float, float]:
    """
    Calcula a probabilidade teórica de classificação incorreta com base no Teorema 3.2 do artigo.

    Args:
        mu1, mu2 (np.ndarray): Vetores de média das populações 1 e 2.
        Sigma1, Sigma2 (np.ndarray): Matrizes de covariância das populações 1 e 2.
        n1, n2 (int): Tamanhos das amostras das populações 1 e 2.
    Returns:
        float: A taxa de erro teórica média.
    """
    p=len(mu1)
    # 1. Calcular o vetor de diferença de médias
    # E[A0(x)|pi_2] = 1/p * (mu1-mu2)'mu2 - 1/2 * (mu1'mu1 - mu2'mu2) = -1/(2p) * ||mu1 - mu2||^2
    mean_diff = mu1 - mu2
    # Média teórica de A0(x)
    expected_value_pi1 = 0.5 * np.dot(mean_diff, mean_diff) / p
    expected_value_pi2 = -expected_value_pi1
    
    # 2. Calcular o traço dos quadrados das matrizes de covariância
    tr_s1_sq = np.trace(np.dot(Sigma1, Sigma1))
    tr_s2_sq = np.trace(np.dot(Sigma2, Sigma2))
    tr_s1_s2_sq = np.trace(np.dot(Sigma1, Sigma2))
    
    sum_term = (tr_s1_sq / (2 * n1 * (n1 - 1))) + (tr_s2_sq / (2 * n2 * (n2 - 1)))
    # 3. Calcular os termos de covariancias
    delta1_sq = (tr_s1_sq / n1) + (tr_s1_s2_sq / n2) + sum_term
    delta2_sq = (tr_s2_sq / n2) + (tr_s1_s2_sq / n1) + sum_term
    
    variance_term1 = np.dot(mean_diff, np.dot(Sigma1, mean_diff))
    variance_term2 = np.dot(mean_diff, np.dot(Sigma2, mean_diff))

    # Variâncias teóricas
    variance1 = (variance_term1 + delta1_sq) / (p**2)
    variance2 = (variance_term2 + delta2_sq) / (p**2)
    
    std_dev1 = np.sqrt(variance1)
    std_dev2 = np.sqrt(variance2)
    
    # Taxas de erro
    # e(2|1) = P(A0 <= 0 | pi_1) = Phi(-E1 / V1)
    error_21 = norm.cdf(-expected_value_pi1 / std_dev1)
    
    # e(1|2) = P(A0 > 0 | pi_2) = 1 - Phi(-E2 / V2) = Phi(E2 / V2)
    error_12 = norm.cdf(expected_value_pi2 / std_dev2)
    
    # Taxa de erro média
    avg_error = 0.5 * (error_21 + error_12)

    return error_21, error_12

def run_simulation_iteration(params: Dict[str, Any], n_test: int = 1000) -> Tuple[float, float]:
    """
    Executa uma iteração e calcula as taxas de erro e(1|0) e e(0|1).
    """
    p = params['p']
    
    mu1 = np.zeros(p)
    mu2 = np.ones(p)
    mu2[:p // 3] = 0

    if params['cov_case'] == 'AR-AR':
        cov1 = create_ar1_covariance(p, 0.3)
        cov2 = create_ar1_covariance(p, 0.7)
    else: # 'AR-UN'
        cov1 = create_ar1_covariance(p, 0.5)
        cov2 = create_unstructured_covariance(p)

    X1_train = generate_data(params['n1'], mu1, cov1, params['dist'])
    X2_train = generate_data(params['n2'], mu2, cov2, params['dist'])
    
    X0_test = generate_data(n_test, mu1, cov1, params['dist']) # Verdadeiro é 0
    X1_test = generate_data(n_test, mu2, cov2, params['dist']) # Verdadeiro é 1
    
    misclassified_1_given_0 = 0
    for x in X0_test:
        if u_classifier_multisample(x, [X1_train, X2_train], return_scores=False) != 0:
            misclassified_1_given_0 += 1
            
    misclassified_0_given_1 = 0
    for x in X1_test:
        if u_classifier_multisample(x, [X1_train, X2_train], return_scores=False) != 1:
            misclassified_0_given_1 += 1
            
    error_1_given_0 = misclassified_1_given_0 / n_test
    error_0_given_1 = misclassified_0_given_1 / n_test
    
    return error_1_given_0, error_0_given_1

def plot_results(results: Dict, title: str, ylabel: str,theoretical_results: Dict = None):
    """Plota os resultados da simulação."""
    plt.figure(figsize=(10, 6))
    
    for label, data in results.items():
        p_values = sorted(data.keys())
        error_rates = [data[p] for p in p_values]
        plt.plot(p_values, error_rates, marker='', linestyle='--', label=label)

    if theoretical_results:
        p_values = sorted(list(theoretical_results.keys()))
        error_rates = [theoretical_results[p] for p in p_values]
        plt.plot(p_values, error_rates, color='black', marker='', linestyle='-', label='Teórico')

    plt.xlabel("Dimensionalidade (p)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.ylim(bottom=-0.015, top=0.35)
    if p_values:
        plt.xlim(left=0, right=max(p_values))
    plt.show()

def plot_qq(scores: np.ndarray, ax: plt.Axes, title: str):
    """Gera um QQ-plot em um eixo específico."""
    probplot(scores, dist="norm", plot=ax)
    ax.set_title(title)
    ax.get_lines()[0].set_markerfacecolor('none')
    ax.get_lines()[0].set_markeredgecolor('black')
    ax.get_lines()[0].set_markersize(4)
    ax.get_lines()[1].set_color('black')
    ax.set_xlabel("Quantis Teóricos")
    ax.set_ylabel("Quantis da Amostra")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)