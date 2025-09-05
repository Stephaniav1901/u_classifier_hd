import numpy as np
from typing import List, Union

def calculate_U_statistic(X: np.ndarray) -> float:
    """
    Calcula a U-estatística de uma amostra.
    A U-estatística é a média de todas as formas bilineares de pares distintos
    de vetores da amostra.

    Args:
        X (np.array): Uma matriz de dados de formato (n, p), onde 'n' é o número
                      de amostras e 'p' é o número de características.

    Returns:
        float: O valor calculado da U-estatística.
    """
    # Extrai as dimensões da matriz de dados: n = número de observações, p = número de características.
    n, p = X.shape

    if n < 2:
        raise ValueError("Sample size must be at least 2 for U-statistic")

    # Q(ni) = ni(ni-1) é o número total de pares distintos que podem ser formados a partir de n observações.    
    Q_n = n * (n - 1)
    
    # --- Cálculo Eficiente do Somatório Σ_{k≠r} x_k' * x_r ---
    # Um loop duplo seria muito lento para dados de alta dimensão.
    # Então fazemos: (Σx)² = Σ(x²) + Σ(x_k' * x_r) para k != r
    # Chegando a (Σx)² - Σ(x²) = Σ_{k≠r} x_k' * x_r

    # Soma todos os vetores de observação: (Σx)
    sum_of_vectors = np.sum(X, axis=0)
    # Soma todos os elementos ao quadrado da matriz, equivale a Σ(x²)
    sum_of_squares = np.sum(X**2)
    # Calcula o produto escalar do vetor soma por ele mesmo, o que equivale a (Σx)²
    total_dot_product = np.dot(sum_of_vectors, sum_of_vectors)
    # (Σx)² - Σ(x²)
    sum_A_ikr = total_dot_product - sum_of_squares

    # Garante que não haverá divisão por zero se p ou Q_n forem zero.
    if p == 0 or Q_n == 0:
        return 0.0
    
    # Calcula a U-estatística final.
    # U_n = Σ_{k≠r}(x_k' * x_r)/(p * Q_n)
    U_n = sum_A_ikr / (p * Q_n)
    
    return U_n

def u_classifier_multisample(x0: np.ndarray, list_of_X: List[np.ndarray], return_scores: bool = False) -> Union[int, np.ndarray]:
    """
    Classifica uma nova observação em uma de 'g' populações (g >= 2).
    Implementa a regra da Seção 3 do artigo.

    A regra é atribuir x à população 'i' que maximiza a função discriminante Aᵢ(x).
    Aᵢ(x) = x'x̄ᵢ / p - Uₙᵢ / 2

    Args:
        x_new (np.array): A nova observação a ser classificada, vetor (p,).
        list_of_X (list): Uma lista de arrays numpy [X1, X2, ..., Xg].

    Returns:
        any: O rótulo da população com o maior escore Aᵢ(x).
    """
    scores = []
    p = x0.shape[0]

    # Calcula o escore Aᵢ(x) para cada população.
    for Xi in list_of_X:
        # x̄_i = Calcula o vetor média da população i
        x_bar_i = np.mean(Xi, axis=0)
        # Calcula a U-estatística para a população i
        U_ni = calculate_U_statistic(Xi)
        
        # Calcula a equação A_0i(x0) para a população i
        # A regra é classificar x0 na população 'k' que maximiza A_0k(x0)
        score_i = (np.dot(x0, x_bar_i) / p) - (U_ni / 2)
        scores.append(score_i)
    
    # A regra de classificação é atribuir x0 à população com a maior pontuação
    classification = np.argmax(scores)

    if return_scores:
        return np.array(scores)
    else:
        return classification