# =================================  TESTY  ===================================
# Testy do tego pliku zostały podzielone na dwie kategorie:
#
#  1. `..._invalid_input`:
#     - Sprawdzające poprawną obsługę nieprawidłowych danych wejściowych.
#
#  2. `..._correct_solution`:
#     - Weryfikujące poprawność wyników dla prawidłowych danych wejściowych.
# =============================================================================
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt



def spare_matrix_Abt(m: int, n: int) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(m, int) or not isinstance(n, int):
        return None
    if m <= 0 or n <= 0:
        return None

    t = np.linspace(0, 1, m)
    A = np.vander(t, N=n, increasing=True)
    
    b = np.cos(4 * t)

    return A, b


def square_from_rectan(
    A: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Funkcja przekształcająca układ równań z prostokątną macierzą współczynników
    na kwadratowy układ równań.
    A^T * A * x = A^T * b  ->  A_new * x = b_new

    Args:
        A (np.ndarray): Macierz A (m,n) zawierająca współczynniki równania.
        b (np.ndarray): Wektor b (m,) zawierający współczynniki po prawej stronie równania.

    Returns:
        (tuple[np.ndarray, np.ndarray]):
            - Macierz A_new o rozmiarze (n,n),
            - Wektor b_new (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(A, np.ndarray) or not isinstance(b, np.ndarray):
        return None

    # Walidacja wymiarów A i b
    if A.ndim != 2 or b.ndim != 1:
        return None

    m, n = A.shape
    if b.shape[0] != m:
        return None

    try:
        A_new = A.T @ A
        b_new = A.T @ b
    except Exception:
        return None

    return A_new, b_new


def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float | None:
    """Funkcja obliczająca normę residuum dla równania postaci:
    Ax = b

    Args:
        A (np.ndarray): Macierz A (m,n) zawierająca współczynniki równania.
        x (np.ndarray): Wektor x (n,) zawierający rozwiązania równania.
        b (np.ndarray): Wektor b (m,) zawierający współczynniki po prawej stronie równania.

    Returns:
        (float): Wartość normy residuum dla podanych parametrów.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(A, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(b, np.ndarray):
        return None

    # Walidacja wymiarów
    if A.ndim != 2 or x.ndim != 1 or b.ndim != 1:
        return None

    m, n = A.shape
    if x.shape[0] != n or b.shape[0] != m:
        return None

    try:
        r = A @ x - b
        return float(np.linalg.norm(r))
    except Exception:
        return None
    
