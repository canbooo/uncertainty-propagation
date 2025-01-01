import numpy as np


def linear(x: np.ndarray, beta: float = 5.0) -> np.ndarray:
    """safety index = beta"""
    return beta * np.sqrt(x.shape[1]) - x.sum(1)


def quadratic_greater(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2, axis=1) + 1.0


def quadratic_lesser(x: np.ndarray) -> np.ndarray:
    return -1.0 * quadratic_greater(x)


def styblinski_tang(x: np.ndarray) -> np.ndarray:
    """safety index ~"""
    return np.sum(x**4 - 16 * x**2 + 5 * x, axis=1) + 100 * np.sqrt(x.shape[1])


def modified_himmblau(x: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    res = ((x[:, 0] ** 2 + x[:, 1]) / 1.81 - 11) ** 2
    res += ((x[:, 0] + x[:, 1] ** 2) / 1.81 - 7) ** 2
    return res - 50 * gamma


def modified_rastrigin(x: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    return 10.0 - np.sum((x / gamma) ** 2 - 5 * np.cos(2 * np.pi * x / gamma), axis=1)
