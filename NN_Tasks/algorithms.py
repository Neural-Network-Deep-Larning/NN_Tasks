# algorithms.py
import numpy as np
from typing import Tuple, List

def initialize_weights(n_features: int, use_bias: bool = True, seed: int = 42, scale: float = 0.01):
    np.random.seed(seed)
    weights = np.random.normal(0.0, scale, n_features)
    bias = np.random.normal(0.0, scale) if use_bias else 0.0
    return weights, bias

def perceptron_train(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float,
                     eta: float = 0.01, epochs: int = 50, use_bias: bool = True):
    n_samples = X.shape[0]
    errors_history = []
    current_weights = weights.copy()
    current_bias = bias
    for epoch in range(epochs):
        errors = 0
        for i in range(n_samples):
            net_input = np.dot(X[i], current_weights) + (current_bias if use_bias else 0)
            prediction = 1 if net_input >= 0 else -1
            if prediction != y[i]:
                errors += 1
                update = eta * y[i]
                current_weights += update * X[i]
                if use_bias:
                    current_bias += update
        errors_history.append(errors)
        if errors == 0:
            break
    return current_weights, current_bias, errors_history

def adaline_train(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float,
                  eta: float = 0.01, epochs: int = 50, use_bias: bool = True, mse_threshold=None):
    n_samples = X.shape[0]
    mse_history = []
    current_weights = weights.copy()
    current_bias = bias
    for epoch in range(epochs):
        errors = []
        for i in range(n_samples):
            net_input = np.dot(X[i], current_weights) + (current_bias if use_bias else 0)
            error = y[i] - net_input
            errors.append(error)
            update = eta * error
            current_weights += update * X[i]
            if use_bias:
                current_bias += update
        mse = np.mean(np.array(errors) ** 2)
        mse_history.append(mse)
        if mse_threshold is not None and mse <= mse_threshold:
            break
    return current_weights, current_bias, mse_history

