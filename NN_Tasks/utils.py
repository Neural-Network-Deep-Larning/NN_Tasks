# utils.py
import numpy as np

def normalize_train_test(X_train: np.ndarray, X_test: np.ndarray):
    mean_vec = np.mean(X_train, axis=0)
    std_vec = np.std(X_train, axis=0)
    std_vec[std_vec == 0] = 1
    X_train_std = (X_train - mean_vec) / std_vec
    X_test_std = (X_test - mean_vec) / std_vec
    return X_train_std, X_test_std, mean_vec, std_vec

def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1):
    TP = int(np.sum((y_true == pos_label) & (y_pred == pos_label)))
    TN = int(np.sum((y_true != pos_label) & (y_pred != pos_label)))
    FP = int(np.sum((y_true != pos_label) & (y_pred == pos_label)))
    FN = int(np.sum((y_true == pos_label) & (y_pred != pos_label)))
    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN}
