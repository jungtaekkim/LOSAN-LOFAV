import numpy as np
import scipy.linalg as scila


def get_inverse_with_cholesky(X):
    assert X.shape[0] == X.shape[1]

    lower = scila.cholesky(X, lower=True)
    return scila.cho_solve((lower, True), np.eye(X.shape[0]))

def get_lambda(sigma, S):
    return sigma**2 / S**2

def sample_unknown_theta(dim_theta, S, random_state):
    theta = random_state.multivariate_normal(np.zeros(dim_theta), np.eye(dim_theta))
    theta /= scila.norm(theta, ord=2)
    theta *= S
    return theta

def loss_l2(by, preds):
    assert by.shape[0] == preds.shape[0]

    return 1 / 2 * np.sum((by - preds)**2)

def weighted_loss_l2(by, preds, bw):
    assert by.ndim == preds.ndim == bw.ndim == 1
    assert by.shape[0] == preds.shape[0] == bw.shape[0]

    return np.matmul(bw**2, 1 / 2 * (preds - by)**2)
