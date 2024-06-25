import numpy as np


def calculate_Sigma(X, lambda_):
    # X: (n, d)

    Sigma = np.matmul(X.T, X)
    assert Sigma.shape[0] == X.shape[1]
    assert Sigma.shape[1] == X.shape[1]

    Sigma += lambda_ * np.eye(X.shape[1])

    return Sigma

def calculate_theta_hat(X, by, Sigma_inverse):
    # X: (n, d)
    # by: (n, )

    theta = np.matmul(np.matmul(Sigma_inverse, X.T), by)
    return theta

def calculate_predictions(X, theta):
    # X: (n, d)
    # theta: (d, )
    assert len(X.shape) == 2
    assert len(theta.shape) == 1

    return np.matmul(X, theta)
