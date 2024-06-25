import numpy as np
import numpy.linalg as npla
import scipy.spatial.distance as scisd

from losan_lofav import linear_models
from losan_lofav import utils_regression


def calculate_D_t(bx, w, Sigma_inverse):
    assert bx.ndim == 1
    assert Sigma_inverse.ndim == 2
    assert bx.shape[0] == Sigma_inverse.shape[0] == Sigma_inverse.shape[1]

    D_t = scisd.mahalanobis(w * bx, np.zeros_like(bx), Sigma_inverse)
    return D_t

def calculate_w_t(bx, Sigma_prev_inverse):
    assert bx.ndim == 1
    assert Sigma_prev_inverse.ndim == 2
    assert bx.shape[0] == Sigma_prev_inverse.shape[0] == Sigma_prev_inverse.shape[1]

    w_t = 1 / scisd.mahalanobis(bx, np.zeros_like(bx), Sigma_prev_inverse)
    w_t = np.minimum(1.0, w_t)
    return w_t

def calculate_Sigma_t(X, lambda_, bw):
    assert X.ndim == 2
    assert bw.ndim == 1
    assert X.shape[0] == bw.shape[0]

    Sigma_t = lambda_ * np.eye(X.shape[1]) + np.matmul(X.T * bw**2, X)
    return Sigma_t

def calculate_theta_hat_t(X, by, bw, Sigma_inverse):
    assert X.ndim == 2
    assert by.ndim == 1
    assert bw.ndim == 1
    assert X.shape[0] == by.shape[0] == bw.shape[0]
    assert X.shape[1] == Sigma_inverse.shape[0] == Sigma_inverse.shape[1]

    theta_hat_t = np.matmul(np.matmul(Sigma_inverse, X.T * bw**2), by)
    return theta_hat_t

def calculate_confidence_set_rhs(terms_second, lambda_, S, sigma_zero, delta):
    rhs = lambda_ / 2 * S**2 + sigma_zero**2 * np.log(1 / delta)

    term_second = np.sum(terms_second)
    rhs += term_second
    return rhs
