import numpy as np
import numpy.linalg as npla
import scipy.spatial.distance as scisd

from losan_lofav import linear_models
from losan_lofav import utils_regression


def calculate_L(T, dim):
    L = np.ceil(1 / 2 * np.log2(T / dim)).astype(np.int64)
    L = np.maximum(1, L)
    return L

def calculate_rho_l(l):
    return 2**(-l)

def calculate_w_tl(bx, Sigma_prev_inverse, rho_l):
    w_tl = rho_l / scisd.mahalanobis(bx, np.zeros_like(bx), Sigma_prev_inverse)
    w_tl = np.minimum(1.0, w_tl)
    return w_tl

def calculate_Sigma_tl(X, bw_l, lambda_l, alpha=None):
    assert X.shape[0] == bw_l.shape[0]
    assert isinstance(lambda_l, float)
    assert isinstance(alpha, (type(None), float))

    Sigma_tl = lambda_l * np.eye(X.shape[1])
    weighted_X_T_X = np.matmul(X.T * bw_l**2, X)
    if alpha is not None:
        assert alpha == 2.0 # now, it is an only case.
        weighted_X_T_X *= alpha
    Sigma_tl += weighted_X_T_X
    return Sigma_tl

def calculate_loss_with_lambda(X, by, theta, bw_l, lambda_l):
    weighted_loss = utils_regression.weighted_loss_l2(by, linear_models.calculate_predictions(X, theta), bw_l)

    return weighted_loss + lambda_l / 2 * npla.norm(theta, ord=2)**2

def calculate_theta_hat_tl(X, by, bw_l, Sigma_inverse):
    assert X.ndim == 2
    assert by.ndim == 1
    assert bw_l.ndim == 1
    assert X.shape[0] == by.shape[0] == bw_l.shape[0]
    assert X.shape[1] == Sigma_inverse.shape[0] == Sigma_inverse.shape[1]

    theta_hat_tl = np.matmul(np.matmul(Sigma_inverse, X.T * bw_l**2), by)
    return theta_hat_tl

def calculate_K_tl(X, by, theta, thetas_hat_l, bw_l, lambda_l):
    term_first = utils_regression.weighted_loss_l2(by, linear_models.calculate_predictions(X, theta), bw_l)
    term_second = np.matmul(bw_l**2 / 2, (np.matmul(X, theta) - np.matmul(np.expand_dims(X, 1), np.expand_dims(thetas_hat_l[:-1], 2))[:, 0, 0])**2)
    term_third = lambda_l / 2 * npla.norm(theta, ord=2)**2

    return term_first + term_second + term_third

def calculate_theta_bar_tl(X, by, thetas_hat_l, bw_l, lambda_l):
    assert X.ndim == 2
    assert by.ndim == 1
    assert thetas_hat_l.ndim == 2
    assert bw_l.ndim == 1
    assert X.shape[0] == by.shape[0]
    assert (thetas_hat_l.shape[0] - 1) == bw_l.shape[0]
    assert X.shape[0] == (thetas_hat_l.shape[0] - 1)

    Q_tl = calculate_Sigma_tl(X, bw_l, lambda_l, alpha=2.0)
    Q_tl_inverse = utils_regression.get_inverse_with_cholesky(Q_tl)

    theta_bar_tl = np.matmul(np.matmul(Q_tl_inverse, X.T * bw_l**2), by) + np.matmul(np.matmul(Q_tl_inverse, X.T * bw_l**2), np.matmul(np.expand_dims(X, 1), np.expand_dims(thetas_hat_l[:-1], 2))[:, 0, 0])

    return theta_bar_tl, Q_tl_inverse

def calculate_F_tl(bx, w, Sigma_inverse):
    return scisd.mahalanobis(w * bx, np.zeros_like(bx), Sigma_inverse)

def calculate_xi_tl(t, k_tl, L, delta):
    xi_tl = np.log(np.sqrt(np.pi * (t + 1)) * (6.8 * L * k_tl * np.log(1 + k_tl)**2) / delta)
    return xi_tl

def calculate_beta_max(betas):
    return np.max(betas)

def calculate_k_tl(beta_max, beta_zero):
    k_tl = np.ceil(np.log2(np.sqrt(beta_max / beta_zero)))
    k_tl = np.maximum(1.0, k_tl)
    return k_tl

def calculate_confidence_set_rhs(X, by, thetas_hat_l, bw_l, weighted_losses_without_Fs_l, weighted_loss_with_Fs_l, beta_max, beta_zero_l, S, L, rho_l, lambda_l, R, delta):
    assert X.ndim == 2
    assert by.ndim == 1
    assert thetas_hat_l.ndim == 2
    assert bw_l.ndim == 1
    assert weighted_losses_without_Fs_l.ndim == 1
    assert X.shape[0] == by.shape[0]
    assert X.shape[0] == (thetas_hat_l.shape[0] - 1)
    assert X.shape[0] == bw_l.shape[0]
    assert X.shape[0] == weighted_losses_without_Fs_l.shape[0]
    assert X.shape[1] == thetas_hat_l.shape[1]

    theta_hat_tl = thetas_hat_l[-1]

    L_tl_with_theta_hat_tl = calculate_loss_with_lambda(X, by, theta_hat_tl, bw_l, lambda_l)

    theta_bar_tl, Q_tl_inverse = calculate_theta_bar_tl(X, by, thetas_hat_l, bw_l, lambda_l)
    K_tl_with_theta_bar_tl = calculate_K_tl(X, by, theta_bar_tl, thetas_hat_l, bw_l, lambda_l)

    k_tl = calculate_k_tl(beta_max, beta_zero_l)
    xi_tl = calculate_xi_tl(X.shape[0], k_tl, L, delta)

    term_fourth = weighted_loss_with_Fs_l
    term_fifth = np.sqrt(8 * rho_l**2 * beta_max * (np.sum(weighted_losses_without_Fs_l) + R**2 * np.log(2 * L / delta)) * xi_tl)
    term_sixth = 2**k_tl * rho_l * R * np.sqrt(2 * beta_zero_l) * xi_tl

    beta_tl = L_tl_with_theta_hat_tl - K_tl_with_theta_bar_tl + lambda_l / 2 * S**2 + term_fourth + term_fifth + term_sixth

    return theta_bar_tl, Q_tl_inverse, beta_tl

def calculate_confidence_set_rhs_extra(term_second, S, L, lambda_l, R, delta):
    term_first = lambda_l / 2 * S**2
    term_third = R**2 * np.log(2 * L / delta)
    return term_first + term_second + term_third
