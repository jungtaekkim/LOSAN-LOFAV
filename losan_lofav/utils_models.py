import numpy as np

from losan_lofav import linear_models
from losan_lofav import losan
from losan_lofav import lofav
from losan_lofav import utils_regression
from losan_lofav import utils_optimization


def run_losan(X_transformed, by_noisy, arms, arms_transformed, Sigmas_inverse, bw, thetas_hat, terms_second, S, sigma_zero, lambda_, delta):
    dim_X = X_transformed.shape[1]

    if Sigmas_inverse is None:
        Sigmas_inverse = np.array([utils_regression.get_inverse_with_cholesky(lambda_ * np.eye(dim_X))])
    if bw is None:
        bw = np.zeros((0, ))
    if thetas_hat is None:
        thetas_hat = np.array([np.zeros(dim_X)])
    if terms_second is None:
        terms_second = np.zeros((0, ))

    w_t = losan.calculate_w_t(X_transformed[-1], Sigmas_inverse[-1])
    bw = np.concatenate([bw, [w_t]], axis=0)

    Sigma_t = losan.calculate_Sigma_t(X_transformed, lambda_, bw)
    Sigma_t_inverse = utils_regression.get_inverse_with_cholesky(Sigma_t)
    Sigmas_inverse = np.concatenate([Sigmas_inverse, [Sigma_t_inverse]], axis=0)

    theta_hat_t = losan.calculate_theta_hat_t(X_transformed, by_noisy, bw, Sigma_t_inverse)
    D_t = losan.calculate_D_t(X_transformed[-1], bw[-1], Sigmas_inverse[-1])

    thetas_hat = np.concatenate([thetas_hat, [theta_hat_t]], axis=0)

    term_second = utils_regression.weighted_loss_l2(
        np.array([by_noisy[-1]]),
        linear_models.calculate_predictions(np.array([X_transformed[-1]]), thetas_hat[-2]),
        np.array([bw[-1]])
    ) * D_t**2
    terms_second = np.concatenate([terms_second, [term_second]], axis=0)
    assert X_transformed.shape[0] == by_noisy.shape[0] == bw.shape[0] == (thetas_hat.shape[0] - 1) == terms_second.shape[0] == (Sigmas_inverse.shape[0] - 1)

    vals = linear_models.calculate_predictions(arms_transformed, thetas_hat[-1])
    beta = losan.calculate_confidence_set_rhs(terms_second, lambda_, S, sigma_zero, delta)
    dists = np.sqrt(np.matmul(np.expand_dims(np.matmul(arms_transformed, Sigma_t_inverse), axis=1), np.expand_dims(arms_transformed, axis=2))[:, 0, 0])

    vals += np.sqrt(2 * beta) * dists
    ind_max = np.argmax(vals)

    arm_selected = arms[ind_max]
    arm_transformed_selected = arms_transformed[ind_max]

    return arm_selected, arm_transformed_selected, Sigmas_inverse, bw, thetas_hat, terms_second

def run_lofav(X_transformed, by_noisy, arms, arms_transformed, Sigmas_inverse, thetas_hat, bw, Fs, weighted_losses_without_Fs, betas, T, S, R, delta):
    dim_X = X_transformed.shape[1]
    L = lofav.calculate_L(T, dim_X)

    if Sigmas_inverse is not None:
        assert Sigmas_inverse.shape[0] == L
    if thetas_hat is not None:
        assert thetas_hat.shape[0] == L
    if bw is not None:
        assert bw.shape[0] == L
    if Fs is not None:
        assert Fs.shape[0] == L
    if weighted_losses_without_Fs is not None:
        assert weighted_losses_without_Fs.shape[0] == L
    if betas is not None:
        assert betas.shape[0] == L

    new_Sigmas_inverse = []
    new_thetas_hat = []
    new_bw = []
    new_Fs = []
    new_weighted_losses_without_Fs_l = []
    new_betas = []

    vals_first = []
    vals_second = []

    for l in range(0, L):
        rho_l = lofav.calculate_rho_l(l + 1)
        lambda_l = utils_regression.get_lambda(rho_l, S / R)

        if Sigmas_inverse is None:
            Sigmas_l_inverse = np.array([utils_regression.get_inverse_with_cholesky(lambda_l * np.eye(dim_X))])
        else:
            Sigmas_l_inverse = Sigmas_inverse[l]

        if thetas_hat is None:
            thetas_hat_l = np.zeros((1, dim_X))
        else:
            thetas_hat_l = thetas_hat[l]

        if bw is None:
            bw_l = np.zeros((0, ))
        else:
            bw_l = bw[l]

        if Fs is None:
            Fs_l = np.zeros((0, ))
        else:
            Fs_l = Fs[l]

        if weighted_losses_without_Fs is None:
            weighted_losses_without_Fs_l = np.zeros((0, ))
        else:
            weighted_losses_without_Fs_l = weighted_losses_without_Fs[l]

        beta_zero_l = 1 / 2 * lambda_l * S**2
        if betas is None:
            betas_l = np.array([beta_zero_l])
        else:
            betas_l = betas[l]

        w_tl = lofav.calculate_w_tl(X_transformed[-1], Sigmas_l_inverse[-1], rho_l)
        bw_l = np.concatenate([bw_l, [w_tl]], axis=0)

        Sigma_tl = lofav.calculate_Sigma_tl(X_transformed, bw_l, lambda_l)
        Sigma_tl_inverse = utils_regression.get_inverse_with_cholesky(Sigma_tl)
        Sigmas_l_inverse = np.concatenate([Sigmas_l_inverse, [Sigma_tl_inverse]], axis=0)

        theta_hat_tl = lofav.calculate_theta_hat_tl(X_transformed, by_noisy, bw_l, Sigma_tl_inverse)
        thetas_hat_l = np.concatenate([thetas_hat_l, [theta_hat_tl]], axis=0)

        F_tl = lofav.calculate_F_tl(X_transformed[-1], bw_l[-1], Sigmas_l_inverse[-1])
        Fs_l = np.concatenate([Fs_l, [F_tl]], axis=0)

        weighted_loss_with_theta_hat_tl = utils_regression.weighted_loss_l2(
            np.array([by_noisy[-1]]),
            linear_models.calculate_predictions(np.array([X_transformed[-1]]), thetas_hat_l[-2]),
            np.array([bw_l[-1]])
        )
        weighted_losses_without_Fs_l = np.concatenate([weighted_losses_without_Fs_l, [weighted_loss_with_theta_hat_tl]], axis=0)
        weighted_loss_with_Fs_l = np.matmul(weighted_losses_without_Fs_l, Fs_l**2)

        beta_max = lofav.calculate_beta_max(betas_l)

        theta_bar_tl, Q_tl_inverse, beta_tl = lofav.calculate_confidence_set_rhs(
            X_transformed,
            by_noisy,
            thetas_hat_l,
            bw_l,
            weighted_losses_without_Fs_l,
            weighted_loss_with_Fs_l,
            beta_max,
            beta_zero_l,
            S,
            L,
            rho_l,
            lambda_l,
            R,
            delta
        )
        betas_l = np.concatenate([betas_l, [beta_tl]], axis=0)

        new_Sigmas_inverse.append(Sigmas_l_inverse)
        new_thetas_hat.append(thetas_hat_l)
        new_bw.append(bw_l)
        new_Fs.append(Fs_l)
        new_weighted_losses_without_Fs_l.append(weighted_losses_without_Fs_l)
        new_betas.append(betas_l)

        ##
        vals_l_first = linear_models.calculate_predictions(arms_transformed, theta_bar_tl)
        dists_l_first = np.sqrt(np.matmul(np.expand_dims(np.matmul(arms_transformed, Q_tl_inverse), axis=1), np.expand_dims(arms_transformed, axis=2))[:, 0, 0])
        assert np.all(vals_l_first.shape == dists_l_first.shape)
        vals_first.append(vals_l_first + np.sqrt(2 * beta_tl) * dists_l_first)

        gamma_tl = lofav.calculate_confidence_set_rhs_extra(weighted_loss_with_Fs_l, S, L, lambda_l, R, delta)

        vals_l_second = linear_models.calculate_predictions(arms_transformed, theta_hat_tl)
        dists_l_second = np.sqrt(np.matmul(np.expand_dims(np.matmul(arms_transformed, Sigma_tl_inverse), axis=1), np.expand_dims(arms_transformed, axis=2))[:, 0, 0])
        assert np.all(vals_l_second.shape == dists_l_second.shape)
        vals_second.append(vals_l_second + np.sqrt(2 * gamma_tl) * dists_l_second)
        ##

    vals = np.min(np.concatenate([
        np.min(vals_first, axis=0)[np.newaxis, ...],
        np.min(vals_second, axis=0)[np.newaxis, ...]
    ], axis=0), axis=0)
    assert vals.ndim == 1
    assert vals.shape[0] == arms_transformed.shape[0]
    ind_max = np.argmax(vals)

    if np.min(vals_first, axis=0)[ind_max] < np.min(vals_second, axis=0)[ind_max]:
        print('the first is selected.', flush=True)

    arm_selected = arms[ind_max]
    arm_transformed_selected = arms_transformed[ind_max]

    return arm_selected, arm_transformed_selected, np.array(new_Sigmas_inverse), np.array(new_thetas_hat), np.array(new_bw), np.array(new_Fs), np.array(new_weighted_losses_without_Fs_l), np.array(new_betas)

def run(
    str_model,
    S,
    sigma_zero,
    num_init,
    num_iter,
    delta,
    arms,
    arms_transformed,
    fun_target_noisy,
    fun_target_noiseless,
    random_state,
):
    X, X_transformed, by_noisy, by_noiseless = utils_optimization.get_initial_points(random_state, arms, arms_transformed, fun_target_noisy, fun_target_noiseless, num_init)

    if str_model == 'losan':
        lambda_ = utils_regression.get_lambda(sigma_zero, S)
        Sigmas_inverse = None
        bw = None
        thetas_hat = None
        terms_second = None
    elif str_model == 'lofav':
        Sigmas_inverse = None
        thetas_hat = None
        bw = None
        Fs = None
        weighted_losses_without_Fs = None
        betas = None
        T = num_iter
    else:
        raise ValueError

    for ind_iter in range(0, num_iter):
        print(f'Iteration {ind_iter + 1}', flush=True)

        if str_model == 'losan':
            arm_selected, arm_transformed_selected, Sigmas_inverse, bw, thetas_hat, terms_second = run_losan(np.array(X_transformed), np.array(by_noisy), arms, arms_transformed, Sigmas_inverse, bw, thetas_hat, terms_second, S, sigma_zero, lambda_, delta)
        elif str_model == 'lofav':
            arm_selected, arm_transformed_selected, Sigmas_inverse, thetas_hat, bw, Fs, weighted_losses_without_Fs, betas = run_lofav(np.array(X_transformed), np.array(by_noisy), arms, arms_transformed, Sigmas_inverse, thetas_hat, bw, Fs, weighted_losses_without_Fs, betas, T, S, sigma_zero, delta)
        else:
            raise ValueError

        X.append(arm_selected)
        X_transformed.append(arm_transformed_selected)
        by_noisy.append(fun_target_noisy(arm_selected))
        by_noiseless.append(fun_target_noiseless(arm_selected))

    X = np.array(X)
    X_transformed = np.array(X_transformed)
    by_noisy = np.array(by_noisy)
    by_noiseless = np.array(by_noiseless)

    bx_best, bx_transformed_best, y_best = utils_optimization.get_best_point(arms, arms_transformed, fun_target_noiseless)

    return X, X_transformed, by_noisy, by_noiseless, bx_best, bx_transformed_best, y_best
