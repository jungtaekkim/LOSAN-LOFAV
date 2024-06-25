import numpy as np

from losan_lofav import linear_models
from losan_lofav import utils_regression


def get_arms(bounds, num_arms, random_state):
    assert isinstance(bounds, np.ndarray)
    assert isinstance(num_arms, int)
    assert bounds.shape[1] == 2

    arms = random_state.uniform(low=0.0, high=1.0, size=(num_arms, bounds.shape[0]))
    arms = (bounds[:, 1] - bounds[:, 0]) * arms + bounds[:, 0]
    return arms

def get_arms_unitsphere(dim_problem, num_arms, S, random_state):
    assert isinstance(dim_problem, int)
    assert isinstance(num_arms, int)

    arms = random_state.multivariate_normal(np.zeros(dim_problem), np.eye(dim_problem), size=num_arms)
    arms /= np.linalg.norm(arms, ord=2, axis=1)[..., np.newaxis]
    arms *= S

    return arms

def generate_rademacher_noises(size, sigma, random_state):
    noises = random_state.uniform(low=0.0, high=1.0, size=size) >= 0.5
    noises = noises.astype(np.float64)
    noises = 2 * noises - 1
    noises *= sigma
    return noises

def generate_noises(size, sigma_zero, sigma_bar, str_noise, random_state):
    probs_noises = random_state.uniform(low=0.0, high=1.0, size=size)
    indices = probs_noises < 0.0

    if str_noise == 'gaussian':
        noises_sampled = random_state.normal(loc=0.0, scale=sigma_bar, size=size)
        noises_sampled[indices] = random_state.normal(loc=0.0, scale=sigma_zero, size=size)[indices]
    elif str_noise == 'rademacher':
        noises_sampled = generate_rademacher_noises(size, sigma_bar, random_state)
        noises_sampled[indices] = generate_rademacher_noises(size, sigma_zero, random_state)[indices]
    else:
        raise ValueError

    return noises_sampled

def evaluate_model_with_theta(
    X, theta,
    random_state=None, sigma_zero=None, sigma_bar=None, use_noise=False, str_noise=None
):
    assert X.shape[1] == theta.shape[0]

    by = linear_models.calculate_predictions(X, theta)

    if use_noise:
        assert random_state is not None
        assert sigma_zero is not None
        assert sigma_bar is not None
        assert str_noise is not None

        noises_sampled = generate_noises(by.shape, sigma_zero, sigma_bar, str_noise, random_state)
        by += noises_sampled

    return by

def initialize(dim_problem, num_arms, S, sigma_zero, sigma_bar, random_state, str_noise='gaussian'):
    assert isinstance(dim_problem, int)
    assert isinstance(num_arms, int)
    assert isinstance(S, float)
    assert isinstance(sigma_zero, float)
    assert isinstance(sigma_bar, float)
    assert isinstance(str_noise, str)
    assert str_noise in ['gaussian', 'rademacher']

    bounds = np.array([
        [0.0, 1.0],
    ] * dim_problem)

    unknown_theta = utils_regression.sample_unknown_theta(dim_problem, S, random_state)
    arms = get_arms_unitsphere(dim_problem, num_arms, S, random_state)
    arms_transformed = arms

    def fun_target_noisy(bx):
        assert bx.ndim == 1

        X = np.atleast_2d(bx)
        by = evaluate_model_with_theta(X, unknown_theta,
            random_state=random_state, sigma_zero=sigma_zero, sigma_bar=sigma_bar, use_noise=True, str_noise=str_noise)
        return by[0]

    def fun_target_noiseless(bx):
        assert bx.ndim == 1

        X = np.atleast_2d(bx)
        by = evaluate_model_with_theta(X, unknown_theta, use_noise=False)
        return by[0]

    return bounds, arms, arms_transformed, unknown_theta, fun_target_noisy, fun_target_noiseless

def get_initial_points(random_state, arms, arms_transformed, fun_target_noisy, fun_target_noiseless, num_init):
    num_arms = arms.shape[0]

    indices_initial = random_state.choice(num_arms, size=num_init, replace=False)
    X = []
    X_transformed = []
    by_noisy = []
    by_noiseless = []

    for ind_initial in indices_initial:
        arm_selected = arms[ind_initial]
        arm_transformed_selected = arms_transformed[ind_initial]

        X.append(arm_selected)
        X_transformed.append(arm_transformed_selected)
        by_noisy.append(fun_target_noisy(arm_selected))
        by_noiseless.append(fun_target_noiseless(arm_selected))

    return X, X_transformed, by_noisy, by_noiseless

def get_best_point(arms, arms_transformed, fun_target_noiseless):
    bx_best = None
    bx_transformed_best = None
    y_best = -np.inf

    for arm, arm_transformed in zip(arms, arms_transformed):
        y = fun_target_noiseless(arm)

        if y > y_best:
            bx_best = arm
            bx_transformed_best = arm_transformed
            y_best = y

    return bx_best, bx_transformed_best, y_best
