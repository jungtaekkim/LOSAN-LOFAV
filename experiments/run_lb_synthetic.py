import numpy as np
import os
import argparse
import time

from losan_lofav import utils_models
from losan_lofav import utils_optimization


path_results = '../results_synthetic'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--noise', type=str, required=True)
    parser.add_argument('--problem', type=int, required=True)
    parser.add_argument('--num_iter', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--num_arms', type=int, required=True)
    parser.add_argument('--norm', type=float, required=True)
    parser.add_argument('--sigma_bar', type=float, required=True)
    parser.add_argument('--save_results', action='store_true')

    args = parser.parse_args()

    str_model = args.model
    str_noise = args.noise
    dim_problem = args.problem
    num_iter = args.num_iter
    seed = args.seed
    num_arms = args.num_arms
    S = args.norm
    sigma_bar = args.sigma_bar
    save_results = args.save_results

    sigma_zero = 1.0
    num_init = 1
    delta = 0.2

    print('')
    print('=' * 40)
    print(f'str_model {str_model}')
    print(f'str_noise {str_noise}')
    print(f'num_init {num_init}')
    print(f'num_iter {num_iter}')
    print(f'sigma_zero {sigma_zero}')
    print(f'sigma_bar {sigma_bar}')
    print(f'dim_problem {dim_problem}')
    print(f'num_arms {num_arms}')
    print(f'S {S}')
    print(f'delta {delta}')
    print(f'seed {seed}')
    print('=' * 40)
    print('')

    assert num_init == 1

    random_state = np.random.RandomState(seed)

    _, arms, arms_transformed, _, fun_target_noisy, fun_target_noiseless = utils_optimization.initialize(
        dim_problem, num_arms, S, sigma_zero, sigma_bar, random_state,
        str_noise=str_noise,
    )
    assert arms.shape[0] == arms_transformed.shape[0] == num_arms

    X, X_transformed, by_noisy, by_noiseless, bx_best, bx_transformed_best, y_best = utils_models.run(str_model, S, sigma_zero, num_init, num_iter, delta, arms, arms_transformed, fun_target_noisy, fun_target_noiseless, random_state)

    assert X.shape[0] == X_transformed.shape[0] == by_noisy.shape[0] == by_noiseless.shape[0]
    assert X.shape[1] == bx_best.shape[0]
    assert X_transformed.shape[1] == bx_transformed_best.shape[0]
    print(X.shape, X_transformed.shape, by_noisy.shape, by_noiseless.shape)
    print(np.sum(y_best - by_noiseless))

    dict_all = {
        'str_model': str_model,
        'str_noise': str_noise,
        'dim_problem': dim_problem,
        'dim_transformed': X_transformed.shape[1],
        'num_arms': num_arms,
        'S': S,
        'sigma_zero': sigma_zero,
        'sigma_bar': sigma_bar,
        'num_init': num_init,
        'num_iter': num_iter,
        'delta': delta,
        'seed': seed,
        'X': X,
        'by_noisy': by_noisy,
        'by_noiseless': by_noiseless,
        'bx_best': bx_best,
        'bx_transformed_best': bx_transformed_best,
        'y_best': y_best,
    }

    if not os.path.exists(path_results):
        os.mkdir(path_results)

    str_file = f'lb_synthetic_{str_model}_{str_noise}_{dim_problem}_{num_arms}_{S}_{sigma_zero}_{sigma_bar}_{num_init}_{num_iter}_{delta}_{seed}.npy'

    if save_results:
        np.save(os.path.join(path_results, str_file), dict_all)
