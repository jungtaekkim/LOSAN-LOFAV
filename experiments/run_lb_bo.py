import numpy as np
import os
import argparse
import bayeso_benchmarks.utils as bb_utils

from losan_lofav import utils_models
from losan_lofav import utils_optimization
from losan_lofav import utils_random_fourier_features

import class_natsbench


path_results = '../results_bo'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--noise', type=str, required=True)
    parser.add_argument('--problem', type=str, required=True)
    parser.add_argument('--num_iter', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--num_arms', type=int, required=True)
    parser.add_argument('--norm', type=float, required=True)
    parser.add_argument('--sigma_bar', type=float, required=True)
    parser.add_argument('--save_results', action='store_true')

    args = parser.parse_args()

    str_model = args.model
    str_noise = args.noise
    str_problem = args.problem
    num_iter = args.num_iter
    seed = args.seed
    num_arms = args.num_arms
    S = args.norm
    sigma_bar = args.sigma_bar
    save_results = args.save_results

    sigma_zero = 1.0
    num_init = 1
    delta = 0.2
    dim_features = 128

    list_str_problem = str_problem.split('_')

    if str_problem in [
        'natsbench_cifar10-valid',
        'natsbench_cifar100',
        'natsbench_ImageNet16-120',
    ]:
        obj_problem = class_natsbench.NATSBench(list_str_problem[1])
    else:
        if len(list_str_problem) == 1:
            obj_problem = bb_utils.get_benchmark(list_str_problem[0])
        elif len(list_str_problem) == 2:
            obj_problem = bb_utils.get_benchmark(list_str_problem[0], dim=int(list_str_problem[1]))
        else:
            raise ValueError

    print('')
    print('=' * 40)
    print(f'str_model {str_model}')
    print(f'str_noise {str_noise}')
    print(f'str_problem {str_problem}')
    print(f'num_init {num_init}')
    print(f'num_iter {num_iter}')
    print(f'sigma_zero {sigma_zero}')
    print(f'sigma_bar {sigma_bar}')
    print(f'num_arms {num_arms}')
    print(f'S {S}')
    print(f'delta {delta}')
    print(f'seed {seed}')
    print('=' * 40)
    print('')

    bounds = obj_problem.get_bounds()
    random_state = np.random.RandomState(seed)

    obj_rff = utils_random_fourier_features.RFF(bounds.shape[0], dim_features, random_state)

    arms = utils_optimization.get_arms(bounds, num_arms, random_state)
    arms_transformed = obj_rff.get_rff(arms)

    dim_transformed = arms_transformed.shape[1]
    assert dim_transformed == dim_features

    def fun_target_noisy(bx):
        assert bx.ndim == 1

        X = np.atleast_2d(bx)
        by = obj_problem.output(X)
        by += utils_optimization.generate_noises(by.shape, sigma_zero, sigma_bar, str_noise, random_state)
        by *= -1.0
        return by[0, 0]

    def fun_target_noiseless(bx):
        assert bx.ndim == 1

        X = np.atleast_2d(bx)
        by = obj_problem.output(X)
        by *= -1.0
        return by[0, 0]

    X, X_transformed, by_noisy, by_noiseless, bx_best, bx_transformed_best, y_best = utils_models.run(
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
    )

    assert X.shape[0] == X_transformed.shape[0] == by_noisy.shape[0] == by_noiseless.shape[0]
    assert X.shape[1] == bx_best.shape[0]
    assert X_transformed.shape[1] == bx_transformed_best.shape[0]
    print(X.shape, X_transformed.shape, by_noisy.shape, by_noiseless.shape)
    print(np.sum(y_best - by_noiseless))
    print(y_best - np.max(by_noiseless))

    dict_all = {
        'str_model': str_model,
        'str_noise': str_noise,
        'str_problem': str_problem,
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

    str_file = f'lb_bo_{str_model}_{str_noise}_{str_problem}_{num_arms}_{S}_{sigma_zero}_{sigma_bar}_{num_init}_{num_iter}_{delta}_{seed}.npy'

    if save_results:
        np.save(os.path.join(path_results, str_file), dict_all)
