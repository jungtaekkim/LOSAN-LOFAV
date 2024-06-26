import numpy as np
from nats_bench import create


possible_choices = [8, 16, 24, 32, 40, 48, 56, 64]


class NATSBench:
    def __init__(self, dataset, num_hyperparameters=5, path_dataset='../../datasets-nats-bench/NATS-sss-v1_0-50262-simple'):
        assert dataset in ['cifar10-valid', 'cifar100', 'ImageNet16-120']

        self.path_dataset = path_dataset
        self.api = create(self.path_dataset, 'sss', fast_mode=True, verbose=False)

        self.dataset = dataset
        self.num_hyperparameters = num_hyperparameters

    def transform_continuous_to_discrete(self, inputs):
        assert isinstance(inputs, np.ndarray)
        assert inputs.shape[0] == self.num_hyperparameters

        hyperparameters = []

        for elem in inputs:
            ind_min = np.argmin(np.abs(possible_choices - elem))
            hyperparameters.append(possible_choices[ind_min])
        hyperparameters = np.array(hyperparameters)

        return hyperparameters

    def get_bounds(self):
        bounds = [
            [
                possible_choices[0] - (possible_choices[1] - possible_choices[0]) / 2,
                possible_choices[-1] + (possible_choices[-1] - possible_choices[-2]) / 2,
            ]
        ] * self.num_hyperparameters
        bounds = np.array(bounds)

        return bounds

    def validate(self, bx):
        assert isinstance(bx, np.ndarray)
        assert len(bx.shape) == 1
        assert bx.shape[0] == self.num_hyperparameters

        bounds = self.get_bounds()
        assert np.all(bx >= bounds[:, 0]) and np.all(bx <= bounds[:, 1])
        return bx

    def sample_uniform(self, num_points, seed=None):
        assert isinstance(num_points, int)
        assert isinstance(seed, (type(None), int))

        random_state = np.random.RandomState(seed)
        dim_problem = self.num_hyperparameters

        bounds = self.get_bounds()

        points = random_state.uniform(size=(num_points, dim_problem))
        points = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * points

        return points

    def get_average_valid_test(self, index):
        assert isinstance(index, int)

        info = self.api.get_more_info(index, self.dataset, is_random=False)
        acc_valid = info['valid-accuracy']
        err_valid = (100.0 - acc_valid) / 100.0

        acc_test = info['test-accuracy']
        err_test = (100.0 - acc_test) / 100.0

        return acc_valid, err_valid, acc_test, err_test

    def get_index(self, hyperparameters):
        assert isinstance(hyperparameters, np.ndarray)
        assert hyperparameters.shape[0] == self.num_hyperparameters

        out_channel_of_1st_conv_layer = hyperparameters[0]
        out_channel_of_1st_cell_stage = hyperparameters[1]
        out_channel_of_1st_residual_block = hyperparameters[2]
        out_channel_of_2nd_cell_stage = hyperparameters[3]
        out_channel_of_2nd_residual_block = hyperparameters[4]

        arch = f'{out_channel_of_1st_conv_layer}:{out_channel_of_1st_cell_stage}:{out_channel_of_1st_residual_block}:{out_channel_of_2nd_cell_stage}:{out_channel_of_2nd_residual_block}'

        index = self.api.query_index_by_arch(arch)

        return index

    def output_valid(self, bx):
        bx = self.validate(bx)
        hyperparameters = self.transform_continuous_to_discrete(bx)
        index = self.get_index(hyperparameters)
        _, err_valid, _, _ = self.get_average_valid_test(index)

        return err_valid

    def output_test(self, bx):
        bx = self.validate(bx)
        hyperparameters = self.transform_continuous_to_discrete(bx)
        index = self.get_index(hyperparameters)
        _, _, _, err_test = self.get_average_valid_test(index)

        return err_test

    def output(self, X):
        assert isinstance(X, np.ndarray)
        assert X.ndim == 2

        Y = []
        for bx in X:
            Y.append([self.output_valid(bx)])
        return np.array(Y)
