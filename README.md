# Noise-Adaptive Confidence Sets for Linear Bandits and Application to Bayesian Optimization

It is an official repository of "[Noise-Adaptive Confidence Sets for Linear Bandits and Application to Bayesian Optimization](https://arxiv.org/abs/2402.07341)," which has been presented at [the 41st International Conference on Machine Learning (ICML 2024)](https://icml.cc/Conferences/2024).

- ICML proceedings
- [arXiv preprint](https://arxiv.org/abs/2402.07341)

## Installation

You can install `losan_lofav` using the following command.
Command it in the root directory.

```shell
pip install .
```

## Experiments

You can run experiments in the `experiments` directory.
Use the following commands to conduct experiments.

- Synthetic functions

```shell
python run_lb_synthetic.py --model losan --noise gaussian --problem 32 --num_iter 100 --seed 42 --num_arms 128 --norm 1.0 --sigma_bar 0.01 --save_results
python run_lb_synthetic.py --model lofav --noise rademacher --problem 32 --num_iter 100 --seed 42 --num_arms 128 --norm 1.0 --sigma_bar 0.01 --save_results
```

- Benchmark functions

```shell
python run_lb_bo.py --model losan --noise gaussian --problem branin --num_iter 100 --seed 42 --num_arms 512 --norm 1.0 --sigma_bar 0.01 --save_results
python run_lb_bo.py --model lofav --noise rademacher --problem branin --num_iter 100 --seed 42 --num_arms 512 --norm 1.0 --sigma_bar 0.01 --save_results
```

- NATS-Bench

```shell
python run_lb_bo.py --model losan --noise gaussian --problem natsbench_ImageNet16-120 --num_iter 100 --seed 42 --num_arms 512 --norm 1.0 --sigma_bar 0.01 --save_results
python run_lb_bo.py --model lofav --noise rademacher --problem natsbench_ImageNet16-120 --num_iter 100 --seed 42 --num_arms 512 --norm 1.0 --sigma_bar 0.01 --save_results
```

## Citation

```
@inproceedings{JunKS2024icml,
    title={Noise-Adaptive Confidence Sets for Linear Bandits and Application to {Bayesian} Optimization},
    author={Jun, Kwang-Sung and Kim, Jungtaek},
    booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
    year={2024},
    address={Vienna, Austria)
}
```

## License

[MIT License](LICENSE)
