# `uncertainty-propagation`: Tools to propagate parameter uncertainty through deterministic black-box functions

<p float="left">
    <img src="docs/source/images/readme_mcs_1d.gif" alt="Image: Monte Carlo simulation to approximate 1-d function distribution" width="300">
    <img src="docs/source/images/readme_ds_1d.gif" alt="Image: Directional simulation to approximate 1-d function distribution" width="300">
    <img src="docs/source/images/readme_ss_1d.gif" alt="Image: Subset simulation to approximate 1-d function distribution" width="300">
</p>

The `uncertainty-propagation` library provides efficient and scalable methods for approximating the output distribution of $Y=f(X)$,
i.e. the probability $P(Y \leq y)$, when the input $X$ is a random variable with a known distribution.
It is especially useful for complex, black-box functions where analytical solutions are infeasible.
Methods like Monte Carlo simulation, directional simulation, and subset simulation as well as first-order reliability method
and importance sampling are integrated to provide flexible and scalable uncertainty propagation.


## Visual example

Consider the 2-d [Rastrigin function](https://www.sfu.ca/~ssurjano/rastr.html):

<img src="docs/source/images/readme_2d_function.png" alt="Image: Modified 2-d Rastrigin function" width="400">

Animations below show how directional and subset simulation tackle the same problem in different ways,
achieving similar approximation accuracy with a fraction of the Monte Carlo sample budget:

<p float="left">
    <img src="docs/source/images/readme_ds_2d.gif" alt="Image: Directional simulation to approximate 2-d function distribution" width="400">
    <img src="docs/source/images/readme_ss_2d.gif" alt="Image: Subset simulation to approximate 2-d function distribution" width="400">
</p>

## Features

- **Optimized for Speed**: Uses parallelization to maximize performance.
- **Minimal Dependencies**: Built with as few dependencies as possible.
- **Machine Learning Friendly**: Designed to integrate seamlessly with machine learning applications using vectorized function
calls.
- **Extensible Framework**: Easily integrate new methods and benchmark different approaches.

## Installation

Install the package from PyPI using

`pip install uncertainty-propagation`

## Usage

Documentation is under construction and will soon be available on ReadTheDocs. In the meantime, explore the examples or
dive into the source code to see the library in action.


## Citing

If this repository has assisted you in your research, please consider citing one of the following works:

- **Journal Paper on Uncertainty Optimization:**
```latex
@Article{Bogoclu2021,
  title       = {Local {L}atin hypercube refinement for multi-objective design uncertainty optimization},
  author      = {Can Bogoclu and Tamara Nestorovi{\'c} and Dirk Roos},
  journal     = {Applied Soft Computing},
  year        = {2021},
  arxiv       = {2108.08890},
  doi         = {10.1016/j.asoc.2021.107807},
  pdf         = {https://www.sciencedirect.com/science/article/abs/pii/S1568494621007286},
}
```
- **PhD Thesis on Uncertainty Quantification and Optimization:**
```latex
@phdthesis{Bogoclu2022,
  title       = {Local {L}atin hypercube refinement for uncertainty quantification and optimization: {A}ccelerating the surrogate-based solutions using adaptive sampling},
  author      = {Bogoclu, Can},
  school      = {Ruhr-Universit\"{a}t Bochum},
  type         = {PhD thesis},
  year        = {2022},
  doi         = {10.13154/294-9143},
  pdf         = {https://d-nb.info/1268193348/34},
}
```
