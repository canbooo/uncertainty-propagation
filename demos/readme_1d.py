import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
from experiment_design import variable
from scipy import stats

from uncertainty_propagation import (
    directional_simulation,
    monte_carlo,
    subset_simulation,
)

plt.rcParams["text.usetex"] = True


def objective(x: np.ndarray) -> np.ndarray:
    return x**2


def plot(title: str, save_name: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(18, 7))

    plt.suptitle(title)
    ax[1].hist(
        output_history,
        bins=np.linspace(0, 10, 51),
        density=True,
        cumulative=True,
        label=r"Sampled data $y_i = f(x_i)$",
        alpha=0.5,
        align="right",
        facecolor="w",
        edgecolor="k",
    )
    ax[1].plot(true_grid, true_cdf, c="b", label="True CDF")
    ax[1].plot(approx_grid, approx_cdf, c="r", label="Approximate CDF")
    ax[1].scatter(approx_grid, approx_cdf, c="r")
    ax[1].legend()

    ax[0].plot(x_grid, f_x_grid, c="k", label="$f(x)$")
    smaller_mask = results[-1].output_history < limit
    ax[0].scatter(
        results[-1].input_history[smaller_mask],
        results[-1].output_history[smaller_mask],
        c="r",
        label=rf"$f(x_i) < {limit:.2f}$",
    )
    ax[0].scatter(
        results[-1].input_history[~smaller_mask],
        results[-1].output_history[~smaller_mask],
        c="g",
        label=rf"$f(x_i) \geq {limit:.2f}$",
    )

    ax[0].axhline(limit, linestyle="--", label=rf"$f(x) = {limit:.2f}$")
    ax[0].legend(loc="upper center")
    ax[0].set_xlim([-3, 3])
    ax[0].set_ylim([0, 10])

    save_path = f"../docs/source/images/{save_name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


np.random.seed(1337)

space = variable.ParameterSpace([stats.norm()])

propagators = [
    monte_carlo.MonteCarloSimulation(),
    subset_simulation.SubsetSimulation(),
    directional_simulation.DirectionalSimulation(),
]


true_grid = np.linspace(0.0, 10.0, 101)
true_cdf = stats.chi2(df=1).cdf(true_grid)


x_grid = np.linspace(-3, 3, 101)
f_x_grid = objective(x_grid)

for propagator in propagators:
    results = []
    approx_grid = []
    total_time = 0.0
    for step, limit in enumerate(np.append([0.01, 0.2], np.linspace(0.4, 10.0, 25))):
        approx_grid.append(limit)
        t0 = time()
        results.append(
            propagator.calculate_probability(space, objective, limit=limit, cache=True)
        )
        total_time += time() - t0
        approx_cdf = [result.probability for result in results]
        output_history = np.vstack([result.output_history for result in results])
        log_value = np.log10(output_history.shape[0])
        exponent = int(log_value)
        factor = round(10 ** (log_value - exponent), 1)
        plot_title = (
            rf"{propagator.__class__.__name__} Step ${step}$ - "
            + rf"Total samples: ${factor} \cdot 10^{exponent}$ Time elapsed: {total_time:.3f}s"
        )
        plot(plot_title, f"{propagator.__class__.__name__}/1D/step_{step}.png")
