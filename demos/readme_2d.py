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
    return 10.0 - np.sum(x**2 - 5 * np.cos(2 * np.pi * x), axis=1)


def plot(title: str, save_name: str) -> None:
    assert mc_approx_cdf is not None
    fig, ax = plt.subplots(1, 2, figsize=(18, 7))

    plt.suptitle(title)
    ax[1].hist(
        output_history,
        bins=np.linspace(-20, 20, 81),
        density=True,
        cumulative=True,
        label=r"Sampled data $y_i = f(x_i)$",
        alpha=0.5,
        align="right",
        facecolor="w",
        edgecolor="k",
    )
    ax[1].plot(
        mc_approx_grid,
        mc_approx_cdf,
        c="b",
        label=f"Empiric CDF from {mc_approx_sample_txt} samples",
    )
    ax[1].plot(approx_grid, approx_cdf, c="r", label="Approximate CDF")
    ax[1].scatter(approx_grid, approx_cdf, c="r")
    ax[1].legend()
    ax[1].grid()

    ax[0].contour(X_grid, Y_grid, f_x_grid, levels=[limit])
    smaller_mask = results[-1].output_history.ravel() < limit

    ax[0].scatter(
        results[-1].input_history[~smaller_mask][:, 0],
        results[-1].input_history[~smaller_mask][:, 1],
        c="g",
        s=3,
        label=rf"$f(x_i) \geq {limit:.2f})$",
    )

    ax[0].scatter(
        results[-1].input_history[smaller_mask][:, 0],
        results[-1].input_history[smaller_mask][:, 1],
        c="r",
        s=3,
        label=rf"$f(x_i) < {limit:.2f}$",
    )

    ax[0].legend(loc="upper center")
    ax[0].grid()
    ax[0].set_xlim([-4, 4])
    ax[0].set_ylim([-4, 4])
    save_and_close_current_figure(save_name)


def save_and_close_current_figure(save_name: str) -> None:
    save_path = f"../docs/source/images/{save_name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def scientific_latex_number(num: int | float) -> str:
    log_value = np.log10(num)
    exponent = int(log_value)
    factor = round(10 ** (log_value - exponent), 1)
    return rf"${factor} \cdot 10^{exponent}$"


np.random.seed(1337)

space = variable.ParameterSpace([stats.norm() for _ in range(2)])

propagators = [
    monte_carlo.MonteCarloSimulation(),
    subset_simulation.SubsetSimulation(),
    directional_simulation.DirectionalSimulation(),
]


true_grid = np.linspace(-5, 20, 51)
mc_approx_cdf, mc_approx_grid, mc_approx_sample_txt = None, None, ""

x_grid = np.linspace(-4, 4, 201)
X_grid, Y_grid = np.meshgrid(x_grid, x_grid)
f_x_grid = objective(np.c_[X_grid.ravel(), Y_grid.ravel()]).reshape(X_grid.shape)

figure = plt.figure(figsize=(12, 7))
plt.contourf(X_grid, Y_grid, f_x_grid)
plt.colorbar()
save_and_close_current_figure("readme_2d_function.png")

for propagator in propagators:
    results = []
    approx_grid = []
    approx_cdf = []
    total_time = 0.0
    for step, limit in enumerate(true_grid):
        approx_grid.append(limit)
        t0 = time()
        results.append(
            propagator.calculate_probability(space, objective, limit=limit, cache=True)
        )
        total_time += time() - t0
        approx_cdf.append(results[-1].probability)
        output_history = np.vstack([result.output_history for result in results])
        if mc_approx_cdf is None:
            # This is the first run to create an approximate but accurate CDF using many samples with MC
            continue
        sample_txt = scientific_latex_number(output_history.shape[0])
        plot_title = (
            rf"{propagator.__class__.__name__} Step ${step}$ - "
            + rf"Total samples: {sample_txt} Time elapsed: {total_time:.3f}s"
        )
        plot(plot_title, f"{propagator.__class__.__name__}/2D/step_{step}.png")

    if mc_approx_cdf is None:
        mc_approx_cdf = approx_cdf.copy()
        mc_approx_grid = approx_grid.copy()
        mc_approx_sample_txt = scientific_latex_number(output_history.shape[0])
