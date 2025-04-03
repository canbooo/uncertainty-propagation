import dataclasses
import logging
import os
from typing import Any, Callable, Type

import numpy as np
from experiment_design import variable
from scipy import optimize, stats

from uncertainty_propagation import integrator, utils
from uncertainty_propagation.directional_simulation.partitioners import (
    DirectionGenerator,
    fekete_directions,
)
from uncertainty_propagation.transform import StandardNormalTransformer


@dataclasses.dataclass
class DirectionalSimulationSettings:
    r"""
     Settings for the directional simulation

    :param  probability_tolerance: Defines the target accuracy of the estimated failure probability in terms
         of digit precision. (Default=1e-12)
     :param n_directions: Number of directions to use. Can be a callable, that accepts the number of dimensions as inputs
         and returns the number of directions to generate. If None (default), it will be set to `80 * space.dimensions`.
     :param min_samples_per_direction: Number of samples to search for sign change in each direction. (Default=32)
     :param direction_generator: Callable that takes number of directions and dimensions as input and returns direction
         vectors with unit length.
     :param n_jobs: Number of jobs for parallel computation of directional searches.
         By default, uses the number of cpu cores.
     :param monotonic: If True, search will terminate after finding the first zero crossing in each direction. Otherwise
         (default), all zero-crossings within the relevant n-sphere as defined by the probability_tolerance will be
         sought.
     :param transformer_cls: Class to use for transforming the propagation function to standard normal space. Must
         follow, StandardNormalTransformer protocol. If None (default), either InverseTransformSampler or
         NatafTransformer will be used depending on if the ParameterSpace has a non-unity correlation matrix.
     :param zero_tolerance: Absolute tolerance to check whether a value is equal to 0. Used to determine zero-crossings.
     :param comparison:  Boolean-comparison operator. Should generally be either `np.less` or `np.less_equal`, depending
         on whether the calculated probability is defined as :math:`$P(Y<y)$` or :math:`$P(Y \leq y)$`. By default, it
         uses `np.less_equal`to match the CDF definition but for reliability analysis use case, using `np.less` might be
         more appropriate. In reality, since :math:`$P(Y=y) = 0$` for continuous Y, this is not expected to have a
         significant effect it most cases.
    """

    probability_tolerance: float = 1e-12
    n_directions: int | Callable[[int], int] | None = None
    min_samples_per_direction: int = 32
    direction_generator: DirectionGenerator = fekete_directions
    direction_generator_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {"max_steps_per_solution": 500}
    )
    n_jobs: int = os.cpu_count()
    monotonic: bool = False
    transformer_cls: Type[StandardNormalTransformer] | None = None
    zero_tolerance: float = 1e-16
    comparison: Callable[
        [np.ndarray | float, np.ndarray | float], np.ndarray | float
    ] = np.less_equal

    def directions_for(self, n_dimensions: int) -> np.ndarray:
        match self.n_directions:
            case None:
                n_directions = n_dimensions * 80
            case int():
                n_directions = self.n_directions
            case _:
                n_directions = self.n_directions(n_dimensions)
        return self.direction_generator(
            n_directions, n_dimensions, **self.direction_generator_kwargs
        )


class DirectionalSimulation(integrator.ProbabilityIntegrator):
    """
    Directional simulation for the probability integration. See Chapter 2.3.2 for equation references in this file
    https://hss-opus.ub.ruhr-uni-bochum.de/opus4/frontdoor/deliver/index/docId/9143/file/diss.pdf

    See DirectionalSimulationSettings documentation for  further details.

    Further references:
    P. Bjerager (1988). “Probability Integration by Directional Simulation”
    J. Nie and B. R. Ellingwood (2000). “Directional methods for structural reliability analysis”
    J. Nie and B. R. Ellingwood (2004). “new directional simulation method for system reliability. Part II: application
    of neural network”

    """

    def __init__(self, settings: DirectionalSimulationSettings | None = None) -> None:
        if settings is None:
            settings = DirectionalSimulationSettings()
        self.settings = settings
        super(DirectionalSimulation, self).__init__(self.settings.transformer_cls)

    def _calculate_probability(
        self,
        space: variable.ParameterSpace,
        envelope: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
        cache: bool = False,
    ) -> tuple[float, float, tuple[np.ndarray | None, np.ndarray | None]]:
        n_dim = space.dimensions
        directions = self.settings.directions_for(n_dim)
        center, history_x, history_y = envelope(np.zeros((1, space.dimensions)))
        max_distance = np.sqrt(
            stats.chi2.ppf(1 - self.settings.probability_tolerance, df=space.dimensions)
        )  # Eq. 2.134
        search_grid = np.linspace(
            max_distance / self.settings.min_samples_per_direction,
            max_distance,
            self.settings.min_samples_per_direction,
        )
        zero_inclusive = bool(self.settings.comparison(np.zeros(1), 0.0))

        def for_loop_body(direction):
            return _directional_probability(
                envelope,
                direction.reshape((1, -1)),
                search_grid,
                center,
                find_all=not self.settings.monotonic,
                zero_tol=self.settings.zero_tolerance,
                zero_inclusive=zero_inclusive,
            )

        results = utils.single_or_multiprocess(
            directions, for_loop_body, n_jobs=self.settings.n_jobs
        )
        probabilities = []
        for result in results:
            probabilities.append(result[0])
            if cache:
                history_x, history_y = utils.extend_cache(
                    history_x,
                    history_y,
                    result[1],
                    result[2],
                    cache_x=cache,
                    cache_y=cache,
                )

        if self.settings.comparison(1, 0):
            # greater comparison so the inverse probability should be returned
            probabilities = 1.0 - np.array(probabilities)
        probability = float(np.mean(probabilities))
        std_err = np.std(probabilities, ddof=1) / np.sqrt(len(probabilities))
        return probability, std_err, (history_x, history_y)


def _directional_probability(
    envelope: Callable[[float | np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    direction: np.ndarray,
    search_grid: np.ndarray,
    center: np.ndarray,
    find_all: bool = True,
    zero_tol: float = 1e-16,
    zero_inclusive: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    roots, history_x, history_y = _find_sign_changes(
        envelope,
        direction,
        search_grid,
        center,
        find_all=find_all,
        zero_tol=zero_tol,
        zero_inclusive=zero_inclusive,
    )
    roots = np.sort(roots)
    roots = np.sign(roots) * roots**2  # to handle -infty

    if center > zero_tol or (
        np.isclose(center, 0, atol=zero_tol) and not zero_inclusive
    ):
        directional_probabilities = 1 - stats.chi2.cdf(
            roots, df=direction.size
        )  # Eq. 2.105
    else:
        directional_probabilities = stats.chi2.cdf(roots, df=direction.size)
    if directional_probabilities.shape[0] == 1:
        return float(directional_probabilities[0]), history_x, history_y
    signs = np.ones(directional_probabilities.shape)
    if np.isclose(center, 0, atol=zero_tol):
        neg_slice = slice(0, None, 2) if zero_inclusive else slice(1, None, 2)
    elif center > 0:
        neg_slice = slice(1, None, 2)
    else:
        neg_slice = slice(0, None, 2)
    signs[neg_slice] = -1
    total_directional_probability = np.sum(
        directional_probabilities * signs
    )  # Eq 2.135
    return total_directional_probability, history_x, history_y


def _find_sign_changes(
    envelope: Callable[[float | np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    direction: np.ndarray,
    search_grid: np.ndarray,
    center: np.ndarray,
    find_all: bool = True,
    zero_tol: float = 1e-16,
    zero_inclusive: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history_x, history_y = [], []

    def direction_envelope(radius: float | np.ndarray) -> np.ndarray:
        radius = np.array(radius).reshape((-1, 1))
        result, hist_x, hist_y = envelope(radius * direction)
        history_x.append(hist_x)
        history_y.append(hist_y)
        return result

    def reshaped_histories() -> tuple[np.ndarray, np.ndarray]:
        if not history_x:
            return np.empty((0, direction.size)), np.empty((0, 1))
        hist_x = history_x[0]
        hist_y = history_y[0]
        for cur_x, cur_y in zip(history_x[1:], history_y[1:]):
            hist_x = np.append(hist_x, cur_x, axis=0)
            hist_y = np.append(hist_y, cur_y, axis=0)
        hist_x = hist_x.reshape((-1, direction.size))
        hist_y = hist_y.reshape((hist_x.shape[0], -1))
        return hist_x, hist_y

    results = direction_envelope(search_grid)
    results = np.append([center], results).tolist()
    search_grid = np.append([0], search_grid).tolist()

    ids = np.where(np.logical_not(np.isclose(results, 0, atol=zero_tol)))[0]

    if ids.size == 0:
        history_x, history_y = reshaped_histories()
        return np.array([float("inf")]), history_x, history_y

    roots = []
    for next_id in range(max(0, int(ids[0]) - 1) + 1, len(results)):
        prev_id = next_id - 1
        sign_check = results[prev_id] * results[next_id]
        if sign_check > zero_tol:
            continue
        solutions = _branch_and_bound_roots(
            direction_envelope,
            search_grid[prev_id],
            search_grid[next_id],
            results[prev_id],
            results[next_id],
        )
        if solutions is not None:
            roots.extend(solutions)
        if not find_all:
            if len(solutions) > 1:
                logging.warning(
                    "Function was assumed to be monotonic but non-monotonic behaviour observed."
                )
            break

    if not roots:
        roots = np.array([float("inf")])
        if (
            np.isclose(center, 0.0, atol=zero_tol)
            and results[-1] < 0
            and not zero_inclusive
        ):
            # Every grid point was negative and center is 0., so we want to set the probability of
            # this direction to 1.
            # In case zero_inclusive, setting root to infinity yields 1. in this case,
            # otherwise negative infinity
            roots = np.array([float("-inf")])

    history_x, history_y = reshaped_histories()
    return np.array(roots), history_x, history_y


def _branch_and_bound_roots(
    direction_envelope: Callable[[float | np.ndarray], np.ndarray],
    r_min: float,
    r_max: float,
    f_r_min: float,
    f_r_max: float,
    x_tol: float = 1e-9,
    zero_tol: float = 1e-16,
) -> list[float] | None:
    if np.isclose(r_min, r_max, atol=x_tol, rtol=0):
        return None
    if np.isclose(f_r_min, 0.0, atol=zero_tol) or np.isclose(f_r_max, 0, atol=zero_tol):
        interval, _ = _find_root_finding_boundary(
            direction_envelope, r_min, r_max, f_r_min, f_r_max
        )
        if interval is None:
            return None
        if np.isclose(f_r_min, 0.0, atol=zero_tol):
            f_r_min = (
                -1.0 * f_r_max
            )  # only needed for sign check so value does not matter
        elif np.isclose(f_r_max, 0.0, atol=zero_tol):
            f_r_max = (
                -1.0 * f_r_min
            )  # only needed for sign check so value does not matter
    else:
        interval = (r_min, r_max)
    root = _call_brent(direction_envelope, interval[0], interval[1])
    if root is None:
        return None
    solutions = [root]
    left_solutions = _branch_and_bound_roots(
        direction_envelope, interval[0], root, f_r_min, 0.0, x_tol=x_tol
    )
    right_solutions = _branch_and_bound_roots(
        direction_envelope, root, interval[1], 0.0, f_r_max, x_tol=x_tol
    )
    for other_solutions in [left_solutions, right_solutions]:
        if other_solutions is None:
            continue
        solutions.extend([sol for sol in _flatten(other_solutions) if sol is not None])
    return solutions


def _call_brent(
    direction_envelope: Callable[[float | np.ndarray], np.ndarray],
    r_min: float,
    r_max: float,
) -> float | None:
    try:
        _, root_result = optimize.brentq(
            direction_envelope, r_min, r_max, maxiter=1000, full_output=True
        )
    except ValueError:
        return None
    return float(root_result.root) if root_result.converged else None


def _find_root_finding_boundary(
    direction_envelope: Callable[[float | np.ndarray], np.ndarray],
    a: float,
    b: float,
    f_a: float,
    f_b: float,
    x_tol: float = 1e-9,
    zero_tol: float = 1e-16,
) -> tuple[list[float] | None, float | None]:
    """Use binary search to find the end of the region containing 0 as well as a valid bracket to search
    for a root. We assume that either a or b is 0 and the other one is non-zero
    """
    if np.isclose(f_a, 0.0, atol=zero_tol) and np.isclose(f_b, 0.0, atol=zero_tol):
        b = (a + b) / 2
        f_b = direction_envelope(b)
        if np.isclose(f_b, 0.0, atol=zero_tol):
            return None, None

    if np.isclose(f_b, 0.0, atol=zero_tol):
        a, b = b, a
        f_a, f_b = f_b, f_a

    closest_observation_to_root_boundary = a

    while np.isclose(a, b, atol=x_tol, rtol=0.0):
        c = (a + b) / 2
        f_c = direction_envelope(c)

        if np.isclose(f_c, 0.0, atol=zero_tol):
            a = closest_observation_to_root_boundary = c
        elif f_c * f_b < 0:
            interval = sorted([c, b])
            return interval, closest_observation_to_root_boundary
        else:
            b = c
            f_b = f_c
    return None, a


def _flatten(seq: list[Any]) -> list[float]:
    for ele in seq:
        if ele is None or isinstance(ele, (int, float)):
            yield ele
        else:
            yield from _flatten(ele)
