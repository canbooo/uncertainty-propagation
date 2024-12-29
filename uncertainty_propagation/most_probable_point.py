import dataclasses
import os
import warnings
from typing import Any, Callable, Type

import numpy as np
from experiment_design import orthogonal_sampling, variable
from experiment_design.experiment_designer import ExperimentDesigner
from scipy import optimize, stats

from uncertainty_propagation.integrator import ProbabilityIntegrator
from uncertainty_propagation.transform import StandardNormalTransformer
from uncertainty_propagation.utils import single_or_multiprocess


@dataclasses.dataclass
class FirstOrderApproximationSettings:
    n_searches: int | None = None
    pooled: bool = True
    n_jobs: int = os.cpu_count()
    transformer_cls: Type[StandardNormalTransformer] | None = None

    def __post_init__(self):
        if self.n_searches is None:
            self.n_searches = self.n_jobs


class FirstOrderApproximation(ProbabilityIntegrator):
    """
    First order i.e. linear approximation of the propagated probability. In the context of reliability analysis,
    this method is knows as FORM.

    Assumes P(Y <= y) = phi_inv(||x_mpp||) where phi_inv is the inverse of the
    standard normal distribution and ||x_mpp|| is the distance to the most probable, i.e. closest point, with
    f(x_mpp) = y in the standard normal space. See FirstOrderApproximationSettings docstring for further details.

    A. M. Hasofer and N. Lind (1974). “Exact and Invariant Second Moment Code Format”
    https://www.researchgate.net/publication/243758427_An_Exact_and_Invariant_First_Order_Reliability_Format

    C. Song and R. Kawai, (2023). "Monte Carlo and variance reduction methods for structural reliability analysis:
    A comprehensive review"
    https://doi.org/10.1016/j.probengmech.2023.103479
    """

    def __init__(self, settings: FirstOrderApproximationSettings | None = None):
        if settings is None:
            settings = FirstOrderApproximationSettings()
        self.settings = settings
        super(FirstOrderApproximation, self).__init__(self.settings.transformer_cls)

    def _calculate_probability(
        self,
        space: variable.ParameterSpace,
        envelope: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
        cache: bool = False,
    ) -> tuple[float, float, tuple[np.ndarray | None, np.ndarray | None]]:
        """Currently, full caching is not available so we cache only the start and solution points"""
        x_starts, mpps = find_most_probable_boundary_points(
            envelope,
            space.dimensions,
            n_starts=self.settings.n_searches,
            n_jobs=self.settings.n_jobs,
        )

        history_x, history_y = None, None
        if cache:
            history_u = np.append(x_starts, mpps, axis=0)
            _, history_x, history_y = envelope(history_u)
        elif mpps.shape[0] > 0:
            history_u = x_starts[[0]]
            _, history_x, history_y = envelope(history_u)

        if mpps.shape[0] == 0:
            return 0.0, 0.0, (history_x, history_y)

        # We depend on the fact that first sample in x_start is [0, 0, ...]
        factor = -1 if history_y[0] >= 0 else 1

        safety_indexes = np.linalg.norm(mpps, axis=1)

        if not self.settings.pooled:
            probability = stats.norm.cdf(factor * np.min(safety_indexes))
            return probability, 0.0, (history_x, history_y)

        probability = stats.norm.cdf(factor * np.mean(safety_indexes))
        std_dev = np.std(stats.norm.cdf(factor * safety_indexes), ddof=1)
        std_err = std_dev / np.sqrt(safety_indexes.shape[0])
        return probability, std_err, (history_x, history_y)


@dataclasses.dataclass
class ImportanceSamplingSettings:
    n_searches: int | None = None
    pooled: bool = True
    n_jobs: int = os.cpu_count()
    n_samples: int = 128
    sample_generator: ExperimentDesigner = (
        orthogonal_sampling.OrthogonalSamplingDesigner()
    )
    sample_generator_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {"steps": 1}
    )
    transformer_cls: Type[StandardNormalTransformer] | None = None
    comparison: Callable[[np.ndarray, float], np.ndarray] = np.less_equal

    def __post_init__(self):
        if self.n_searches is None:
            self.n_searches = self.n_jobs


class ImportanceSampling(ProbabilityIntegrator):
    """
    Importance Sampling Procedure Using Design point

    Importance sampling uses an auxilary distribution q* to estimate the
    integral with lower variance compared to MC. ISPUD transforms the space
    to the standard normal and uses MPP as the mean of q*, which is estimated
    as a normal distribution with unit variance.

    U. Bourgund (1986). "Importance Sampling Procedure Using Design Points, ISPUD: A User's Manual: An Efficient,
    Accurate and Easy-to-use Multi-purpose Computer Code to Determine Structural Reliability"

    A. Tabandeh et al. (2022). "A review and assessment of importance sampling methods for reliability analysis"

    A. B. Owen (2013). "Monte Carlo theory, methods and examples"
    https://artowen.su.domains/mc/
    """

    def __init__(self, settings: ImportanceSamplingSettings | None = None):
        if settings is None:
            settings = ImportanceSamplingSettings()
        self.settings = settings
        super(ImportanceSampling, self).__init__(self.settings.transformer_cls)

    def _calculate_probability(
        self,
        space: variable.ParameterSpace,
        envelope: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
        cache: bool = False,
    ) -> tuple[float, float, tuple[np.ndarray | None, np.ndarray | None]]:
        x_starts, mpps = find_most_probable_boundary_points(
            envelope,
            space.dimensions,
            n_starts=self.settings.n_searches,
            n_jobs=self.settings.n_jobs,
        )

        history_x, history_y = None, None
        if cache:
            history_u = np.append(x_starts, mpps, axis=0)
            _, history_x, history_y = envelope(history_u)
        elif mpps.shape[0] > 0:
            history_u = x_starts[[0]]
            _, history_x, history_y = envelope(history_u)

        if mpps.shape[0] == 0:
            return 0.0, 0.0, (history_x, history_y)

        if not self.settings.pooled:
            distances = np.linalg.norm(mpps, axis=1)
            mpps = mpps[[np.argmin(distances)]]

        def for_loop_body(x):
            return _importance_sample(
                envelope,
                x,
                self.settings.sample_generator,
                self.settings.n_samples,
                self.settings.sample_generator_kwargs,
                std_dev=1.0,
                comparison=self.settings.comparison,
            )

        results = single_or_multiprocess(mpps, for_loop_body, self.settings.n_jobs)

        probabilities = np.empty(0)
        for result in results:
            cur_probs, cur_hist_x, cur_hist_y = result
            probabilities = np.append(probabilities, cur_probs)
            if cache:
                history_x = np.append(history_x, cur_hist_x, axis=0)
                history_y = np.append(history_y, cur_hist_y, axis=0)

        probability = probabilities.mean()
        std_err = np.std(probabilities, ddof=1) / np.sqrt(probabilities.shape[0])
        return probability, std_err, (history_x, history_y)


def find_most_probable_boundary_points(
    envelope: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_dim: int,
    n_starts: int = 12,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """

    :param envelope:
    :param n_dim:
    :param n_starts:
    :param n_jobs:
    :return: mpps and their objectives
    """

    def optimization_envelope(x):
        x = np.array(x)
        if x.ndim < 2:
            x = x.reshape((1, -1))
        return envelope(x)[0]

    lim = 7
    bounds = [(-lim, lim) for _ in range(n_dim)]
    x_starts = np.zeros((1, n_dim))
    if n_starts > 1:
        designer = orthogonal_sampling.OrthogonalSamplingDesigner()
        additional = designer.design(
            variable.ParameterSpace([stats.uniform(-2, 4) for _ in range(n_dim)]),
            n_starts - 1,
            steps=1,
        )
        x_starts = np.append(x_starts, additional, axis=0)

    def for_loop_body(x):
        return _find_mpp(optimization_envelope, x, bounds=bounds)

    mpps = single_or_multiprocess(x_starts, for_loop_body, n_jobs=n_jobs)
    mpps = [point for point in mpps if point is not None]

    if mpps:
        mpps = np.vstack(mpps)
    else:
        mpps = np.empty((0, x_starts.shape[1]))
    return x_starts, mpps


def _importance_sample(
    std_norm_envelope: Callable[
        [np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
    mean: np.ndarray,
    sample_generator: ExperimentDesigner,
    n_sample: int,
    sample_generator_kwargs: dict[str, Any] | None = None,
    std_dev: float = 1.0,
    comparison: Callable[[np.ndarray, float], np.ndarray] = np.less_equal,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_dists = [stats.norm(x_i, std_dev) for x_i in mean.ravel()]
    doe = sample_generator.design(
        variable.ParameterSpace(sample_dists), n_sample, **sample_generator_kwargs
    )
    y_min, history_x, history_y = std_norm_envelope(doe)
    weights = np.prod(stats.norm.pdf(doe), axis=1)
    denominator = np.zeros_like(doe)
    for i_dim in range(doe.shape[1]):
        denominator[:, i_dim] = sample_dists[i_dim].pdf(doe[:, i_dim])
    weights /= np.prod(denominator, axis=1)
    probabilities = weights * comparison(y_min, 0.0)
    return probabilities, history_x, history_y


def _find_mpp(
    limit_state: Callable[[np.ndarray], np.ndarray], x_start: np.ndarray, bounds=None
) -> np.ndarray | None:
    """Get MPP using SLSQP and if that fails, use the slower cobyla"""

    def mpp_obj(x: np.ndarray) -> np.ndarray:
        return np.sum(x**2)

    def mpp_jac(x: np.ndarray) -> np.ndarray:
        return 2 * x

    def call_optimizer(method: str) -> optimize.OptimizeResult:
        """calls scipy optimizer"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return optimize.minimize(
                mpp_obj,
                x_start,
                jac=mpp_jac,
                method=method,
                constraints=constraints,
                bounds=bounds,
            )

    constraints = {"type": "eq", "fun": limit_state}

    try:
        res = call_optimizer(method="SLSQP")
    except ValueError:
        pass
    else:
        success = res.get("status") not in [5, 6] and res.success
        if success:
            return res.get("x")

    constraints = (
        {"type": "ineq", "fun": limit_state},
        {"type": "ineq", "fun": lambda x: -limit_state(x)},
    )
    res = call_optimizer(method="COBYLA")
    if res.success:
        return res.get("x")
    return None
