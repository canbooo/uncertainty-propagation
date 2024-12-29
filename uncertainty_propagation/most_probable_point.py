import dataclasses
import os
import warnings
from typing import Any, Callable, Type

import joblib
import numpy as np
from experiment_design import orthogonal_sampling, variable
from experiment_design.experiment_designer import ExperimentDesigner
from scipy import optimize, stats

from uncertainty_propagation.integrator import ProbabilityIntegrator
from uncertainty_propagation.transform import StandardNormalTransformer


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
            variable.ParameterSpace([stats.uniform(-1, 2) for _ in range(n_dim)]),
            n_starts - 1,
            steps=1,
        )
        x_starts = np.append(x_starts, additional, axis=0)

    mpps = []
    if n_jobs == 1 or n_starts == 1:
        for x_start in x_starts:
            x_cur = _find_mpp(optimization_envelope, x_start, bounds=bounds)
            if x_cur is None:
                continue
            mpps.append(x_cur)
        if mpps:
            mpps = np.vstack(mpps)
        else:
            mpps = np.empty((0, x_starts.shape[1]))
        return x_starts, mpps

    with joblib.Parallel(n_jobs=n_jobs) as para:
        mpps = para(
            joblib.delayed(_find_mpp)(
                optimization_envelope, x_starts[i_st], bounds=bounds
            )
            for i_st in range(x_starts.shape[0])
        )
        mpps = [point for point in mpps if point is not None]

    if mpps:
        mpps = np.vstack(mpps)
    else:
        mpps = np.empty((0, x_starts.shape[1]))

    return x_starts, mpps


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

    use_multiprocessing: bool = True
    use_standard_normal_space: bool = True

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

        if cache:
            history_u = np.append(x_starts, mpps, axis=0)
            _, history_x, history_y = envelope(history_u)
        else:
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
    n_samples: int = 256
    sample_generator: ExperimentDesigner = (
        orthogonal_sampling.OrthogonalSamplingDesigner()
    )
    sample_generator_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {"steps": 1}
    )
    transformer_cls: Type[StandardNormalTransformer] | None = None

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
