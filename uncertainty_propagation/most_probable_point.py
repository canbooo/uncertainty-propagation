import dataclasses
import warnings
from typing import Callable, Type

import joblib
import numpy as np
from experiment_design import orthogonal_sampling, variable
from scipy import optimize, stats

from uncertainty_propagation.integrator import ProbabilityIntegrator
from uncertainty_propagation.transform import StandardNormalTransformer


def _find_mpp(
    limit_state: Callable[[np.ndarray], np.ndarray], x_start: np.ndarray, bounds=None
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
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
    return None, None


def find_most_probable_points(
    envelope: Callable[[np.ndarray], np.ndarray],
    n_dim: int,
    n_starts: int = 12,
    n_jobs: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """

    :param envelope:
    :param n_dim:
    :param n_starts:
    :param n_jobs:
    :return: mpps and their objectives
    """

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
            x_cur = _find_mpp(envelope, x_start, bounds=bounds)
            if x_cur is None:
                continue
            mpps.append(x_cur)
        return x_starts, np.vstack(mpps)

    with joblib.Parallel(n_jobs=n_jobs) as para:
        mpps = para(
            joblib.delayed(_find_mpp)(
                envelope, x_starts[[i_st]], bounds=bounds, give_vals=False
            )
            for i_st in range(x_starts.shape[0])
        )
        mpps = np.vstack(mpps)

    return x_starts, mpps


@dataclasses.dataclass
class FirstOrderApproximationSettings:
    n_searches: int = 8
    pooled: bool = True
    n_jobs: int = -1
    transformer_cls: Type[StandardNormalTransformer] | None = None


class FirstOrderApproximation(ProbabilityIntegrator):
    """
    First order i.e. linear approximation of the propagated probability. In the context of reliability analysis,
    this method is knows as FORM.

    Assumes P(Y <= y) = phi_inv(||x_mpp||) where phi_inv is the inverse of the
    standard normal distribution and ||x_mpp|| is the distance to the most probable, i.e. closest point, with
    f(x_mpp) = y in the standard normal space. See FirstOrderApproximationSettings docstring for further details.
    """

    use_multiprocessing: bool = True
    use_standard_normal_space: bool = True

    def __init__(self, settings: FirstOrderApproximationSettings):
        self.settings = settings
        super(FirstOrderApproximation, self).__init__(self.settings.transformer_cls)

    def _calculate_probability(
        self,
        space: variable.ParameterSpace,
        envelope: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
        cache: bool = False,
    ) -> tuple[float, float, tuple[np.ndarray | None, np.ndarray | None]]:
        """Currently, full caching is not available so we cache only the start and solution points"""
        x_starts, mpps = find_most_probable_points(
            lambda x: envelope(x)[0],
            space.dimensions,
            n_starts=self.settings.n_searches,
            n_jobs=self.settings.n_jobs,
        )
        history_x, history_y = None, None
        if cache:
            history_u = np.append(x_starts, mpps)
            _, history_x, history_y = envelope(history_u)
        safety_indexes = np.linalg.norm(mpps, axis=1)
        fail_prob_mu = np.mean(stats.norm.cdf(-safety_indexes))
        return fail_prob_mu, 0.0, (history_x, history_y)
