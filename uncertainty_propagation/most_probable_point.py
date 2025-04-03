import dataclasses
import os
import warnings
from typing import Any, Callable, Type

import numpy as np
from experiment_design import orthogonal_sampling, variable
from experiment_design.experiment_designer import ExperimentDesigner
from scipy import optimize, stats

from uncertainty_propagation import utils
from uncertainty_propagation.integrator import ProbabilityIntegrator
from uncertainty_propagation.transform import StandardNormalTransformer


@dataclasses.dataclass
class FirstOrderApproximationSettings:
    r"""
    Settings for first order approximation or FORM

    :param n_searches: Number of searches for the initial most probable boundary point search. If None (default), it
    will be set to n_jobs.
    :param pooled: If True (default), average distance of all found most probable boundary points will be used to
    compute the probability, otherwise the smallest will be used.
    :param n_jobs: Number of jobs for parallel computation for the most probable boundary point search. By default,
    uses the number of cpu cores.
    :param transformer_cls: Class to use for transforming the propagation function to standard normal space. Must follow,
    StandardNormalTransformer protocol. If None (default), either InverseTransformSampler or NatafTransformer will be
    used depending on if the ParameterSpace has a non-unity correlation matrix.
    :param comparison: Boolean-comparison operator. Should generally be either `np.less` or `np.less_equal`, depending
    on whether the calculated probability is defined as :math:`$P(Y<y)$` or :math:`$P(Y \leq y)$`. By default, it uses
    `np.less_equal`to match the CDF definition but for reliability analysis use case, using `np.less` might be more
    appropriate. In reality, since :math:`$P(Y=y) = 0$` for continuous Y, this is not expected to have a significant
    effect.
    """

    n_searches: int | None = None
    pooled: bool = True
    n_jobs: int = os.cpu_count()
    transformer_cls: Type[StandardNormalTransformer] | None = None
    comparison: Callable[
        [np.ndarray | float, np.ndarray | float], np.ndarray | float
    ] = np.less_equal

    def __post_init__(self):
        if self.n_searches is None:
            self.n_searches = self.n_jobs


class FirstOrderApproximation(ProbabilityIntegrator):
    r"""
    First order i.e. linear approximation of the propagated probability. In the context of reliability analysis,
    this method is knows as FORM.

    Assumes :math:`P(Y \leq y) = \phi^{-1}(||x^*||)` where :math:`\phi^{-1}` is the inverse of the
    standard normal distribution and :math:`||x^*||` is the distance to the most probable, i.e. closest point, with
    :math:`f(x^*) = y` in the standard normal space. See FirstOrderApproximationSettings docstring for further details.

    A. M. Hasofer and N. Lind (1974). “Exact and Invariant Second Moment Code Format”
    https://www.researchgate.net/publication/243758427_An_Exact_and_Invariant_First_Order_Reliability_Format

    C. Song and R. Kawai, (2023). "Monte Carlo and variance reduction methods for structural reliability analysis:
    A comprehensive review"
    https://doi.org/10.1016/j.probengmech.2023.103479
    """

    def __init__(self, settings: FirstOrderApproximationSettings | None = None) -> None:
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
        mpps, history_x, history_y = find_most_probable_boundary_points(
            envelope,
            space.dimensions,
            n_search=self.settings.n_searches,
            n_jobs=self.settings.n_jobs,
            cache=cache,
        )

        # Although this is a recomputation, it is difficult to avoid
        # as history_y alone is not sufficient to handle cases, where the passed limit to
        # calculated probability is != 0. We do not reinsert it to the history.
        if self.settings.comparison(envelope(np.zeros((1, space.dimensions)))[0], 0.0):
            factor = 1
        else:
            factor = -1.0

        if mpps.shape[0] == 0:
            # following is already in the history if caching is enabled
            cur, _, _ = envelope(np.zeros((1, space.dimensions)))
            probability = 1.0 if self.settings.comparison(cur, 0.0) else 0.0
            return probability, 0.0, (history_x, history_y)

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
    r"""
    Settings for importance sampling

    :param n_searches: Number of searches for the initial most probable boundary point search. If None (default), it
    will be set to n_jobs.
    :param pooled: If True (default), importance sampling will be conducted at all found most probable boundary points.
    Otherwise, the closest one will be selected.
    :param n_jobs: Number of jobs for parallel computation both for most probable boundary point search and for the
    consequent sampling. By default, uses the number of cpu cores.
    :param n_samples: Number of samples to generate at each used most probable boundary point. (Default=128)
    :param sample_generator: ExperimentDesigner to generate samples from. (Default: OrthogonalSamplingDesigner)
    :param sample_generator_kwargs: Any keyword arguments for the passed ExperimentDesigner. (Default = `{"steps": 1}`)
    :param transformer_cls: Class to use for transforming the propagation function to standard normal space. Must follow,
    StandardNormalTransformer protocol If None (default), either InverseTransformSampler or NatafTransformer will be
    used depending on if the ParameterSpace has a non-unity correlation matrix.
    :param comparison: Boolean-comparison operator. Should generally be either `np.less` or `np.less_equal`, depending
    on whether the calculated probability is defined as :math:`$P(Y<y)$` or :math:`$P(Y \leq y)$`. By default, it uses
    `np.less_equal`to match the CDF definition but for reliability analysis use case, using `np.less` might be more
    appropriate. In reality, since :math:`$P(Y=y) = 0$` for continuous Y, this is not expected to have a significant
    effect.
    """

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
    comparison: Callable[
        [np.ndarray | float, np.ndarray | float], np.ndarray | float
    ] = np.less_equal

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

    def __init__(self, settings: ImportanceSamplingSettings | None = None) -> None:
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
        mpps, history_x, history_y = find_most_probable_boundary_points(
            envelope,
            space.dimensions,
            n_search=self.settings.n_searches,
            n_jobs=self.settings.n_jobs,
            cache=cache,
        )

        if mpps.shape[0] == 0:
            # following is already in the history if caching is enabled
            cur, _, _ = envelope(np.zeros((1, space.dimensions)))
            probability = 1.0 if self.settings.comparison(cur, 0.0) else 0.0
            return probability, 0.0, (history_x, history_y)

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

        results = utils.single_or_multiprocess(
            mpps, for_loop_body, self.settings.n_jobs
        )

        probabilities = np.empty(0)
        for result in results:
            cur_probs, cur_hist_x, cur_hist_y = result
            probabilities = np.append(probabilities, cur_probs)
            history_x, history_y = utils.extend_cache(
                history_x,
                history_y,
                cur_hist_x,
                cur_hist_y,
                cache_x=cache,
                cache_y=cache,
            )

        probability = probabilities.mean()
        std_err = np.std(probabilities, ddof=1) / np.sqrt(probabilities.shape[0])
        return probability, std_err, (history_x, history_y)


def find_most_probable_boundary_points(
    envelope: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_dim: int,
    n_search: int = 12,
    n_jobs: int = -1,
    cache: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Find the zero-crossings that are closest to the origin.

    :param envelope: The function to search for the zero-crossings
    :param n_dim: number of dimensions
    :param n_search: number of searches, i.e. restarts of the optimization from a different starting point (Default=12).
    :param n_jobs: number of jobs to compute in parallel. If -1 (default), it will be equal to the number of cpus.
    :param cache: If True (default), the history of inputs and outputs are returned as well
    :return: mpps and the history of inputs and outputs if `cache=True`
    """

    lim = 7
    bounds = [(-lim, lim) for _ in range(n_dim)]
    x_starts = np.zeros((1, n_dim))
    if n_search > 1:
        designer = orthogonal_sampling.OrthogonalSamplingDesigner()
        additional = designer.design(
            variable.ParameterSpace([stats.uniform(-2, 4) for _ in range(n_dim)]),
            n_search - 1,
            steps=1,
        )
        x_starts = np.append(x_starts, additional, axis=0)

    def for_loop_body(x):
        return _find_mpp(envelope, x, bounds=bounds)

    results = utils.single_or_multiprocess(x_starts, for_loop_body, n_jobs=n_jobs)
    history_x, history_y, mpps = None, None, []
    for result in results:
        history_x, history_y = utils.extend_cache(
            history_x, history_y, result[1], result[2], cache_x=cache, cache_y=cache
        )
        if result[0] is not None:
            mpps.append(result[0])

    if mpps:
        mpps = np.array(mpps)
    else:
        mpps = np.empty((0, x_starts.shape[1]))
    return mpps, history_x, history_y


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
    envelope: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    x_start: np.ndarray,
    bounds=None,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Get MPP using SLSQP and if that fails, use the slower cobyla"""

    history_x, history_y = [], []

    def optimization_envelope(x: np.ndarray) -> np.ndarray:
        x = np.array(x)
        if x.ndim < 2:
            x = x.reshape((1, -1))
        result, hist_x, hist_y = envelope(x)
        history_x.append(hist_x)
        history_y.append(hist_y)
        return result

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

    constraints = {"type": "eq", "fun": optimization_envelope}

    try:
        res = call_optimizer(method="SLSQP")
    except ValueError:
        pass
    else:
        success = res.get("status") not in [5, 6] and res.success
        if success:
            history_x = np.array(history_x).reshape((-1, x_start.size))
            history_y = np.array(history_y).reshape((history_x.shape[0], -1))
            return res.get("x"), history_x, history_y

    constraints = (
        {"type": "ineq", "fun": optimization_envelope},
        {"type": "ineq", "fun": lambda x: -optimization_envelope(x)},
    )
    res = call_optimizer(method="COBYLA")

    history_x = np.array(history_x).reshape((-1, x_start.size))
    history_y = np.array(history_y).reshape((history_x.shape[0], -1))

    if res.success:
        return res.get("x"), np.vstack(history_x), np.vstack(history_y)
    return None, np.vstack(history_x), np.vstack(history_y)
