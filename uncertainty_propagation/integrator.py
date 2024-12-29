import abc
import dataclasses
import os
from typing import Any, Callable, Type

import joblib
import numpy as np
from experiment_design import variable
from scipy import stats

from uncertainty_propagation import transform


@dataclasses.dataclass
class IntegrationResult:
    probability: float
    standard_error: float
    input_history: np.ndarray | None
    output_history: np.ndarray | None

    def __post_init__(self):
        if (
            self.input_history is not None
            and self.output_history is not None
            and self.input_history.shape[0] != self.output_history.shape[0]
        ):
            raise ValueError("Inconsistent shapes of input and output histories!")

    @property
    def safety_index(self) -> float:
        return -stats.norm.ppf(self.probability)


class ProbabilityIntegrator(abc.ABC):
    use_standard_normal_space: bool = True

    def __init__(
        self,
        transformer_cls: Type[transform.StandardNormalTransformer] | None = None,
    ) -> None:
        self.transformer_cls = transformer_cls

    def calculate_probability(
        self,
        space: variable.ParameterSpace,
        propagate_through: (
            Callable[[np.ndarray], np.ndarray]
            | list[Callable[[np.ndarray], np.ndarray]]
        ),
        limit: int | float = 0,
        cache: bool = False,
    ) -> IntegrationResult:
        """
        Given the parameter space and the function(s) to propagate through the uncertainty, computes the probability
        of exceeding the limit.

        :param space: Parameter space describing the uncertainty of parameters
        :param propagate_through: Function(s) to propagate the uncertainty of the inputs that will be evaluated.
        In case multiple functions are passed as propagate_through, this computation will consider the lower envelope,
        i.e. the minimum of all functions, thus yielding a series system in reliability engineering use case. If you
        want to compute individual failure probabilities to, e.g. to simulate a parallel system, you need to call
        this method with each function separately and take the minimum of the probabilities afterward.
        :param cache: if True, track the used samples and the corresponding outputs. The outputs belong to the
        used envelope and the individual outputs are not tracked.
        :param limit: the CDF of the ParameterSpace will be evaluated at this value

        :return: estimated probability and the standard error of the estimate as well as arrays of evaluated inputs
        and the corresponding outputs if `cache=True`.
        """
        envelope = transform_to_zero_centered_envelope(propagate_through, limit)
        if self.use_standard_normal_space:
            transformer = _initialize(self.transformer_cls, space)
            envelope = transform_to_standard_normal_envelope(envelope, transformer)

        probability, std_error, cached = self._calculate_probability(
            space, envelope, cache
        )
        return IntegrationResult(
            probability=probability,
            standard_error=std_error,
            input_history=cached[0],
            output_history=cached[1],
        )

    @abc.abstractmethod
    def _calculate_probability(
        self,
        space: variable.ParameterSpace,
        envelope: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
        cache: bool = False,
    ) -> tuple[float, float, tuple[np.ndarray | None, np.ndarray | None]]:
        raise NotImplementedError


def _initialize(
    transformer_cls: Type[transform.StandardNormalTransformer] | None,
    space: variable.ParameterSpace,
) -> transform.StandardNormalTransformer:
    if transformer_cls is not None:
        return transformer_cls(space)
    if np.isclose(space.correlation, np.eye(space.dimensions)).all():
        return transform.InverseTransformSampler(space)
    return transform.NatafTransformer(space)


def transform_to_zero_centered_envelope(
    propagate_through: (
        Callable[[np.ndarray], np.ndarray] | list[Callable[[np.ndarray], np.ndarray]]
    ),
    limit: int | float,
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Given function(s) to propagate through the uncertainty, center their lower envelope to limit, i.e. if
    the min(func(x) for func in propagate_through) is equal to limit, envelope(x) is equal to 0.
    """
    if not isinstance(propagate_through, list):
        propagate_through = [propagate_through]

    def zero_centered_envelope(
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = np.ones((x.shape[0], len(propagate_through)))
        for i_col, fun in enumerate(propagate_through):
            y[:, i_col] = fun(x).reshape(-1)
        return np.min(y, axis=1) - limit, x, y

    return zero_centered_envelope


def transform_to_standard_normal_envelope(
    envelope: Callable[[np.ndarray], Any],
    transformer: transform.StandardNormalTransformer,
) -> Callable[[np.ndarray], Any]:
    """Given a function, construct a new one that accepts inputs from standard normal space and converts them to
    original space before passing them to the original function.
    """

    def standard_normal_envelope(u: np.ndarray) -> Any:
        x = transformer.inverse_transform(u)
        return envelope(x)

    return standard_normal_envelope


def _parallel_processed_envelope(
    envelope: Callable[[np.ndarray], Any], n_jobs: int | None
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if n_jobs is None:
        n_jobs = os.cpu_count()

    def parallel_envelope(x: np.ndarray):
        block_size, rem = divmod(x.shape[0], n_jobs)
        slices = [slice(i * block_size, (i + 1) * block_size) for i in range(n_jobs)]
        if rem > 0:
            slices.append(slice(n_jobs * block_size, None))
        all_results = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(envelope)(x[sli]) for sli in slices
        )
        result = np.hstack([r[0] for r in all_results])
        x_ = np.vstack([r[1] for r in all_results])
        y_ = np.vstack([r[2] for r in all_results])
        return result, x_, y_

    return parallel_envelope
