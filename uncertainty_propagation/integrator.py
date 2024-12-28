import abc
from typing import Any, Callable, Type

import numpy as np
from experiment_design.variable import ParameterSpace

from uncertainty_propagation.transform import (
    InverseTransformSampler,
    NatafTransformer,
    StandardNormalTransformer,
)


class ProbabilityIntegrator(abc.ABC):

    use_standard_normal_space: bool = True

    def __init__(
        self, transformer_cls: Type[StandardNormalTransformer] | None = None
    ) -> None:
        self.transformer_cls = transformer_cls

    def calculate_probability(
        self,
        space: ParameterSpace,
        propagate_through: (
            Callable[[np.ndarray], np.ndarray]
            | list[Callable[[np.ndarray], np.ndarray]]
        ),
        limit: int | float = 0,
        cache: bool = False,
    ) -> tuple[float, float] | tuple[float, float, tuple[np.ndarray, np.ndarray]]:
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
        envelope = _zero_centered_envelope(propagate_through, limit)
        if self.use_standard_normal_space:
            transformer = _initialize(self.transformer_cls, space)
            envelope = _transform_to_standard_normal_space(envelope, transformer)

        probability, std_error, cached = self._calculate_probability(
            space, envelope, cache
        )
        if cache:
            return probability, std_error, cached
        return probability, std_error

    @abc.abstractmethod
    def _calculate_probability(
        self,
        space: ParameterSpace,
        envelope: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
        cache: bool = False,
    ) -> tuple[float, float, tuple[np.ndarray, np.ndarray] | None]:
        raise NotImplementedError


def _initialize(
    transformer_cls: Type[StandardNormalTransformer] | None, space: ParameterSpace
) -> StandardNormalTransformer:
    if transformer_cls is not None:
        return transformer_cls(space)
    if np.isclose(space.correlation, np.eye(space.dimensions)).all():
        return InverseTransformSampler(space)
    return NatafTransformer(space)


def _zero_centered_envelope(
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
        y = np.ones(x.shape)
        for i_col, fun in enumerate(propagate_through):
            y[:, i_col] = fun(x)
        return np.min(y, axis=1) - limit, x, y

    return zero_centered_envelope


def _transform_to_standard_normal_space(
    envelope: Callable[[np.ndarray], Any], transformer: StandardNormalTransformer
) -> Callable[[np.ndarray], np.ndarray]:
    """Given a function, construct a new one that accepts inputs from standard normal space and converts them to
    original space before passing them to the original function.
    """

    def standard_normal_envelope(u: np.ndarray) -> np.ndarray:
        x = transformer.inverse_transform(u)
        return envelope(x)

    return standard_normal_envelope
