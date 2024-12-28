import dataclasses
from typing import Any, Callable

import numpy as np
from experiment_design import random_sampling, variable
from experiment_design.experiment_designer import ExperimentDesigner

from uncertainty_propagation import integrator, utils


@dataclasses.dataclass
class MonteCarloSimulatorSettings:
    probability_tolerance: float = 1e-4
    batch_size: int = 100_000
    target_variation_coefficient: float = 0.1
    chebyshev_confidence_level: float = 0  # Eq. 2.100
    early_stopping: bool = True
    sample_generator: ExperimentDesigner = random_sampling.RandomSamplingDesigner(
        exact_correlation=False
    )
    sample_generator_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {"steps": 1}
    )
    sample_limit: int = dataclasses.field(init=False)

    def __post_init__(self):
        sample_limit = (
            self.target_variation_coefficient**-2 / self.probability_tolerance
        )
        sample_limit /= 1 - self.chebyshev_confidence_level
        self.sample_limit = int(np.ceil(sample_limit))
        if self.batch_size < 1 or self.batch_size > self.sample_limit:
            self.batch_size = self.sample_limit


class MonteCarloSimulation(integrator.ProbabilityIntegrator):
    """
    Monte Carlo simulation for the probability integration. See Chapter 2.3.1 for equation references in this file
    https://hss-opus.ub.ruhr-uni-bochum.de/opus4/frontdoor/deliver/index/docId/9143/file/diss.pdf
    """

    use_standard_normal_space: bool = False
    use_multiprocessing: bool = False

    def __init__(self, settings: MonteCarloSimulatorSettings):
        self.settings = settings
        super().__init__()

    def _calculate_probability(
        self,
        space: variable.ParameterSpace,
        envelope: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
        cache: bool = False,
    ) -> tuple[float, float, tuple[np.ndarray, np.ndarray] | None]:
        total_samples = 0
        history_x, history_y = None, None
        probability = 0.0
        while total_samples < self.settings.sample_limit:
            batch_size = min(
                self.settings.batch_size, self.settings.sample_limit - total_samples
            )
            x = self.settings.sample_generator.design(
                space,
                batch_size,
                old_sample=history_x,
                **self.settings.sample_generator_kwargs,
            )
            y_min, x_to_cache, y_to_cache = envelope(x)
            history_x, history_y = utils.extend_cache(
                history_x,
                history_y,
                x_to_cache,
                y_to_cache,
                cache_x=True,
                cache_y=cache,
            )
            probability = probability * total_samples + (
                y_min <= 0
            )  # TODO: parametrize comparison
            total_samples += batch_size
            probability /= total_samples
            if self.settings.early_stopping and probability > 0:
                cov = np.sqrt(
                    (1 - probability) / probability / total_samples
                )  # estimate CoV using 2.80 from
                if cov <= self.settings.target_variation_coefficient:
                    break
        std_err = probability * (1 - probability) / total_samples
        return probability, std_err, (history_x, history_y)
