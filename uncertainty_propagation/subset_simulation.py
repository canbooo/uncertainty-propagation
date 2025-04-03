import dataclasses
from typing import Any, Callable, Type

import numpy as np
from experiment_design import orthogonal_sampling, variable
from experiment_design.experiment_designer import ExperimentDesigner
from scipy import signal, stats

from uncertainty_propagation import utils
from uncertainty_propagation.integrator import ProbabilityIntegrator
from uncertainty_propagation.transform import StandardNormalTransformer


@dataclasses.dataclass
class SubsetSimulationSettings:
    max_subsets: int = 16
    samples_per_chain: int = 1024
    min_subset_probability: float = 0.1
    covariate_correction: bool = True
    mcmc_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {"initial_std_dev": 1.0, "adaptation_frequency": 0.1}
    )
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


class SubsetSimulation(ProbabilityIntegrator):
    """
    Subset simulation for probability computation. Equations numbers in comments refer to equations from the paper:
    I. Papaioannou et al. (2015). "MCMC algorithms for Subset Simulation"
    """

    def __init__(self, settings: SubsetSimulationSettings | None = None) -> None:
        if settings is None:
            settings = SubsetSimulationSettings()
        self.settings = settings
        super(SubsetSimulation, self).__init__(self.settings.transformer_cls)

    def _calculate_probability(
        self,
        space: variable.ParameterSpace,
        envelope: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
        cache: bool = False,
    ) -> tuple[float, float, tuple[np.ndarray | None, np.ndarray | None]]:
        sample_dists = [stats.norm() for _ in range(space.dimensions)]
        inputs = self.settings.sample_generator.design(
            variable.ParameterSpace(sample_dists),
            self.settings.samples_per_chain,
            **self.settings.sample_generator_kwargs,
        )
        outputs, history_x, history_y = envelope(inputs)
        if not cache:
            history_x, history_y = None, None
        n_seeds = int(
            np.ceil(
                self.settings.samples_per_chain * self.settings.min_subset_probability
            )
        )
        ids = _next_seed_ids(n_seeds, outputs)
        current_limit = float(outputs[ids[-1]])
        if self.settings.comparison(current_limit, 0.0):
            # Assumption: sufficient samples (>= settings.min_subset_probability) in the integration area so we can
            # already compute the result using the Monte Carlo estimate
            indicators = self.settings.comparison(outputs, 0.0)
            probability = float(np.mean(indicators))

            if probability > 1 - self.settings.min_subset_probability:
                # Handle edge case where we have a probability, that is larger than 1 - settings.min_subset_probability
                # in this case, our assumption is correct but not sufficient for an accurate estimate.
                # We solve the inverse problem to increase the expected accuracy
                def inverse_envelope(
                    x: np.ndarray,
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                    y, h_x, h_y = envelope(x)
                    return -1.0 * y, h_x, h_y

                inverse_probability, std_err, (hist_x, hist_y) = (
                    self._calculate_probability(space, inverse_envelope, cache=cache)
                )
                history_x, history_y = utils.extend_cache(
                    history_x, history_y, hist_x, hist_y, cache_x=cache, cache_y=cache
                )
                return (1 - inverse_probability), std_err, (history_x, history_y)

            std_err = np.sqrt(probability * (1 - probability) / inputs.shape[0])
            return probability, std_err, (history_x, history_y)

        alphas = []
        n_samples_per_subset = self.settings.samples_per_chain * n_seeds
        probabilities = [np.mean(self.settings.comparison(outputs, current_limit))]
        deltas = [
            _subset_coefficient_of_variation(
                probabilities[-1], n_samples_per_subset, gamma=0.0
            )
        ]
        lambda_iter = 0.6  # recommended initial value for lambda
        for i_subset in range(1, self.settings.max_subsets):
            inputs, outputs, lambda_iter, hist_x, hist_y = (
                parallel_adaptive_conditional_sampling(
                    envelope,
                    inputs[ids],
                    outputs[ids],
                    self.settings.samples_per_chain,
                    current_limit,
                    lambda_iter=lambda_iter,
                    **self.settings.mcmc_kwargs,
                )
            )
            history_x, history_y = utils.extend_cache(
                history_x, history_y, hist_x, hist_y, cache_x=cache, cache_y=cache
            )

            ids = _next_seed_ids(n_seeds, outputs)
            # We need to have indicators with the shape (samples, chains) so
            # we do not assign the raveled version yet
            next_limit = max(float(outputs.ravel()[ids[-1]]), 0.0)
            indicators = np.asarray(
                self.settings.comparison(outputs, next_limit), dtype=int
            )
            # Now that we have computed the indicator, we can ravel an assign them
            inputs = inputs.reshape((-1, space.dimensions))
            outputs = outputs.ravel()

            probabilities.append(indicators.mean())
            if self.settings.covariate_correction:
                # A. Abdollahi et al. (2020). "A refined subset simulation for reliability analysis using the subset
                # control variate"
                # Eq. 18
                old_probabilities = (outputs < current_limit).mean()
                if old_probabilities == 0:
                    alphas.append(1.0)
                else:
                    alphas.append(probabilities[-1] / old_probabilities)
            current_limit = next_limit
            gamma = _correlation_factor_gamma(indicators, n_seeds)
            deltas.append(
                _subset_coefficient_of_variation(
                    probabilities[-1], n_samples_per_subset, gamma
                )
            )
            if current_limit <= 0:
                break

        deltas = np.array(deltas)
        if self.settings.covariate_correction:
            # A. Abdollahi et al. (2020). "A refined subset simulation for reliability analysis using the subset
            # control variate"
            # Eq. 18
            probability = np.prod(alphas) * probabilities[0]
        else:
            probability = np.prod(probabilities)

        # Technically, std_dev and thus error should be smaller if covariate_correction is set to True so
        # the following is an upper bound approximation at most
        std_dev = np.sum(deltas**2) * probability
        std_err = std_dev / np.sqrt(
            i_subset * n_samples_per_subset + self.settings.samples_per_chain
        )
        return probability, std_err, (history_x, history_y)


def _next_seed_ids(n_seeds: int, outputs: np.ndarray) -> np.ndarray:
    return np.argsort(outputs, axis=None)[:n_seeds]


def _subset_coefficient_of_variation(
    probability: float, n_samples: int, gamma: float = 0.0
) -> float:
    if probability > 0:
        # Eq. 9 in I. Papaioannou et al. (2015)
        return np.sqrt((1 - probability) * (1 + gamma) / n_samples / probability)
    return np.inf


def _correlation_factor_gamma(indicator: np.ndarray, probability: float) -> float:
    """
    S.K. Au and J. L. Beck (2001). "Estimation of small failure probabilities in high dimensions by subset
    simulation"
    """
    samples_per_chain, n_chains = indicator.shape
    samples = samples_per_chain * n_chains
    # weird indexing coming from the definition of how signal.correlate computes lags
    corr = signal.correlate(indicator, indicator)[samples_per_chain - 1 :, n_chains - 1]
    corr = (
        1 / (samples - np.arange(samples_per_chain) * n_chains) * corr - probability**2
    )  # Eq. 29
    rho = corr[1:] / corr[0]  # Eq. 25
    gamma = 2 * np.sum(
        (1 - np.arange(1, samples_per_chain) * n_chains / samples) * rho
    )  # Eq. 28
    return gamma


def parallel_adaptive_conditional_sampling(
    envelope,
    seeds,
    seed_outputs,
    n_samples_per_chain,
    limit_value,
    lambda_iter: float = 0.6,
    initial_std_dev: float | None = None,
    adaptation_frequency: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    r"""
    A trajectory parallelized implementation of adaptive conditional sampling. Equations numbers in comments refer
    to equations from the original paper:
    I. Papaioannou et al. (2015). "MCMC algorithms for Subset Simulation"

    Also proof checked against the implementation from the same authors at
    https://www.cee.ed.tum.de/era/software/reliability/subset-simulation/

    :param envelope: Function to propagate through the uncertainty
    :param seeds: Initial samples of the MCMC chains with shape (n_chains, n_dims)
    :param seed_outputs: Propagated values of the seeds (one value for each seed).
    :param n_samples_per_chain: Number of samples to generate per chain
    :param limit_value: Current limit used by the subset simulation, in general :math:`\geq 0`
    :param lambda_iter: Initial value of the scaling parameter :math:`lambda` in Eqs. 23-26
    :param initial_std_dev: Initial standard deviations of normal distributions to sample from, each with the mean
        equal to one of the seeds.
    :param adaptation_frequency: Determines update frequency of the acceptance rate as well as the scaling parameter.
        corresponds to :math:`p_a` in the cited work. Author recommend using :math:`0.1 \leq p_a \leq 0.2`.
    :return: Generated inputs and outputs in the standard normal space, the updated lambda_iter as well as the
        history inputs and outputs
    """
    n_chains, n_dims = seeds.shape
    inputs = np.zeros((n_samples_per_chain + 1, n_chains, n_dims))
    outputs = np.zeros((n_samples_per_chain + 1, n_chains))
    inputs[0] = seeds
    outputs[0] = seed_outputs.ravel()
    accepts = np.zeros_like(outputs)
    n_adaptations = int(adaptation_frequency * n_samples_per_chain)

    a_star = 0.44  # Eq. 26
    if initial_std_dev is None:
        stds = np.std(seeds, axis=0, ddof=1)  # Step 1.b, Eqs. 27-28
    else:
        stds = np.ones(n_dims)  # Step 1.a
        if initial_std_dev is not None and not np.any(initial_std_dev == 0):
            stds *= initial_std_dev  # Another alternative
    stds = np.repeat(stds.reshape((1, -1)), n_chains, 0)

    def update_sigma_rho() -> tuple[float, float]:
        # Step 3.a
        sigma_i = np.minimum(lambda_iter * stds, np.ones_like(stds))  # Eq. 23
        rho_i = np.sqrt(1 - sigma_i**2)  # Eq. 24
        return sigma_i, rho_i

    sigma, rho = update_sigma_rho()
    history_x, history_y = None, None
    for i_sample in range(n_samples_per_chain):
        # Step 3.b generate samples and decide whether to accept them
        candidates = stats.norm(loc=rho * inputs[i_sample], scale=sigma).rvs()
        candidate_outputs, hist_x, hist_y = envelope(candidates)
        history_x, history_y = utils.extend_cache(
            history_x, history_y, hist_x, hist_y, cache_x=True, cache_y=True
        )
        improvements = candidate_outputs <= limit_value
        inputs[i_sample + 1] = np.where(
            improvements.reshape((-1, 1)), candidates, inputs[i_sample]
        )
        outputs[i_sample + 1] = np.where(
            improvements, candidate_outputs, outputs[i_sample]
        )
        accepts[i_sample + 1] = improvements

        i_adapt, remainder = divmod(i_sample, n_adaptations)
        if i_sample and not remainder:  #
            # Step 3.c evaluate average acceptance rate
            mu_acc = np.mean(accepts[-n_adaptations:])  # Eq. 25
            # Step 3.d compute new scaling parameter
            zeta = 1 / np.sqrt(i_adapt + 1)
            lambda_iter = np.exp(
                np.log(lambda_iter) + zeta * (mu_acc - a_star)
            )  # Eq. 26
            sigma, rho = update_sigma_rho()

    return inputs[1:], outputs[1:], lambda_iter, history_x, history_y
