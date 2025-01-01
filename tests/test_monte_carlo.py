import functools

import numpy as np

import uncertainty_propagation.monte_carlo as module_under_test
from tests import reliability_test_functions
from tests.shared_fixtures import *  # noqa: F403


class TestMonteCarloSimulation:
    @staticmethod
    def get_instance(
        settings: module_under_test.MonteCarloSimulationSettings | None = None,
    ) -> module_under_test.MonteCarloSimulation:
        return module_under_test.MonteCarloSimulation(settings)

    def test_linear_std_norm(
        self, linear_beta, std_norm_parameter_space, std_norm_10d_parameter_space
    ):
        np.random.seed(1337)
        instance = self.get_instance()
        for space in [std_norm_parameter_space, std_norm_10d_parameter_space]:
            fun = functools.partial(reliability_test_functions.linear, beta=linear_beta)
            result = instance.calculate_probability(space, fun)
            assert np.isclose(result.safety_index, linear_beta, atol=1e-1)

    def test_quadratic_zero(self, std_norm_parameter_space):
        np.random.seed(1337)
        instance = self.get_instance()
        result = instance.calculate_probability(
            std_norm_parameter_space, reliability_test_functions.quadratic_greater
        )
        assert np.isclose(result.probability, 0.0, atol=1e-16)

    def test_quadratic_one(self, std_norm_parameter_space):
        np.random.seed(1337)
        instance = self.get_instance()
        result = instance.calculate_probability(
            std_norm_parameter_space, reliability_test_functions.quadratic_lesser
        )
        assert np.isclose(result.probability, 1.0)

    def test_linear_non_norm(self, linear_beta, non_norm_parameter_space):
        np.random.seed(1337)
        instance = self.get_instance()
        fun = functools.partial(reliability_test_functions.linear, beta=linear_beta)
        result = instance.calculate_probability(non_norm_parameter_space, fun)
        expected = {3: 1.66, 2: 0.82, 0: -0.45, -1: -0.92}[linear_beta]
        assert np.isclose(result.safety_index, expected, atol=1e-1)

    def test_linear_corr(self, linear_beta, correlated_norm_parameter_space):
        np.random.seed(1337)
        instance = self.get_instance()
        fun = functools.partial(reliability_test_functions.linear, beta=linear_beta)
        result = instance.calculate_probability(correlated_norm_parameter_space, fun)
        expected = {3: 2.46, 2: 1.63, 0: 0.00, -1: -0.82}[linear_beta]
        assert np.isclose(result.safety_index, expected, atol=1e-1)

    def test_styblinski_tang(
        self, std_norm_parameter_space, std_norm_10d_parameter_space
    ):
        np.random.seed(1337)
        instance = self.get_instance()
        for space in [std_norm_parameter_space, std_norm_10d_parameter_space]:
            result = instance.calculate_probability(
                space, reliability_test_functions.styblinski_tang
            )
            expected = 3.64 if space.dimensions == 2 else 3.10
            assert np.isclose(result.safety_index, expected, atol=1e-1)

    def test_modified_himmblau(self, std_norm_parameter_space):
        np.random.seed(1337)
        instance = self.get_instance()
        result = instance.calculate_probability(
            std_norm_parameter_space, reliability_test_functions.modified_himmblau
        )
        assert np.isclose(result.safety_index, 3.10, atol=1e-1)

    def test_modified_rastrigin(self, std_norm_parameter_space):
        np.random.seed(1337)
        instance = self.get_instance()
        result = instance.calculate_probability(
            std_norm_parameter_space, reliability_test_functions.modified_rastrigin
        )
        assert np.isclose(result.safety_index, 1.45, atol=1e-1)

    def test_settings(self, std_norm_parameter_space):
        settings = module_under_test.MonteCarloSimulationSettings(
            target_variation_coefficient=0.1,
            early_stopping=True,
        )
        instance = self.get_instance(settings)
        result = instance.calculate_probability(
            std_norm_parameter_space,
            functools.partial(reliability_test_functions.linear, beta=1),
            cache=True,
        )

        assert result.input_history.shape[0] < settings.sample_limit

        settings = module_under_test.MonteCarloSimulationSettings(
            probability_tolerance=1e-2,
            target_variation_coefficient=0.1,
            early_stopping=False,
        )
        instance = self.get_instance(settings)
        result = instance.calculate_probability(
            std_norm_parameter_space,
            functools.partial(reliability_test_functions.linear, beta=2),
            cache=True,
        )
        assert result.input_history.shape[0] == settings.sample_limit

        max_samples_wo_chebyshev = settings.sample_limit

        settings = module_under_test.MonteCarloSimulationSettings(
            probability_tolerance=1e-2,
            target_variation_coefficient=0.1,
            early_stopping=False,
            chebyshev_confidence_level=0.9,
        )
        instance = self.get_instance(settings)
        result = instance.calculate_probability(
            std_norm_parameter_space,
            functools.partial(reliability_test_functions.linear, beta=2),
            cache=True,
        )

        assert result.input_history.shape[0] > max_samples_wo_chebyshev
