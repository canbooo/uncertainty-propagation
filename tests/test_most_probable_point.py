import functools

import numpy as np

import uncertainty_propagation.most_probable_point as module_under_test
from tests import reliability_test_functions
from tests.shared_fixtures import *  # noqa: F403


def test_test_find_most_probable_points():
    np.random.seed(42)

    def reshaped(fun):
        def inner(x):
            x = np.array(x)
            if x.ndim < 2:
                x = x.reshape((1, -1))
            y = fun(x)
            return y, x, y

        return inner

    target_fun = reshaped(functools.partial(reliability_test_functions.linear, beta=6))
    for dim in [2, 10, 50]:
        solutions, history_x, history_y = (
            module_under_test.find_most_probable_boundary_points(
                target_fun, n_dim=dim, n_search=16, n_jobs=1
            )
        )

        assert np.max(np.std(solutions, axis=1)) < 1e-4
        assert np.isclose(np.sqrt(np.sum(solutions[0] ** 2)), 6)

    solutions, history_x, history_y = (
        module_under_test.find_most_probable_boundary_points(
            target_fun, n_dim=2, n_search=16, n_jobs=-1
        )
    )

    assert np.max(np.std(solutions, axis=1)) < 1e-4
    assert np.isclose(np.sqrt(np.sum(solutions[0] ** 2)), 6)

    target_fun = reshaped(functools.partial(reliability_test_functions.linear, beta=15))
    solutions, history_x, history_y = (
        module_under_test.find_most_probable_boundary_points(
            target_fun, n_dim=2, n_search=16, n_jobs=1
        )
    )

    assert solutions.shape[0] == 0

    target_fun = reshaped(functools.partial(reliability_test_functions.linear, beta=15))
    solutions, history_x, history_y = (
        module_under_test.find_most_probable_boundary_points(
            target_fun, n_dim=2, n_search=4, n_jobs=2
        )
    )

    assert solutions.shape[0] == 0


class TestFirstOrderApproximation:
    @staticmethod
    def get_instance(
        settings: module_under_test.FirstOrderApproximationSettings | None = None,
    ) -> module_under_test.FirstOrderApproximation:
        return module_under_test.FirstOrderApproximation(settings)

    def test_linear_std_norm(
        self, linear_beta, std_norm_parameter_space, std_norm_10d_parameter_space
    ):
        np.random.seed(1337)
        instance = self.get_instance()
        for space in [std_norm_parameter_space, std_norm_10d_parameter_space]:
            fun = functools.partial(reliability_test_functions.linear, beta=linear_beta)
            result = instance.calculate_probability(space, fun)
            assert np.isclose(result.safety_index, linear_beta, atol=5e-2)

    def test_linear_no_mpp(self, std_norm_parameter_space):
        np.random.seed(1337)
        instance = self.get_instance()
        fun = functools.partial(reliability_test_functions.linear, beta=12)
        result = instance.calculate_probability(std_norm_parameter_space, fun)
        assert result.probability == 0.0

    def test_linear_non_norm(self, linear_beta, non_norm_parameter_space):
        np.random.seed(1337)
        instance = self.get_instance()
        fun = functools.partial(reliability_test_functions.linear, beta=linear_beta)
        result = instance.calculate_probability(non_norm_parameter_space, fun)
        expected = {3: 1.66, 2: 0.82, 0: -0.45, -1: -0.92}[linear_beta]
        assert np.isclose(result.safety_index, expected, atol=5e-2)

    def test_linear_corr(self, linear_beta, correlated_norm_parameter_space):
        np.random.seed(1337)
        instance = self.get_instance()
        fun = functools.partial(reliability_test_functions.linear, beta=linear_beta)
        result = instance.calculate_probability(correlated_norm_parameter_space, fun)
        expected = {3: 2.46, 2: 1.63, 0: 0.00, -1: -0.82}[linear_beta]
        assert np.isclose(result.safety_index, expected, atol=5e-2)

    def test_styblinski_tang(self, std_norm_parameter_space):
        np.random.seed(1337)
        settings = module_under_test.FirstOrderApproximationSettings(n_searches=32)
        instance = self.get_instance(settings)
        result = instance.calculate_probability(
            std_norm_parameter_space, reliability_test_functions.styblinski_tang
        )
        # increased tolerance due to multi-modality
        assert np.isclose(result.safety_index, 3.64, atol=1)

    def test_modified_himmblau(self, std_norm_parameter_space):
        np.random.seed(1337)
        settings = module_under_test.FirstOrderApproximationSettings(n_searches=32)
        instance = self.get_instance(settings)
        result = instance.calculate_probability(
            std_norm_parameter_space, reliability_test_functions.modified_himmblau
        )
        # increased tolerance due to multi-modality
        assert np.isclose(result.safety_index, 3.10, atol=1)


class TestImportanceSampling:
    @staticmethod
    def get_instance(
        settings: module_under_test.FirstOrderApproximationSettings | None = None,
    ) -> module_under_test.FirstOrderApproximation:
        return module_under_test.FirstOrderApproximation(settings)

    def test_linear_std_norm(
        self, linear_beta, std_norm_parameter_space, std_norm_10d_parameter_space
    ):
        np.random.seed(1337)
        instance = self.get_instance()
        for space in [std_norm_parameter_space, std_norm_10d_parameter_space]:
            fun = functools.partial(reliability_test_functions.linear, beta=linear_beta)
            result = instance.calculate_probability(space, fun)
            assert np.isclose(result.safety_index, linear_beta, atol=5e-2)

    def test_linear_no_mpp(self, std_norm_parameter_space):
        np.random.seed(1337)
        instance = self.get_instance()
        fun = functools.partial(reliability_test_functions.linear, beta=12)
        result = instance.calculate_probability(std_norm_parameter_space, fun)
        assert result.probability == 0.0

    def test_linear_non_norm(self, linear_beta, non_norm_parameter_space):
        np.random.seed(1337)
        instance = self.get_instance()
        fun = functools.partial(reliability_test_functions.linear, beta=linear_beta)
        result = instance.calculate_probability(non_norm_parameter_space, fun)
        expected = {3: 1.66, 2: 0.82, 0: -0.45, -1: -0.92}[linear_beta]
        assert np.isclose(result.safety_index, expected, atol=5e-2)

    def test_linear_corr(self, linear_beta, correlated_norm_parameter_space):
        np.random.seed(1337)
        instance = self.get_instance()
        fun = functools.partial(reliability_test_functions.linear, beta=linear_beta)
        result = instance.calculate_probability(correlated_norm_parameter_space, fun)
        expected = {3: 2.46, 2: 1.63, 0: 0.00, -1: -0.82}[linear_beta]
        assert np.isclose(result.safety_index, expected, atol=5e-2)

    def test_styblinski_tang(self, std_norm_parameter_space):
        np.random.seed(1337)
        settings = module_under_test.FirstOrderApproximationSettings(n_searches=32)
        instance = self.get_instance(settings)
        result = instance.calculate_probability(
            std_norm_parameter_space, reliability_test_functions.styblinski_tang
        )
        # increased tolerance due to multi-modality
        assert np.isclose(result.safety_index, 3.64, atol=0.2)

    def test_modified_himmblau(self, std_norm_parameter_space):
        np.random.seed(1337)
        settings = module_under_test.FirstOrderApproximationSettings(n_searches=32)
        instance = self.get_instance(settings)
        result = instance.calculate_probability(
            std_norm_parameter_space, reliability_test_functions.modified_himmblau
        )
        # increased tolerance due to multi-modality
        assert np.isclose(result.safety_index, 3.10, atol=0.5)
