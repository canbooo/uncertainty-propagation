from itertools import combinations

import numpy as np
import pytest
from experiment_design import orthogonal_sampling, variable
from scipy import stats
from scipy.stats._distn_infrastructure import rv_frozen

import uncertainty_propagation.transform as module_under_test


@pytest.fixture(params=[0, 0.1, 0.25, 0.5, 0.75])
def current_correlation_matrix(request) -> np.ndarray:
    corr = request.param
    corr_mat = np.eye(2)
    corr_mat[0, 1] = corr_mat[1, 0] = corr
    return corr_mat


@pytest.fixture
def distributions_to_test() -> list[rv_frozen]:
    return [stats.norm(5, 10), stats.uniform(-1, 1), stats.lognorm(0.3), stats.cosine()]


class TestInverseTransformSampler:

    @staticmethod
    def get_instance(
        distributions: list[rv_frozen],
    ) -> module_under_test.InverseTransformSampler:
        return module_under_test.InverseTransformSampler(
            variable.DesignSpace(variables=distributions)
        )

    def test_transform(self, distributions_to_test: list[rv_frozen]) -> None:
        np.random.seed(42)
        for dist1, dist2 in combinations(distributions_to_test, 2):
            instance = self.get_instance([dist1, dist2])
            samples = np.c_[dist1.rvs(int(1e3)), dist2.rvs(int(1e3))]
            transformed = instance.transform(samples)
            res1 = stats.cramervonmises(transformed[:, 0], stats.norm(0, 1).cdf)
            res2 = stats.cramervonmises(transformed[:, 1], stats.norm(0, 1).cdf)
            # Note that the failure of below tests does not necessarily mean broken code
            # as the test itself is not very robust
            assert res1.pvalue > 0.05, f"{dist1.dist.name} failed"
            assert res2.pvalue > 0.05, f"{dist2.dist.name} failed"

    def test_inverse_transform(self, distributions_to_test: list[rv_frozen]) -> None:
        np.random.seed(42)
        for dist1, dist2 in combinations(distributions_to_test, 2):
            instance = self.get_instance([dist1, dist2])
            samples = stats.norm(0, 1).rvs((int(1e3), 2))
            transformed = instance.inverse_transform(samples)
            res1 = stats.cramervonmises(transformed[:, 0], dist1.cdf)
            res2 = stats.cramervonmises(transformed[:, 1], dist2.cdf)
            # Note that the failure of below tests does not necessarily mean broken code
            # as the test itself is not very robust
            assert res1.pvalue > 0.05, f"{dist1.dist.name} failed"
            assert res2.pvalue > 0.05, f"{dist2.dist.name} failed"


def test_solve_nataf_normal_dist(current_correlation_matrix: np.ndarray) -> None:
    marginals = [stats.norm(0, 1) for _ in range(2)]
    correlate, uncorrelate = module_under_test.solve_nataf(
        marginals, current_correlation_matrix
    )
    assert np.isclose(correlate.T.dot(correlate), current_correlation_matrix).all()
    full_uncorrelate = uncorrelate.dot(uncorrelate.T)
    assert np.isclose(current_correlation_matrix.dot(full_uncorrelate), np.eye(2)).all()


def test_solve_nataf_non_normal_dist(
    current_correlation_matrix: np.ndarray, distributions_to_test: list[rv_frozen]
) -> None:
    for dist1, dist2 in combinations(distributions_to_test, 2):
        correlate, uncorrelate = module_under_test.solve_nataf(
            [dist1, dist2], current_correlation_matrix
        )
        full_uncorrelate = uncorrelate.dot(uncorrelate.T)
        full_correlate = correlate.T.dot(correlate)
        test_matrix = full_correlate.dot(full_uncorrelate)
        assert np.isclose(test_matrix, np.eye(2), rtol=0.0, atol=1e-5).all()


class TestNatafTransformation:

    @staticmethod
    def get_instance(
        distributions: list[rv_frozen], correlation_matrix: np.ndarray
    ) -> module_under_test.InverseTransformSampler:
        return module_under_test.NatafTransformation(
            variable.DesignSpace(variables=distributions),
            correlation_matrix,
        )

    def test_transform(
        self,
        current_correlation_matrix: np.ndarray,
        distributions_to_test: list[rv_frozen],
    ) -> None:
        np.random.seed(42)
        for dist1, dist2 in combinations(distributions_to_test, 2):
            designer = orthogonal_sampling.OrthogonalSamplingDesigner(
                target_correlation=current_correlation_matrix
            )
            doe = designer.design(
                variables=variable.create_variables_from_distributions([dist1, dist2]),
                sample_size=1000,
                steps=10,
            )
            corr_mat = current_correlation_matrix  # np.corrcoef(doe, rowvar=False)
            instance = self.get_instance([dist1, dist2], corr_mat)
            transformed = instance.transform(doe)
            if not np.isclose(
                np.corrcoef(transformed, rowvar=False), np.eye(2), rtol=5e-2, atol=1e-1
            ).all():
                assert np.isclose(
                    np.corrcoef(transformed, rowvar=False),
                    np.eye(2),
                    rtol=5e-2,
                    atol=1e-1,
                ).all()

    def test_inverse_transform(
        self,
        current_correlation_matrix: np.ndarray,
        distributions_to_test: list[rv_frozen],
    ) -> None:
        np.random.seed(42)
        for dist1, dist2 in combinations(distributions_to_test, 2):
            designer = orthogonal_sampling.OrthogonalSamplingDesigner(
                target_correlation=0
            )
            doe = designer.design(
                variables=variable.create_variables_from_distributions(
                    [stats.norm(0, 1) for _ in range(2)]
                ),
                sample_size=1000,
                steps=10,
            )
            instance = self.get_instance([dist1, dist2], current_correlation_matrix)
            untransformed = instance.inverse_transform(doe)
            assert np.isclose(
                np.corrcoef(untransformed, rowvar=False),
                current_correlation_matrix,
                rtol=5e-2,
                atol=5e-2,
            ).all()
