import pytest
from experiment_design import variable
from scipy import stats


@pytest.fixture(params=[3, 2, 0, -1])
def linear_beta(request):
    return request.param


@pytest.fixture
def std_norm_parameter_space():
    return variable.ParameterSpace([stats.norm() for _ in range(2)])


@pytest.fixture
def std_norm_10d_parameter_space():
    return variable.ParameterSpace([stats.norm() for _ in range(10)])


@pytest.fixture
def non_norm_parameter_space(request):
    dists = [stats.gumbel_l(loc=1, scale=2), stats.weibull_min(1.79)]
    return variable.ParameterSpace(dists)


@pytest.fixture
def correlated_norm_parameter_space():
    return variable.ParameterSpace([stats.norm() for _ in range(2)], correlation=0.5)
