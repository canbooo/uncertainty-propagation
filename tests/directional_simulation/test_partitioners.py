import numpy as np
import pytest
from scipy.spatial import distance

import uncertainty_propagation.directional_simulation.partitioners as module_under_test


@pytest.fixture(params=[2, 5])
def dimensions(request):
    return request.param


@pytest.fixture(params=[40, 80])
def directions(request, dimensions):
    return dimensions * request.param


def test_random_directions(directions, dimensions):
    result = module_under_test.random_directions(directions, dimensions)
    assert result.shape == (directions, dimensions)
    assert np.isclose(np.linalg.norm(result, axis=1), 1).all()


def test_heuristic_fekete_solver(directions, dimensions):
    result = module_under_test.heuristic_fekete_solver(
        directions, dimensions, seed=1337, max_steps=10
    )

    assert result.shape == (directions, dimensions)
    assert np.isclose(np.linalg.norm(result, axis=1), 1).all()

    worse_result = module_under_test.random_directions(directions, dimensions)

    min_dist_c = distance.pdist(result, metric="cosine").min()
    min_dist_r = distance.pdist(result, metric="euclidean").min()

    min_dist_c_worse = distance.pdist(worse_result, metric="cosine").min()
    min_dist_r_worse = distance.pdist(worse_result, metric="euclidean").min()

    assert min_dist_c > min_dist_c_worse
    assert min_dist_r > min_dist_r_worse


def test_fekete_directions(directions, dimensions):
    result = module_under_test.fekete_directions(directions, dimensions)
    assert result.shape == (directions, dimensions)
    assert np.isclose(np.linalg.norm(result, axis=1), 1).all()


def test_iterative_fekete_solver(directions, dimensions):
    result = module_under_test.iterative_fekete_solver(
        directions, dimensions, seed=1337, max_steps=10
    )

    assert result.shape == (directions, dimensions)
    assert np.isclose(np.linalg.norm(result, axis=1), 1).all()

    worse_result = module_under_test.random_directions(directions, dimensions)

    min_dist_c = distance.pdist(result, metric="cosine").min()
    min_dist_r = distance.pdist(result, metric="euclidean").min()

    min_dist_c_worse = distance.pdist(worse_result, metric="cosine").min()
    min_dist_r_worse = distance.pdist(worse_result, metric="euclidean").min()

    assert min_dist_c > min_dist_c_worse
    assert min_dist_r > min_dist_r_worse
