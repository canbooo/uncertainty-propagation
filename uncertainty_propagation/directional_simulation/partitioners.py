import functools
import itertools
import logging
import warnings
from typing import Protocol

import numpy as np
from scipy import stats
from scipy.spatial import distance

from uncertainty_propagation import utils


class DirectionGenerator(Protocol):
    def __call__(self, n_dir: int, n_dim: int) -> np.ndarray: ...


def random_directions(n_dir: int, n_dim: int) -> np.ndarray:
    """
    Generate directions using random sampling

    :param n_dir: Number of directions to generate
    :param n_dim: Input space dimension size
    :return: Array of directions with shape (n_dir, n_dim), where each direction has the length 1
    """
    std_norm_points = stats.norm.rvs(size=(n_dir, n_dim))
    return std_norm_points / np.linalg.norm(std_norm_points, axis=1, keepdims=True)


@functools.lru_cache(maxsize=1)
def fekete_directions(
    n_dir: int,
    n_dim: int,
    n_starts: int = 4,
    max_steps_per_solution: int = 500,
    convergence_tolerance: float = 1e-18,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Generate directions using a heuristic solution of the Fekete problem.
    See https://en.wikipedia.org/wiki/Fekete_problem for a brief introduction, although note that we use cosine
    distance as the energy computation here instead of the l2-norm.

    :param n_dir: Number of directions to generate
    :param n_dim: Input space dimension size
    :param n_starts: Number of restarts for the optimization.
    :param max_steps_per_solution: Maximum number of optimization steps used for each start
    :param convergence_tolerance: Optimization tolerance for stagnation i.e. a lack of change in the results
    :param n_jobs: Number of jobs to compute in parallel. Use 1 to deactivate multiprocessing.
    :return: Array of directions with shape (n_dir, n_dim), where each direction has the length 1
    """
    if n_dim == 1:
        return np.array([-1, 1])
    if n_dim == 2:
        phi = np.linspace(0, 2 * np.pi, n_dir)
        return np.c_[np.cos(phi), np.sin(phi)]

    def for_loop_body(seed):
        cur_dir = heuristic_fekete_solver(
            n_dir, n_dim, max_steps_per_solution, convergence_tolerance, seed=seed
        )
        cur_dist = np.min(distance.pdist(cur_dir, metric="cosine"))
        return cur_dist, cur_dir

    seeds = stats.randint.rvs(69, 420_000_000, size=n_starts)
    results = utils.single_or_multiprocess(seeds, for_loop_body, n_jobs=n_jobs)
    # return result with the maximum min pairwise distance
    return sorted(results, key=lambda x: x[0])[-1][1]


def heuristic_fekete_solver(
    n_dir: int,
    n_dim: int,
    max_steps: int = 500,
    convergence_tolerance: float = 1e-18,
    min_delta: float = 1e-12,
    seed: int = 0,
) -> np.ndarray:
    """
    Calculates Fekete points heuristically using a zero centered repulsive power

    :param n_dir: Number of directions to generate
    :param n_dim: Input space dimension size
    :param max_steps: Maximum number of optimization steps
    :param convergence_tolerance: Optimization tolerance for stagnation i.e. a lack of change in the results
    :param min_delta: minimum displacement of simulated particles
    :param seed: random seed that will be set before generating random initial points. Required to ensure randomness
        is preserved in multiprocessing setting, where different seeds for each process must be passed.
    :return: Array of directions with shape (n_dir, n_dim), where each direction has the length 1
    """
    np.random.seed(seed)
    base_points = tmp_points = random_directions(n_dir, n_dim)
    dist_min = np.min(distance.pdist(base_points, metric="cosine"))
    for k in range(3):
        candidates = random_directions(n_dir, n_dim)
        dist_cand = np.min(distance.pdist(base_points, metric="cosine"))
        if dist_min < dist_cand:
            base_points = tmp_points = candidates
            dist_min = dist_cand

    diagonal_ids = np.arange(n_dir)
    for i_iter in range(max_steps):
        motions = distance.pdist(base_points, metric="cosine")
        dist_min = np.min(motions)
        motions = distance.squareform(motions ** (-i_iter / 2 - 1))
        motions = base_points[:, np.newaxis, :] * (motions[:, :, np.newaxis])
        motions = motions.sum(0) - motions[diagonal_ids, diagonal_ids, :]
        # subtraction removes the diagonals from the distance matrix
        # which would have infinite magnitude
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            motions /= np.linalg.norm(motions, axis=1, keepdims=True)
        dist_min_tmp = 0
        delta_cur = 1.0
        # above is reasonable since generally step size
        # should be decreasing, but we look for a better starting from
        # slightly larger value as the last one
        motions = base_points - motions
        while dist_min_tmp <= dist_min and delta_cur > min_delta:
            tmp_points = base_points + delta_cur * motions
            tmp_points /= np.linalg.norm(tmp_points, axis=1, keepdims=True)
            dist_min_tmp = np.min(distance.pdist(tmp_points, metric="cosine"))
            delta_cur *= 0.9

        if dist_min_tmp < dist_min:
            logging.info(
                f"No more improvement could be achieved with Fekete points after {i_iter} iterations."
            )
            break

        base_points = tmp_points.copy()
        if abs(dist_min_tmp - dist_min) < convergence_tolerance:
            logging.info(
                f"Fekete points converged.  Change in min.u dist: {abs(dist_min_tmp - dist_min)}"
            )
            break
    return base_points


def iterative_fekete_solver(
    n_dir: int,
    n_dim: int,
    max_steps: int = 1000,
    convergence_tolerance: float = 1e-18,
    min_delta: float = 1e-12,
    n_neighbours: int = -1,
    seed: int = 0,
) -> np.ndarray:
    """
    Calculates Fekete points using pairwise repulsive forces

    :param n_dir: Number of directions to generate
    :param n_dim: Input space dimension size
    :param max_steps: Maximum number of optimization steps
    :param convergence_tolerance: Optimization tolerance for stagnation i.e. a lack of change in the results
    :param min_delta: minimum displacement of simulated particles
    :param n_neighbours: number of neighbours that apply a force
    :param seed: random seed that will be set before generating random initial points. Required to ensure randomness
    is preserved in multiprocessing setting, where different seeds for each process must be passed.
    :return: Array of directions with shape (n_dir, n_dim), where each direction has the length 1
    """
    np.random.seed(seed)
    base_points = tmp_points = random_directions(n_dir, n_dim)
    dist_min = np.min(distance.pdist(base_points, metric="cosine"))
    for k in range(3):
        candidates = random_directions(n_dir, n_dim)
        dist_cand = np.min(distance.pdist(base_points, metric="cosine"))
        if dist_min < dist_cand:
            base_points = tmp_points = candidates
            dist_min = dist_cand

    pdist_ids = np.array(list(itertools.combinations(np.arange(n_dir), 2)))

    def top_k_force(i_dir: int):
        matches = pdist_ids == i_dir
        mask = matches.any(1)
        other_columns = np.where(np.logical_not(matches[mask]))[1]
        other_point_ids = pdist_ids[mask, other_columns]
        cur_distances = distances[mask]
        top_k = np.argsort(cur_distances)[:n_neighbours]
        force_directions = base_points[[i_dir]] - base_points[other_point_ids[top_k]]
        force_directions /= np.linalg.norm(force_directions)
        final_force = np.sum(
            cur_distances[top_k, np.newaxis] ** -2 * force_directions, axis=0
        )
        return final_force

    for i_iter in range(max_steps):
        distances = distance.pdist(base_points, metric="cosine")
        dist_min = np.min(distances)
        forces = np.array([top_k_force(i_dir) for i_dir in range(n_dir)])
        # print(i_iter, dist_min)
        # above is reasonable since generally step size
        # should be decreasing, but we look for a better starting from
        # slightly larger value as the last one
        delta_cur = 1.0
        dist_min_tmp = 0.0
        while dist_min_tmp <= dist_min and delta_cur > min_delta:
            tmp_points = base_points + delta_cur * forces
            tmp_points /= np.linalg.norm(tmp_points, axis=1, keepdims=True)
            dist_min_tmp = np.min(distance.pdist(tmp_points, metric="cosine"))
            delta_cur *= 0.9

        if dist_min_tmp < dist_min:
            logging.info(
                f"No more improvement could be achieved with Fekete points after {i_iter} iterations."
            )
            break

        base_points = tmp_points.copy()
        if abs(dist_min_tmp - dist_min) < convergence_tolerance:
            logging.info(
                f"Fekete points converged.  Change in min. dist: {abs(dist_min_tmp - dist_min)}"
            )
            break
    return base_points
