import logging
from typing import Protocol

import numpy as np
from experiment_design import variable
from scipy import linalg, stats
from scipy.stats._distn_infrastructure import rv_frozen


class StandardNormalTransformer(Protocol):

    def transform(self, samples: np.ndarray) -> np.ndarray:
        """Transform samples from original space to standard normal space"""

    def inverse_transform(self, samples: np.ndarray) -> np.ndarray:
        """Transform samples from standard normal space to original space"""


class InverseTransformSampler:
    """
    A StandardNormalTransformer for independent variables
    """

    def __init__(self, space: variable.ParameterSpace) -> None:
        self.space = space
        self.standard_normal_space = variable.ParameterSpace(
            variable.create_variables_from_distributions(
                [stats.norm(loc=0, scale=1) for _ in space.variables]
            )
        )

    def transform(self, samples: np.ndarray) -> np.ndarray:
        """Transform samples from original space to standard normal space"""
        return self.standard_normal_space.value_of(self.space.cdf_of(samples))

    def inverse_transform(self, samples: np.ndarray) -> np.ndarray:
        """Transform samples from standard normal space to original space"""
        return self.space.value_of(self.standard_normal_space.cdf_of(samples))


class NatafTransformer(InverseTransformSampler):
    """
    A StandardNormalTransformer for linearly dependent variables (Gaussian Copula)
    """

    def __init__(self, space: variable.ParameterSpace) -> None:
        super().__init__(space)
        self.correlate_matrix, self.decorrelate_matrix = solve_gaussian_copula(
            [var.distribution for var in space.variables],
            space.correlation,
        )

    def transform(self, samples: np.ndarray) -> np.ndarray:
        """Transform samples from original space to standard normal space"""
        transformed = super().transform(samples)
        return transformed.dot(self.decorrelate_matrix)

    def inverse_transform(self, samples: np.ndarray) -> np.ndarray:
        """Transform samples from standard normal space to original space"""
        return super().inverse_transform(samples.dot(self.correlate_matrix))


def _is_normal_distribution(distribution: rv_frozen) -> bool:
    return distribution.dist.name == "norm"


def solve_gaussian_copula(
    marginal_distributions: list[rv_frozen],
    correlation_matrix: np.ndarray,
    order: int = 13,
    tolerance: float = 1e-6,
    max_iteration: int = 1000,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the modified correlation matrix assuming Gaussian copula."""

    if order <= 1:
        msg = "The specified integration order " + str(order)
        msg += "must be greater than 1!"
        raise ValueError(msg)

    num_variables = len(marginal_distributions)
    herm_coords, herm_weights = np.polynomial.hermite.hermgauss(order)
    u1_coords, u2_coords = np.meshgrid(herm_coords, herm_coords)
    u1_coords, u2_coords = np.sqrt(2) * u1_coords, np.sqrt(2) * u2_coords
    weights = np.dot(np.transpose([herm_weights]), [herm_weights])
    std_norm = stats.norm(0.0, 1.0)
    z_mod_rho = np.eye(num_variables)
    for row_no in range(num_variables):
        row_var_marg = marginal_distributions[row_no]
        for col_no in range(row_no):
            col_var_marg = marginal_distributions[col_no]
            rho_x = correlation_matrix[row_no, col_no]
            rho_z = rho_x
            if np.abs(rho_x) < 0.05 or np.abs(rho_x) > 0.99:
                z_mod_rho[row_no, col_no] = rho_z
                continue
            if _is_normal_distribution(row_var_marg) and _is_normal_distribution(
                col_var_marg
            ):
                z_mod_rho[row_no, col_no] = rho_z
                continue

            rho_x_acc = np.inf
            rho_z_acc = np.inf
            denominator = np.pi * row_var_marg.std() * col_var_marg.std()
            for num_iteration in range(max_iteration):
                if rho_x_acc <= tolerance and rho_z_acc <= tolerance:
                    break

                rho_z_sqr = np.sqrt(1.0 - rho_z * rho_z)
                z1_coords = u1_coords
                z2_coords = rho_z * u1_coords + rho_z_sqr * u2_coords

                # Transform into the initial distribution space
                x1_coords = row_var_marg.ppf(std_norm.cdf(z1_coords))
                x2_coords = col_var_marg.ppf(std_norm.cdf(z2_coords))
                if not np.isfinite(x1_coords).all() or not np.isfinite(x2_coords).all():
                    break
                x1_stds = x1_coords - row_var_marg.mean()
                x2_stds = x2_coords - col_var_marg.mean()

                # Calculate the result of the integral as in C-Script
                rho_x_new = np.sum(x1_stds * x2_stds * weights) / denominator

                # Calculate derivative
                d_rho_x = u1_coords - rho_z * u2_coords / rho_z_sqr
                d_rho_x *= std_norm.pdf(z2_coords) / col_var_marg.pdf(x2_coords)
                d_rho_x = np.sum(d_rho_x * weights * x1_stds) / denominator

                # Evaluate the new rho_z and clip it to [-1,+1]
                rho_z_old = rho_z
                rho_z = rho_z_old + (rho_x - rho_x_new) / d_rho_x

                if not np.isfinite(rho_z):
                    rho_z = rho_z_old
                    break

                if np.abs(rho_z) > 1.0:
                    rho_z = 0.5 * (rho_z_old + np.sign(rho_z))

                if np.abs(rho_z) >= (1 - tolerance):
                    rho_z = rho_z_old
                    break

                # Calculate the accuracies
                rho_x_acc = np.abs(rho_x - rho_x_new)
                rho_z_acc = np.abs(rho_z - rho_z_old)

            if verbose and (rho_x_acc > tolerance or rho_z_acc > tolerance):
                msg = "Optimization not converged for"
                msg += "variables" + str(row_no) + "and"
                msg += str(col_no) + "."
                logging.warning(msg)
            z_mod_rho[row_no, col_no] = rho_z

    rho_u = z_mod_rho + np.tril(z_mod_rho, -1).T  # pylint: disable=no-member
    try:
        correlation_inducing_matrix = linalg.cholesky(rho_u, lower=False)
    except np.linalg.LinAlgError:
        if verbose:
            logging.warning(
                "Cholesky factorization failed the during Nataf transformation. "
                "Please check the passed correlation matrix for possible singularity. "
                "Proceeding with eigenvalue decomposition."
            )
        w_z, v_z = linalg.eigh(rho_u)
        correlation_inducing_matrix = np.dot(v_z, np.diag(np.sqrt(w_z))).T

    try:
        correlation_reducing_matrix = np.linalg.solve(
            correlation_inducing_matrix, np.eye(rho_u.shape[0])
        )
    except np.linalg.LinAlgError:
        if verbose:
            logging.warning(
                "Numpy-matrix inversion failed during the Nataf transformation. "
                "Please check the passed correlation matrix for possible singularity. "
                "Proceeding with pseudo-inverse, which may lead to unexpected behavior."
            )
        correlation_reducing_matrix = np.linalg.pinv(correlation_inducing_matrix)

    return correlation_inducing_matrix, correlation_reducing_matrix
