import os

import joblib
import numpy as np
import pytest

import uncertainty_propagation.integrator as module_under_test


def valid_base_function_1(x: np.ndarray) -> np.ndarray:
    return np.sum(2 * x, axis=1, keepdims=True)


def valid_base_function_2(x: np.ndarray) -> np.ndarray:
    return np.min(x**2, axis=1)


def invalid_base_function(x: np.ndarray) -> np.ndarray:
    # invalid because returns multi dimensional output
    return 2 * x


def test_transform_to_zero_centered_envelope_single_function():
    np.random.seed(1337)
    test_fun_1 = module_under_test.transform_to_zero_centered_envelope(
        valid_base_function_1, limit=1
    )
    x = np.random.randn(20, 2)
    expected = valid_base_function_1(x)
    result, x_, y_ = test_fun_1(x)
    assert np.isclose(x_, x).all()
    assert np.isclose(y_, expected).all()
    assert np.isclose(result, expected.reshape(-1) - 1).all()

    test_fun_2 = module_under_test.transform_to_zero_centered_envelope(
        invalid_base_function, limit=1
    )

    with pytest.raises(ValueError):
        test_fun_2(x)


def test_transform_to_zero_centered_envelope_multiple_functions():
    np.random.seed(1337)
    test_fun = module_under_test.transform_to_zero_centered_envelope(
        [valid_base_function_1, valid_base_function_2], limit=2
    )
    x = np.random.randn(20, 2)
    expected = np.c_[valid_base_function_1(x).reshape(-1), valid_base_function_2(x)]
    result, x_, y_ = test_fun(x)
    assert np.isclose(x_, x).all()
    assert np.isclose(y_, expected).all()
    assert np.isclose(result, np.min(expected, axis=1) - 2).all()


def test_transform_to_zero_centered_multiprocessed_caching():
    np.random.seed(1337)
    test_fun = module_under_test.transform_to_zero_centered_envelope(
        valid_base_function_1, limit=1
    )
    x = np.random.randn(32, 2)
    expected = valid_base_function_1(x)
    n_jobs = min(4, os.cpu_count())
    block_size = 32 // n_jobs
    slices = [slice(i * block_size, (i + 1) * block_size) for i in range(n_jobs)]
    all_results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(test_fun)(x[sli]) for sli in slices
    )
    result = np.hstack([r[0] for r in all_results])
    x_ = np.vstack([r[1] for r in all_results])
    y_ = np.vstack([r[2] for r in all_results])

    assert np.isclose(x_, x).all()
    assert np.isclose(y_, expected).all()
    assert np.isclose(result, expected.reshape(-1) - 1).all()
