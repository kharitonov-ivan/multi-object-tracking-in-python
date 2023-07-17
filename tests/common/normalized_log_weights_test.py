import numpy as np

from mot.common.normalize_log_weights import normalize_log_weights


def test_normalize_log_weights_single_element():
    # Test case with a single weight
    test_weights = np.array([10.0])
    expected_log_sum = 10.0
    eps = 1e-4

    actual_log_norm_weights, actual_log_sum = normalize_log_weights(test_weights)

    assert np.abs(actual_log_sum - expected_log_sum) < eps, f"check log sum calculation: " f"actual_log_sum = {actual_log_sum}"


def test_normalize_log_weights_multiple_elements():
    # Test case with multiple weights
    test_weights = np.array([10.0, 20.0, 30.0])
    expected_log_sum = np.logaddexp.reduce(test_weights)
    eps = 1e-4

    actual_log_norm_weights, actual_log_sum = normalize_log_weights(test_weights)

    # Check that the log sum is calculated correctly
    assert np.abs(actual_log_sum - expected_log_sum) < eps, f"check log sum calculation: " f"actual_log_sum = {actual_log_sum}"

    # Check that the normalized weights sum to 1 when exponentiated
    norm_weights_sum = np.sum(np.exp(actual_log_norm_weights))
    assert np.abs(norm_weights_sum - 1.0) < eps, f"check normalized weights sum to 1: " f"norm_weights_sum = {norm_weights_sum}"
