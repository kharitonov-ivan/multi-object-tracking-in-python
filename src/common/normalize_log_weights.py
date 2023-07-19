import typing as tp

import numpy as np


def normalize_log_weights(
    log_weights: tp.Annotated[np.ndarray, "n_components"],
    axis=None,
) -> tuple[tp.Annotated[np.ndarray, "n_components"], float]:
    """
    Normalize a list of log weights such that when the weights are exponentiated, they sum up to 1.

    This operation is useful when dealing with weights represented in the log domain, typically probabilities
    or likelihoods, where due to numerical precision, direct exponentiation could result in underflow or overflow.

    The normalization is done in the log domain for numerical stability.

    Args:
        log_weights (np.ndarray): array of log weights, e.g. log probabilities or likelihoods.
        axis: Axis along which the weights should be normalized. Default is None, in which case weights
            will be flattened before normalization.

    Returns:
        normalized_log_weights (np.ndarray): log of the normalized weights
        log_sum_w (float): log of the sum of the non-normalized weights
    """
    # compute the log of the sum of the unnormalized weights
    log_sum_w = np.logaddexp.reduce(log_weights, axis=axis)

    # Subtracting log_sum_w from log_weights normalizes the weights in log space,
    # so that when exponentiated, they will sum to 1.
    normalized_log_weights = log_weights - log_sum_w

    return normalized_log_weights, log_sum_w
