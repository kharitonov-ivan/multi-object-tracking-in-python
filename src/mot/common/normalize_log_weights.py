import numpy as np
import nptyping as npt


def normalize_log_weights(
    log_weights: npt.NDArray[npt.Shape["N_components"], npt.Float],
    axis=None,
) -> tuple[npt.NDArray[npt.Shape["N_components"], npt.Float], float]:
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
    # np.logaddexp.reduce performs a pairwise reduce operation using logaddexp.
    # It is a stable way to sum the exponentiated log_weights and take the log of the sum.
    # The logaddexp function computes log(exp(a) + exp(b)) in a way that is numerically stable.
    # Thus, log_sum_w is equivalent to log(sum(exp(log_weights))) but computed in a way that is more numerically stable.
    log_sum_w = np.logaddexp.reduce(log_weights, axis=axis)

    # Subtracting log_sum_w from log_weights normalizes the weights in log space,
    # so that when exponentiated, they will sum to 1.
    normalized_log_weights = log_weights - log_sum_w

    return normalized_log_weights, log_sum_w
