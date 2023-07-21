import time
import typing as tp

import numpy as np
import scipy
import scipy.stats


def multiple_logpdfs_vec_input(xs, means, covs):
    """`multiple_logpdfs` assuming `xs` has shape (N samples, P features)."""
    # NumPy broadcasts `eigh`.
    vals, vecs = np.linalg.eigh(covs)

    # Compute the log determinants across the second axis.
    logdets = np.sum(np.log(vals), axis=1)

    # Invert the eigenvalues.
    valsinvs = 1.0 / vals

    # Add a dimension to `valsinvs` so that NumPy broadcasts appropriately.
    Us = vecs * np.sqrt(valsinvs)[:, None]
    devs = xs[:, None, :] - means[None, :, :]

    # Use `einsum` for matrix-vector multiplications across the first dimension.
    devUs = np.einsum("jnk,nki->jni", devs, Us)

    # Compute the Mahalanobis distance by squaring each term and summing.
    mahas = np.sum(np.square(devUs), axis=2)

    # Compute and broadcast scalar normalizers.
    dim = xs.shape[1]
    log2pi = np.log(2 * np.pi)

    out = -0.5 * (dim * log2pi + mahas + logdets[None, :])
    return out.T


def vectorized_gaussian_logpdf(
    data_points: tp.Annotated[np.ndarray, "(n_points, point_dim)"],
    means: tp.Annotated[np.ndarray, "(n_gaussians, point_dim)"],
    covariances: tp.Annotated[np.ndarray, "(n_gaussians, point_dim, point_dim)"],
) -> tp.Annotated[np.ndarray, "N_points, N_gaussians"]:
    """
    Compute log N(x_i; mu_i, sigma_i) for each x_i, mu_i, sigma_i
    https://stackoverflow.com/a/50809036/8395138
    Args:
        X : shape (n, d)
            Data points
        means : shape (n, d)
            Mean vectors
        covariances : shape (n, d)
            Diagonal covariance matrices
    Returns:
        logpdfs : shape (n,)
            Log probabilities
    """
    # _, point_dim = data_points.shape
    # constant = point_dim * np.log(2 * np.pi)
    # log_determinants = np.log(np.prod(covariances, axis=1))
    # deviations:  NDArray[Shape["N_points, N_gaussians, Point_dim"], Float] = data_points[:, np.newaxis] - means
    # inverses = 1 / covariances
    # deviations_inverses = np.einsum('ijk,jkl->ijl', deviations, inverses)
    # import pdb; pdb.set_trace()
    # deviations_inverses_deviations = np.einsum('ijk,ikl->ijl', deviations_inverses, deviations)

    # return -0.5 * (constant + log_determinants + np.sum(deviations * inverses * deviations, axis=1))
    # расширяем размерности для возможности векторизованных операций
    # расширяем размерности для возможности векторизованных операций
    # data_points_exp = data_points[:, np.newaxis, :]
    # means_exp = means[np.newaxis, :, :]

    # # вычисляем разности между точками и средними значениями
    # diff = data_points_exp - means_exp

    # # вычисляем обратные матрицы ковариации
    # inv_covariances = np.linalg.inv(covariances)

    # # вычисляем многомерное Гауссово распределение с использованием векторизованных операций
    # exponent = np.einsum('...i,ij,...j->...', diff, -0.5 * inv_covariances, diff)
    # norm_factor = 0.5 * np.sum(np.log(np.linalg.det(2 * np.pi * covariances)), axis=-1)

    # # возвращаем суммарную логарифмическую плотность вероятности
    # return np.sum(exponent + norm_factor)
    n_gaussians = means.shape[0]
    n_data_points = data_points.shape[0]

    result = multiple_logpdfs_vec_input(data_points, means, covariances)

    # result = np.zeros((n_gaussians, n_data_points))
    # for j in range(n_gaussians):
    #     result[j] = scipy.stats.multivariate_normal.logpdf(data_points, means[j], covariances[j])
    return result


def make_positive_definite(tensor):
    return 0.5 * (tensor + tensor.transpose(0, 2, 1))


if __name__ == "__main__":
    n_gaussians = 256
    state_dim = 2
    n_data_points = 128

    means = np.random.randn(n_gaussians, state_dim)
    covariances = np.random.randn(n_gaussians, state_dim, state_dim)
    data_points = np.random.randn(n_data_points, state_dim)

    for i in range(n_gaussians):
        covariances[i] = np.eye(state_dim) + 1e-3

    reference_array = np.zeros((n_gaussians, n_data_points))
    ref_start = time.time()
    for j in range(n_gaussians):
        reference_array[j] = scipy.stats.multivariate_normal.logpdf(data_points, means[j], covariances[j])
    ref_time = time.time() - ref_start
    import pdb

    pdb.set_trace()
    fast_start = time.time()
    my_array = vectorized_gaussian_logpdf(data_points, means, covariances)
    fast_time = time.time() - fast_start

    assert np.allclose(my_array, reference_array)

    print("Reference time:", ref_time)
    print("Vectorized time:", fast_time)
    print("Speedup:", ref_time / fast_time)
