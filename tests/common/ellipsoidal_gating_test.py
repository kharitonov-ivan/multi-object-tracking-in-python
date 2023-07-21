import numpy as np
from scipy.stats import chi2

from src.common.gaussian_density import GaussianDensity
from src.measurement_models import ConstantVelocityMeasurementModel


def test_ellipsoidal_gating_known_values():
    # Test setup
    means = np.array([[10.0, 20.0, 30.0, 40.0]])  # (1, 4)
    covs = np.array(
        [
            [
                [2.0, 0.5, 0.1, 0.2],
                [0.5, 3.0, 0.1, 0.2],
                [0.1, 0.1, 2.0, 0.3],
                [0.2, 0.2, 0.3, 4.0],
            ]
        ]
    )  # (1, 4, 4)
    measurements = np.array([[11.0, 22.0], [12.0, 24.0]])  # ( 2, 2)
    measurement_model = ConstantVelocityMeasurementModel(sigma_r=0.1)
    confidence_level = 0.99

    # Perform gating
    meas_mask, mah_dists = GaussianDensity.ellipsoidal_gating(GaussianDensity(means, covs), measurements, measurement_model, confidence_level)

    # Given the known inputs, check that the outputs have the expected values
    assert meas_mask.all() == True, "At least one measurement should be inside the gate"
    assert np.all(mah_dists < chi2.ppf(confidence_level, df=measurement_model.dim)), "All measurements should be outside the gate"


def test_ellipsoidal_gating_known_values_outside_gate():
    # Test setup
    means = np.array([[10.0, 20.0, 30.0, 40.0]])  # (1, 4)
    covs = np.array(
        [
            [
                [2.0, 0.5, 0.1, 0.2],
                [0.5, 3.0, 0.1, 0.2],
                [0.1, 0.1, 2.0, 0.3],
                [0.2, 0.2, 0.3, 4.0],
            ]
        ]
    )  # (1, 4, 4)
    measurements = np.array([[100.0, 200.0], [200.0, 400.0]])  # (2, 2) - Measurements are far from the means
    measurement_model = ConstantVelocityMeasurementModel(sigma_r=0.1)
    confidence_level = 0.99

    # Perform gating
    meas_mask, mah_dists = GaussianDensity.ellipsoidal_gating(GaussianDensity(means, covs), measurements, measurement_model, confidence_level)

    # Given the known inputs, check that the outputs have the expected values
    assert meas_mask.all() == False, "No measurement should be inside the gate"
    assert np.all(mah_dists > chi2.ppf(confidence_level, df=measurement_model.dim)), "All measurements should be outside the gate"
