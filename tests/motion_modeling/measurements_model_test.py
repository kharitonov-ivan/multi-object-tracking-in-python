import numpy as np

from src.measurement_models import ConstantVelocityMeasurementModel


def test_constant_velocity_measurement_model():
    sigma_r = 0.5
    model = ConstantVelocityMeasurementModel(sigma_r)
    assert isinstance(model, ConstantVelocityMeasurementModel)
    state_vector = np.array([1.0, 2.0, 3.0, 4.0])
    expected_measurement = np.array([1.0, 2.0])
    assert np.allclose(model.h(state_vector), expected_measurement)
    assert np.allclose(model.H(state_vector), np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))
    assert np.allclose(model._noise_covariance, (sigma_r**2) * np.eye(2))
    assert str(model) == f"ConstantVelocityMeasurementModel(d=2, R={model.R}, "
